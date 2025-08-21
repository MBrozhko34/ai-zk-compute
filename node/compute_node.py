#!/usr/bin/env python3
"""Join lobby, wait for start, claim task, train XOR, prove (Groth16),
auto-settlement on last proof, then withdraw payout — with robust error handling (fixed send_tx API)."""
import json, os, subprocess, tempfile, pathlib, time, hashlib
from typing import Optional

from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount
import torch, torch.nn as nn


# ───────── env
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0) or 0)

ZKEY      = pathlib.Path("../circuits/xor_final.zkey")
VK_JSON   = pathlib.Path("../circuits/verification_key.json")
CIRC_DIR  = pathlib.Path("../circuits/XorCircuit_js")
ABI_PATH  = pathlib.Path("../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"


# ───────── helpers (mirror circuit)
def bin01_list(tensor_flat):
    # IMPORTANT: no rounding and strict > 0.0 threshold
    return [1 if float(v) > 0.0 else 0 for v in tensor_flat]

clamp3 = lambda x: min(max(int(x), 0), 3)    # 0..3 for Step()

def xor_forward(x0, x1, wIH, bH, bO):
    h0 = clamp3(wIH[0]*x0 + wIH[1]*x1 + bH[0])
    h1 = clamp3(wIH[2]*x0 + wIH[3]*x1 + bH[1])
    s0 = 1 if h0 >= 1 else 0
    s1 = 1 if h1 >= 1 else 0
    o  = clamp3(s0 - s1 + bO)
    return 1 if o >= 1 else 0


# ───────── chain handles
w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)
orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR),
                       abi=json.load(open(ABI_PATH))["abi"])


# ───────── error decoding utilities
ERROR_STRING_SELECTOR = bytes.fromhex("08c379a0")  # Error(string)
PANIC_SELECTOR        = bytes.fromhex("4e487b71")  # Panic(uint256)
REENTRANCY_SELECTOR   = bytes.fromhex("3ee5aeb5")  # ReentrancyGuardReentrantCall()

def _decode_error_string(data_hex: str) -> Optional[str]:
    try:
        if not data_hex or not data_hex.startswith("0x"):
            return None
        b = bytes.fromhex(data_hex[2:])
        if len(b) < 4:
            return None
        selector = b[:4]
        if selector == ERROR_STRING_SELECTOR:
            if len(b) >= 4 + 32 + 32:
                strlen = int.from_bytes(b[4+32:4+32+32], "big")
                start  = 4 + 32 + 32
                end    = start + strlen
                if end <= len(b):
                    return b[start:end].decode("utf-8", errors="replace")
        elif selector == PANIC_SELECTOR:
            return "panic()"
        elif selector == REENTRANCY_SELECTOR:
            return "ReentrancyGuardReentrantCall()"
        return None
    except Exception:
        return None

def explain_web3_error(e: Exception) -> str:
    if isinstance(e, ValueError) and e.args and isinstance(e.args[0], dict):
        err = e.args[0]
        msg = err.get("message", "")
        data = err.get("data")
        for needle in [
            "reverted with reason string",
            "execution reverted",
            "Transaction ran out of gas",
            "nonce too low",
            "replacement transaction underpriced",
            "already known",
        ]:
            if needle in msg:
                if "reverted with reason string" in msg:
                    tail = msg.split("reverted with reason string")[-1].strip()
                    return f"revert {tail.strip(': ').strip()}"
                return msg
        if isinstance(data, dict):
            s = _decode_error_string(data.get("data") or "")
            if s: return f"revert '{s}'"
        elif isinstance(data, str):
            s = _decode_error_string(data)
            if s: return f"revert '{s}'"
        return msg or str(e)
    return str(e)


# ───────── tx helpers (ContractFunction only)
def next_nonce() -> int:
    return w3.eth.get_transaction_count(acct.address, "pending")

def send_tx(fn_call, *, min_gas: int = 200_000, headroom_num: int = 15, headroom_den: int = 10):
    """
    Send a ContractFunction:
      - estimate gas (with headroom), fall back to generous default if needed
      - sign, send, wait
      - basic retries for nonce/gossip races
    """
    # estimate gas with headroom
    try:
        est = int(fn_call.estimate_gas({"from": acct.address}))
        gas_limit = max(min_gas, est * headroom_num // headroom_den)
    except Exception as e:
        print("   gas estimation failed, defaulting:", explain_web3_error(e))
        gas_limit = max(min_gas, 2_500_000)

    n_retries = 0
    while True:
        try:
            tx = fn_call.build_transaction({
                "from": acct.address,
                "gas":  gas_limit,
                "gasPrice": w3.eth.gas_price,
                "nonce": next_nonce(),
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.rawTransaction)
            return w3.eth.wait_for_transaction_receipt(h)
        except Exception as e:
            msg = explain_web3_error(e)
            if any(x in msg for x in ["nonce too low", "already known", "replacement transaction underpriced"]):
                n_retries += 1
                if n_retries <= 3:
                    time.sleep(0.5)
                    continue
            if "ran out of gas" in msg.lower() or "intrinsic gas too low" in msg.lower():
                raise RuntimeError(f"transaction OOG — try higher gas. Details: {msg}") from e
            raise RuntimeError(msg) from e


# ───────── workflow
addr = acct.address
print(f"[worker:{addr}] joining lobby for request {REQ_ID}…")

# Early exit if closed
try:
    closed, *_ = orch.functions.getResult(REQ_ID).call()
    if closed:
        print("   request is already closed; exiting.")
        raise SystemExit(0)
except Exception as e:
    print("   warning: getResult() failed (continuing):", explain_web3_error(e))

# 0) join the lobby
try:
    if orch.functions.joined(REQ_ID, addr).call():
        print("   already joined lobby.")
    else:
        send_tx(orch.functions.joinLobby(REQ_ID), min_gas=200_000)
        print("   ✓ joined lobby")
except RuntimeError as e:
    msg = str(e)
    if "already joined" in msg:
        print("   already joined lobby.")
    elif "started" in msg:
        print("   lobby already started; cannot join. Exiting.")
        raise SystemExit(0)
    elif "lobby full" in msg:
        print("   lobby is full; cannot join. Exiting.")
        raise SystemExit(0)
    elif "closed" in msg:
        print("   request is closed; cannot join. Exiting.")
        raise SystemExit(0)
    else:
        print("   joinLobby failed:", msg); raise SystemExit(1)

# 0b) wait until ready
while True:
    try:
        needed, joined, ready = orch.functions.lobbyCounts(REQ_ID).call()
        print(f"   lobby: {joined}/{needed} ready={ready}")
        if ready: break
        time.sleep(3)
    except Exception as e:
        print("   lobbyCounts() failed (retrying):", explain_web3_error(e))
        time.sleep(2)

# 1) claim assigned task
try:
    send_tx(orch.functions.claimTask(REQ_ID), min_gas=200_000)
    print("   ✓ claimed task")
except RuntimeError as e:
    msg = str(e)
    if "not ready" in msg:
        print("   lobby not ready; should not happen now. Exiting."); raise SystemExit(1)
    if "not in lobby" in msg:
        print("   not in lobby; cannot claim. Exiting."); raise SystemExit(1)
    if "index taken" in msg:
        print("   task index already claimed; exiting."); raise SystemExit(0)
    if "closed" in msg:
        print("   request closed before claim; exiting."); raise SystemExit(0)
    print("   claimTask failed:", msg); raise SystemExit(1)

# resolve own task/hp
try:
    idx = orch.functions.taskOf(REQ_ID, addr).call()
    lr_ppm, steps = orch.functions.getSpace(REQ_ID).call()[idx]
    print(f"[worker:{addr}] idx={idx}, lr_ppm={lr_ppm}, steps={steps}")
except Exception as e:
    print("   could not fetch assigned HP:", explain_web3_error(e))
    raise SystemExit(1)

# deterministic seed per HP
seed_src = f"{lr_ppm}:{steps}".encode()
seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:8], "big")
torch.manual_seed(seed)

# 2) train small MLP on XOR
X = torch.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

net = nn.Sequential(nn.Linear(2,2), nn.Sigmoid(),
                    nn.Linear(2,1), nn.Sigmoid())
opt = torch.optim.SGD(net.parameters(), lr_ppm/1_000_000)
crit = nn.BCELoss()
for _ in range(int(steps)):
    opt.zero_grad(); crit(net(X),y).backward(); opt.step()

# 3) binarise & circuit-compatible accuracy
w_flat = net[0].weight.detach().flatten().tolist()
b_flat = net[0].bias.detach().flatten().tolist()
wIH = bin01_list(w_flat)          # 4 ints in {0,1}
bH  = bin01_list(b_flat)          # 2 ints in {0,1}
bO  = 1                           # avoid negatives

preds   = [xor_forward(*xy, wIH, bH, bO) for xy in [(0,0),(1,0),(0,1),(1,1)]]
correct = sum(int(p==t) for p,t in zip(preds, [0,1,1,0]))
acc_bps = correct * 2500
print(f"   circuit-compatible acc = {correct}/4 → {acc_bps/100}%")

# 4) witness & proof
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    (tmp/"input.json").write_text(json.dumps({
        "lr": lr_ppm, "steps": steps, "acc_bps": acc_bps,
        "wIH": wIH, "bH":  bH, "bO":  bO
    }, indent=2))

    try:
        subprocess.check_call([
            "node", str(CIRC_DIR/"generate_witness.js"),
            str(CIRC_DIR/"XorCircuit.wasm"),
            str(tmp/"input.json"),
            str(tmp/"witness.wtns")
        ])
        subprocess.check_call([
            "snarkjs", "groth16", "prove",
            str(ZKEY), str(tmp/"witness.wtns"),
            str(tmp/"proof.json"), str(tmp/"public.json")
        ])
        print("   verifying off-chain with snarkjs…")
        subprocess.check_call([
            "snarkjs", "groth16", "verify",
            str(VK_JSON), str(tmp/"public.json"), str(tmp/"proof.json")
        ])
        print("   ✓ off-chain verification OK")
    except subprocess.CalledProcessError as e:
        print("   snarkjs failed with code", e.returncode); raise SystemExit(1)

    proof          = json.loads((tmp/"proof.json").read_text())
    public_signals = [int(x) for x in json.loads((tmp/"public.json").read_text())]

# 5) format proof for Solidity (swap inner limbs of B) & submit
a = [int(proof["pi_a"][0]), int(proof["pi_a"][1])]
b = [
    [int(proof["pi_b"][0][1]), int(proof["pi_b"][0][0])],
    [int(proof["pi_b"][1][1]), int(proof["pi_b"][1][0])]
]
c = [int(proof["pi_c"][0]), int(proof["pi_c"][1])]
print("   pubSignals (lr,steps,acc_bps):", public_signals)

try:
    # Heavy path when last worker triggers settlement → use larger min_gas
    send_tx(orch.functions.submitProof(REQ_ID, a, b, c, public_signals), min_gas=1_200_000)
    print("   ✓ proof submitted")
except RuntimeError as e:
    msg = str(e)
    if "bad proof" in msg:
        print("   ❌ on-chain verifier rejected the proof: bad proof"); raise SystemExit(1)
    if "already proven" in msg:
        print("   task already proven; moving to settlement watch.")
    elif "task not yours" in msg or "wrong index" in msg:
        print("   not authorized for this hp index; exiting."); raise SystemExit(1)
    elif "not started" in msg or "closed" in msg:
        print(f"   cannot submit: {msg}; exiting."); raise SystemExit(0)
    else:
        print("   submitProof failed:", msg); raise SystemExit(1)

# 6) wait for auto-settlement & withdraw
while True:
    try:
        closed, bestAcc, winnersCount, perWinnerWei, perLoserWei = orch.functions.getResult(REQ_ID).call()
        if closed:
            winners = [orch.functions.winnerAt(REQ_ID, i).call() for i in range(winnersCount)]
            my_credit = orch.functions.getCredit(REQ_ID, addr).call()
            print(f"[worker:{addr}] settled! bestAcc={bestAcc/100:.2f}% winners={winnersCount} "
                  f"perWinner={w3.from_wei(perWinnerWei,'ether')} ETH perLoser={w3.from_wei(perLoserWei,'ether')} ETH")
            print("   winners:", winners)
            if my_credit > 0:
                print(f"   Your credit: {w3.from_wei(my_credit, 'ether')} ETH → withdrawing…")
                try:
                    send_tx(orch.functions.withdraw(REQ_ID), min_gas=120_000)
                    print("   ✓ withdrawn")
                except RuntimeError as e:
                    msg = str(e)
                    if "no credit" in msg:
                        print("   race: no credit at withdraw time.")
                    else:
                        print("   withdraw failed:", msg)
                finally:
                    break
            else:
                print("   No payout for this address.")
                break
        print("   waiting for settlement…")
        time.sleep(3)
    except Exception as e:
        print("   getResult/winnerAt/getCredit failed (retrying):", explain_web3_error(e))
        time.sleep(3)

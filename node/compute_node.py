#!/usr/bin/env python3
"""Join lobby, wait for start (minWorkers), claim tasks with bond, train XOR, prove (Groth16),
auto-settlement on last proof, reclaim timed-out tasks using chain time, then withdraw payout."""
import json, os, subprocess, tempfile, pathlib, time, hashlib
from typing import Optional, List

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

# ---- robust parser for snarkjs calldata (handles "two arrays on one line") ----
def _to_int(x):
    if isinstance(x, int):
        return x
    s = str(x)
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    return int(s)

def parse_plonk_calldata_arrays(txt: str):
    """
    Returns (proof24, pub3) as lists of ints.
    Works whether snarkjs prints:
      - two arrays on separate lines, OR
      - both arrays on a single line.
    """
    # collect all top-level JSON arrays by bracket depth
    arr_strings = []
    depth = 0
    start = -1
    for i, ch in enumerate(txt):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    arr_strings.append(txt[start:i+1])
                    start = -1

    # fallbacks (just in case)
    if len(arr_strings) < 2:
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                arr_strings.append(line)

    if len(arr_strings) < 2:
        raise RuntimeError(f"Unexpected soliditycalldata output: {txt[:200]}...")

    proof_arr  = [_to_int(x) for x in json.loads(arr_strings[0])]
    pub_signals = [_to_int(x) for x in json.loads(arr_strings[1])]

    if len(proof_arr) != 24:
        raise RuntimeError(f"Proof array length {len(proof_arr)} != 24")
    if len(pub_signals) != 3:
        raise RuntimeError(f"pubSignals length {len(pub_signals)} != 3")

    return proof_arr, pub_signals


# ───────── helpers (mirror circuit)
def bin01_list(tensor_flat):
    return [1 if float(v) > 0.0 else 0 for v in tensor_flat]

clamp3 = lambda x: min(max(int(x), 0), 3)
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

# after creating `w3` and `acct` and before any txs:
addr = acct.address
bal0 = int(w3.eth.get_balance(addr))
print(f"[worker:{addr}] starting on request {REQ_ID}…")
print(f"   balance before: {w3.from_wei(bal0,'ether')} ETH")

# ───────── err decode
def _decode_error_string(data_hex: str) -> Optional[str]:
    try:
        if not data_hex or not data_hex.startswith("0x"):
            return None
        b = bytes.fromhex(data_hex[2:])
        if len(b) < 4: return None
        sel = b[:4]
        if sel == bytes.fromhex("08c379a0"):
            if len(b) >= 4 + 32 + 32:
                strlen = int.from_bytes(b[4+32:4+32+32], "big")
                s = b[4+32+32:4+32+32+strlen]
                return s.decode("utf-8", errors="replace")
        elif sel == bytes.fromhex("4e487b71"):
            return "panic()"
        elif sel == bytes.fromhex("3ee5aeb5"):
            return "ReentrancyGuardReentrantCall()"
        return None
    except Exception:
        return None

def explain_web3_error(e: Exception) -> str:
    if isinstance(e, ValueError) and e.args and isinstance(e.args[0], dict):
        err = e.args[0]; msg = err.get("message",""); data = err.get("data")
        if isinstance(data, dict):
            s = _decode_error_string(data.get("data") or "")
            if s: return f"revert '{s}'"
        elif isinstance(data, str):
            s = _decode_error_string(data)
            if s: return f"revert '{s}'"
        return msg or str(e)
    return str(e)

# ───────── tx helpers
def next_nonce() -> int:
    return w3.eth.get_transaction_count(acct.address, "pending")

def send_tx(fn_call, *, value: int = 0, min_gas: int = 200_000, headroom_num: int = 15, headroom_den: int = 10):
    try:
        est = int(fn_call.estimate_gas({"from": acct.address, "value": value}))
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
                "value": value,
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
                raise RuntimeError(f"OOG — try higher gas. Details: {msg}") from e
            raise RuntimeError(msg) from e

# ───────── small helpers
def chain_now() -> int:
    return int(w3.eth.get_block("latest")["timestamp"])

def space_len() -> int:
    return len(orch.functions.getSpace(REQ_ID).call())

def count_proven(n: int) -> int:
    c = 0
    for i in range(n):
        if orch.functions.taskAcc(REQ_ID, i).call() != 0:
            c += 1
    return c

# ───────── workflow
addr = acct.address
print(f"[worker:{addr}] starting on request {REQ_ID}…")

# Early exit if already closed
try:
    closed, *_ = orch.functions.getResult(REQ_ID).call()
    if closed:
        print("   request is already closed; exiting.")
        raise SystemExit(0)
except Exception as e:
    print("   warning: getResult() failed (continuing):", explain_web3_error(e))

# bond / ttl
try:
    bond_wei = int(orch.functions.CLAIM_BOND_WEI().call())
    ttl_sec  = int(orch.functions.CLAIM_TTL().call())
except Exception:
    bond_wei = int(Web3.to_wei(0.005, "ether")); ttl_sec = 10

# 0) join
try:
    if orch.functions.joined(REQ_ID, addr).call():
        print("   already joined lobby.")
    else:
        # preflight join
        orch.functions.joinLobby(REQ_ID).call({"from": addr})
        send_tx(orch.functions.joinLobby(REQ_ID))
        print("   ✓ joined lobby")
except Exception as e:
    msg = explain_web3_error(e)
    if "already joined" in msg: print("   already joined lobby.")
    elif "closed" in msg:      print("   request closed; exiting."); raise SystemExit(0)
    else:                      print("   joinLobby failed:", msg);  raise SystemExit(1)

# 0b) wait for start
while True:
    try:
        needed, joined, ready = orch.functions.lobbyCounts(REQ_ID).call()
        print(f"   lobby: {joined}/{needed} ready={ready}")
        if ready: break
        time.sleep(1.5)
    except Exception as e:
        print("   lobbyCounts() failed:", explain_web3_error(e)); time.sleep(1.5)

# Train/prove/submit helper – PLONK (fixed arrays)
def do_task(idx: int, lr_ppm: int, steps: int):
    seed = int.from_bytes(hashlib.sha256(f"{lr_ppm}:{steps}".encode()).digest()[:8], "big")
    torch.manual_seed(seed)

    X = torch.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

    net = nn.Sequential(nn.Linear(2,2), nn.Sigmoid(),
                        nn.Linear(2,1), nn.Sigmoid())
    opt = torch.optim.SGD(net.parameters(), lr_ppm/1_000_000)
    crit = nn.BCELoss()
    for _ in range(int(steps)):
        opt.zero_grad(); crit(net(X),y).backward(); opt.step()

    wIH = bin01_list(net[0].weight.detach().flatten().tolist())
    bH  = bin01_list(net[0].bias.detach().flatten().tolist())
    bO  = 1
    preds   = [xor_forward(*xy, wIH, bH, bO) for xy in [(0,0),(1,0),(0,1),(1,1)]]
    correct = sum(int(p==t) for p,t in zip(preds, [0,1,1,0]))
    acc_bps = correct * 2500
    print(f"   [task {idx}] acc = {correct}/4 → {acc_bps/100}%")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp/"input.json").write_text(json.dumps({
            "lr": lr_ppm, "steps": steps, "acc_bps": acc_bps,
            "wIH": wIH, "bH": bH, "bO": bO
        }, indent=2))

        # witness
        subprocess.check_call([
            "node", str(CIRC_DIR/"generate_witness.js"),
            str(CIRC_DIR/"XorCircuit.wasm"),
            str(tmp/"input.json"),
            str(tmp/"witness.wtns")
        ])

        # PLONK prove + verify
        subprocess.check_call([
            "snarkjs", "plonk", "prove",
            str(ZKEY), str(tmp/"witness.wtns"),
            str(tmp/"proof.json"), str(tmp/"public.json")
        ])
        print("   verifying off-chain with snarkjs…")
        subprocess.check_call([
            "snarkjs", "plonk", "verify",
            str(VK_JSON), str(tmp/"public.json"), str(tmp/"proof.json")
        ])
        print("   ✓ off-chain verification OK")

        # Export calldata in ARRAY form (uint256[24], uint256[3])
        raw = subprocess.check_output([
            "snarkjs", "zkey", "export", "soliditycalldata",
            str(tmp/"public.json"), str(tmp/"proof.json")
        ]).decode()

    # Robustly parse even when snarkjs prints two arrays on one line
    proof_arr, public_signals = parse_plonk_calldata_arrays(raw)

    # Submit to the contract (fits uint256[24], uint256[3])
    send_tx(orch.functions.submitProof(REQ_ID, proof_arr, public_signals),
            min_gas=1_200_000)
    print(f"   ✓ submitted proof for task {idx}")

# Claim one task (preflight & parse event)
def try_claim_one() -> Optional[int]:
    # preflight: skip if it would revert
    try:
        orch.functions.claimTask(REQ_ID).call({"from": addr, "value": bond_wei})
    except Exception as e:
        msg = explain_web3_error(e)
        if "no tasks left" in msg or "closed" in msg or "not started" in msg or "bond" in msg:
            return None
        # unknown issue
        print("   claimTask preflight failed:", msg)
        return None

    # send TX
    rcpt = send_tx(orch.functions.claimTask(REQ_ID), value=bond_wei, min_gas=230_000)
    # parse event for idx
    evs = orch.events.TaskClaimed().process_receipt(rcpt)
    if evs:
        idx = int(evs[-1]["args"]["idx"])
        return idx

    # fallback (shouldn’t hit; kept for safety)
    n = space_len()
    for i in range(n):
        owner = orch.functions.taskOwner(REQ_ID, i).call()
        acc   = orch.functions.taskAcc(REQ_ID, i).call()
        if owner == addr and acc == 0:
            return i
    return None

ZERO = "0x0000000000000000000000000000000000000000"

def maybe_reassign_timeouts():
    try:
        n = space_len()
        proven = count_proven(n)
        now = chain_now()

        # If exactly one task is left unproven, try immediate reassign on it.
        if proven == n - 1:
            for i in range(n):
                acc = orch.functions.taskAcc(REQ_ID, i).call()
                if acc != 0:
                    continue
                owner = orch.functions.taskOwner(REQ_ID, i).call()
                if owner == ZERO:
                    # It's already unassigned; outer loop will be able to claim it.
                    continue
                # Fast-path: contract will allow this (skips TTL & majority for last task).
                try:
                    # Preflight
                    orch.functions.reassignTimedOut(REQ_ID, i).call({"from": addr})
                except Exception:
                    pass
                else:
                    try:
                        send_tx(orch.functions.reassignTimedOut(REQ_ID, i), min_gas=180_000)
                        print(f"   reassignTimedOut(last, {i})")
                    except Exception:
                        pass
            return  # done; outer loop will attempt claim immediately

        # Normal path: majority proven + TTL elapsed per task.
        # (Contract still enforces both; we only preflight to avoid noisy reverts.)
        for i in range(n):
            acc = orch.functions.taskAcc(REQ_ID, i).call()
            if acc != 0:
                continue
            owner = orch.functions.taskOwner(REQ_ID, i).call()
            if owner == ZERO:
                continue
            t = orch.functions.claimedAt(REQ_ID, i).call()
            if t == 0:
                continue
            remain = (t + ttl_sec) - now
            if remain > 0:
                # Keep it informative but not chatty.
                print(f"   waiting TTL for idx {i}… {remain}s")
                continue

            # Preflight reassign
            try:
                orch.functions.reassignTimedOut(REQ_ID, i).call({"from": addr})
            except Exception:
                continue
            try:
                send_tx(orch.functions.reassignTimedOut(REQ_ID, i), min_gas=180_000)
                print(f"   reassignTimedOut({i})")
            except Exception:
                pass
    except Exception:
        pass

# Main loop
while True:
    claimed_any = False
    while True:
        idx = try_claim_one()
        if idx is None:
            break
        claimed_any = True
        lr_ppm, steps = orch.functions.getSpace(REQ_ID).call()[idx]
        print(f"   claimed task {idx} (lr={lr_ppm}, steps={steps}) with bond {w3.from_wei(bond_wei,'ether')} ETH")
        try:
            do_task(int(idx), int(lr_ppm), int(steps))
        except subprocess.CalledProcessError as e:
            print(f"   snark/proof failed for task {idx}; it will time out and be reassignable.")
        except Exception as e:
            print(f"   task {idx} failed: {e}")

    # closed?
    closed, bestAcc, winTaskCount, perWinTaskWei, perLoseTaskWei = orch.functions.getResult(REQ_ID).call()
    if closed:
        break

    # Help free stale tasks only when we didn’t claim anything this pass
    if not claimed_any:
        maybe_reassign_timeouts()

    time.sleep(1.5)

# Settlement summary
closed, bestAcc, winTaskCount, perWinTaskWei, perLoseTaskWei = orch.functions.getResult(REQ_ID).call()
space = orch.functions.getSpace(REQ_ID).call()

my_total = my_proven = my_win = my_lose = 0
for i in range(len(space)):
    if orch.functions.taskOwner(REQ_ID, i).call() == addr:
        my_total += 1
        acc = int(orch.functions.taskAcc(REQ_ID, i).call())
        if acc != 0:
            my_proven += 1
            if acc == int(bestAcc): my_win += 1
            else:                   my_lose += 1

my_credit = int(orch.functions.credit(REQ_ID, addr).call())  # direct mapping getter
bond_refund = my_proven * bond_wei
reward_expected = my_win * int(perWinTaskWei) + my_lose * int(perLoseTaskWei)
note = "" if (bond_refund + reward_expected) == my_credit else " (incl. remainder)"

print(f"[worker:{addr}] settled! bestAcc={bestAcc/100:.2f}% "
      f"winnerTasks={winTaskCount} perWinnerTask={w3.from_wei(perWinTaskWei,'ether')} ETH "
      f"perLoserTask={w3.from_wei(perLoseTaskWei,'ether')} ETH")
print(f"   my tasks: total={my_total}, proven={my_proven}, wins={my_win}, loses={my_lose}")
print(f"   bond refunds: {w3.from_wei(bond_refund,'ether')} ETH")
print(f"   rewards:      {w3.from_wei(reward_expected,'ether')} ETH{note}")
print(f"   total credit: {w3.from_wei(my_credit,'ether')} ETH")

if my_credit > 0:
    try:
        send_tx(orch.functions.withdraw(REQ_ID), min_gas=120_000)
        print("   ✓ withdrawn")
    except Exception as e:
        msg = explain_web3_error(e)
        if "no credit" in msg: print("   (race) no credit left at withdraw time.")
        else: print("   withdraw failed:", msg)

# right after a successful withdraw (or after settlement block, regardless of withdraw outcome)
bal1 = int(w3.eth.get_balance(addr))
delta = bal1 - bal0
print(f"   balance after:  {w3.from_wei(bal1,'ether')} ETH (Δ {w3.from_wei(delta,'ether')} ETH)")
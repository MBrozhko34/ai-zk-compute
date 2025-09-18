#!/usr/bin/env python3
"""
Worker for XOR PLONK circuit:
 - Trains deterministically (mirrors circuit),
 - Proves with snarkjs,
 - Submits (uint256[24], uint256[3]) to AiOrchestrator.submitProof.
"""

import json, os, subprocess, tempfile, pathlib, time
from typing import Optional, List, Tuple

from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount

# ───────── env
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0) or 0)

ZKEY       = pathlib.Path("../circuits/xor_final.zkey")
VK_JSON    = pathlib.Path("../circuits/verification_key.json")
CIRC_DIR   = pathlib.Path("../circuits/XorCircuit_js")
ABI_PATH   = pathlib.Path("../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
SCRIPTS_JS = pathlib.Path("../scripts/plonk_calldata.js")  # normalize calldata

assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

# ───────── robust parsing helpers
def _first_json_array_slice(s: str) -> str:
    s = s.strip()
    b = s.find('[')
    if b == -1:
        raise RuntimeError("soliditycalldata: '[' not found")
    depth = 0
    e = None
    for i, c in enumerate(s[b:], start=b):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                e = i
                break
    if e is None:
        raise RuntimeError("soliditycalldata: unmatched brackets")
    return s[b:e+1]

def _to_int(v):
    if isinstance(v, int): return v
    if isinstance(v, str):
        v = v.strip()
        return int(v, 16) if v.lower().startswith("0x") else int(v)
    raise ValueError(f"bad numeric type: {type(v)}")

def _read_public_signals(public_json_path: pathlib.Path) -> List[int]:
    # snarkjs plonk prove writes {"publicSignals":[...], "proof":{...}} OR just array
    data = json.loads(public_json_path.read_text())
    if isinstance(data, dict) and "publicSignals" in data:
        pubs = data["publicSignals"]
    elif isinstance(data, list):
        pubs = data  # some very old outputs
    else:
        raise RuntimeError("public.json: unrecognized shape")
    pubs = [_to_int(x) for x in pubs]
    if len(pubs) != 3:
        raise RuntimeError(f"expected 3 public signals, got {len(pubs)}: {pubs}")
    return pubs

def parse_plonk_calldata_any(txt: str, public_json_path: pathlib.Path) -> Tuple[List[int], List[int]]:
    """
    Accepts:
      A) [[24 words],[pubs]]
      B) [24 words]  => pubs loaded from public.json
      C) ["0x<proof-hex>", [pubs]] => reject for this verifier (needs 24 words)
    Returns (proof_words[24] ints, pubs[3] ints)
    """
    arr = json.loads(_first_json_array_slice(txt))

    # Case A: [[24], [3]]
    if (isinstance(arr, list) and len(arr) == 2
            and isinstance(arr[0], list) and isinstance(arr[1], list)):
        proof_raw, pubs_raw = arr[0], arr[1]
        if len(proof_raw) != 24:
            raise RuntimeError(f"expected 24-word proof, got {len(proof_raw)}")
        proof_words = [_to_int(x) for x in proof_raw]
        pubs = [_to_int(x) for x in pubs_raw]
        if len(pubs) != 3:
            raise RuntimeError(f"expected 3 public signals, got {len(pubs)}")
        return proof_words, pubs

    # Case C: ["0x...", [pubs]] (hex proof) — not compatible with uint256[24] verifier
    if (isinstance(arr, list) and len(arr) == 2
            and isinstance(arr[0], str) and arr[0].lower().startswith("0x")):
        # We cannot convert hex → 24 field elements reliably for the verifier.
        # Hint to use JS script path that returns arrays or fall back to zkey with array mode.
        raise RuntimeError("Got hex proof; need 24-word array. Ensure plonk_calldata.js returns arrays.")

    # Case B: [24] only (proof words), pubs in public.json
    if isinstance(arr, list) and len(arr) == 24 and all(isinstance(x, (str, int)) for x in arr):
        proof_words = [_to_int(x) for x in arr]
        pubs = _read_public_signals(public_json_path)
        return proof_words, pubs

    # Some snarkjs versions print a flat array: [24 proof words, then 3 pubs]
    if isinstance(arr, list) and len(arr) >= 27:
        proof_words = [_to_int(x) for x in arr[:24]]
        pubs       = [_to_int(x) for x in arr[24:24+3]]
        return proof_words, pubs

    raise RuntimeError(f"Unexpected calldata JSON shape: {str(arr)[:120]}...")

# ───────── circuit-mirroring helpers
def step_gate(v: int) -> int:
    return 1 if v >= 1 else 0

def lr_bucket(lr_ppm: int) -> int:
    return 1 + (1 if lr_ppm >= 50_000 else 0) + (1 if lr_ppm >= 100_000 else 0)

def sat_add3_floor(x: int, inc: int, dec: int, delta: int, floor: int = 0) -> int:
    x_cap = x + inc * delta
    if x_cap > 3: x_cap = 3
    sub = dec * delta
    if x_cap - sub < floor: return floor
    return x_cap - sub

def train_and_eval(lr_ppm: int, steps: int, max_epochs: int = 300) -> int:
    w0 = w1 = w2 = w3 = 1
    b0 = b1 = 0
    bO = 1
    X = [(0,0),(1,0),(0,1),(1,1)]
    Y = [0,1,1,0]
    L = lr_bucket(lr_ppm)
    T = min(int(steps), max_epochs)
    for _e in range(T):
        for (x0,x1), y in zip(X, Y):
            s0 = step_gate(w0*x0 + w1*x1 + b0)
            s1 = step_gate(w2*x0 + w3*x1 + b1)
            o  = step_gate(s0 - s1 + bO)
            pos = (1 - o) * y
            neg = (1 - y) * o
            d0 = L * x0
            d1 = L * x1
            w0 = sat_add3_floor(w0, pos, neg, d0, 0)
            w1 = sat_add3_floor(w1, pos, neg, d1, 0)
            b0 = sat_add3_floor(b0, pos, neg, L,  0)
            w2 = sat_add3_floor(w2, neg, pos, d0, 0)
            w3 = sat_add3_floor(w3, neg, pos, d1, 0)
            b1 = sat_add3_floor(b1, neg, pos, L,  0)
            bO = sat_add3_floor(bO, pos, neg, L,  1)
    correct = 0
    for (x0,x1), y in zip(X, Y):
        s0 = step_gate(w0*x0 + w1*x1 + b0)
        s1 = step_gate(w2*x0 + w3*x1 + b1)
        o  = step_gate(s0 - s1 + bO)
        correct += int(o == y)
    return correct * 2500

# ───────── chain handles
w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)
orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR),
                       abi=json.load(open(ABI_PATH))["abi"])

addr = acct.address
bal0 = int(w3.eth.get_balance(addr))
print(f"[worker:{addr}] starting on request {REQ_ID}…")
print(f"   balance before: {w3.from_wei(bal0,'ether')} ETH")

# ───────── error decode helpers
def _decode_error_string(data_hex: str):
    try:
        if not data_hex or not data_hex.startswith("0x"): return None
        b = bytes.fromhex(data_hex[2:])
        if len(b) < 4: return None
        sel = b[:4]
        if sel == bytes.fromhex("08c379a0"):
            if len(b) >= 4 + 32 + 32:
                strlen = int.from_bytes(b[4+32:4+32+32], "big")
                s = b[4+32+32:4+32+32+strlen]
                return s.decode("utf-8", errors="replace")
        elif sel == bytes.fromhex("4e487b71"): return "panic()"
        elif sel == bytes.fromhex("3ee5aeb5"): return "ReentrancyGuardReentrantCall()"
        return None
    except Exception:
        return None

def explain_web3_error(e: Exception) -> str:
    if isinstance(e, ValueError) and e.args and isinstance(e.args[0], dict):
        err = e.args[0]; data = err.get("data")
        if isinstance(data, dict):
            s = _decode_error_string(data.get("data") or "")
            if s: return f"revert '{s}'"
        elif isinstance(data, str):
            s = _decode_error_string(data)
            if s: return f"revert '{s}'"
        return err.get("message","") or str(e)
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
                    time.sleep(0.5); continue
            if "ran out of gas" in msg.lower() or "intrinsic gas too low" in msg.lower():
                raise RuntimeError(f"OOG — try higher gas. Details: {msg}") from e
            raise RuntimeError(msg) from e

# ───────── small helpers
def chain_now() -> int:
    return int(w3.eth.get_block("latest")["timestamp"])

def space_len() -> int:
    return len(orch.functions.getSpace(REQ_ID).call())

def count_proven(n: int) -> int:
    return sum(1 for i in range(n) if orch.functions.taskAcc(REQ_ID, i).call() != 0)

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

# Train/prove/submit helper
def do_task(idx: int, lr_ppm: int, steps: int):
    acc_bps = train_and_eval(lr_ppm, steps)
    print(f"   [task {idx}] trained {min(steps,300)} epochs (bucket={lr_bucket(lr_ppm)}) → acc {acc_bps/100}%")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        (tmp/"input.json").write_text(json.dumps({
            "lr": lr_ppm, "steps": steps, "acc_bps": acc_bps
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

        # Prefer the JS normalizer so we get arrays (not hex)
        if SCRIPTS_JS.exists():
            raw = subprocess.check_output([
                "node", str(SCRIPTS_JS),
                str(tmp/"proof.json"), str(tmp/"public.json")
            ]).decode()
        else:
            # Fallback to snarkjs CLI (may return a string); parser handles flat/array forms
            raw = subprocess.check_output([
                "snarkjs", "zkey", "export", "soliditycalldata",
                str(tmp/"public.json"), str(tmp/"proof.json")
            ]).decode()

    proof_words, public_signals = parse_plonk_calldata_any(raw, tmp / "public.json")

    if len(proof_words) != 24 or len(public_signals) != 3:
        raise RuntimeError("bad calldata shapes after parse()")

    # Call on-chain with (uint256[24], uint256[3])
    send_tx(orch.functions.submitProof(REQ_ID, proof_words, public_signals),
            min_gas=1_200_000)
    print(f"   ✓ submitted proof for task {idx}")

# Claim one task (preflight & parse event)
def try_claim_one() -> Optional[int]:
    try:
        orch.functions.claimTask(REQ_ID).call({"from": addr, "value": bond_wei})
    except Exception as e:
        msg = explain_web3_error(e)
        if "no tasks left" in msg or "closed" in msg or "not started" in msg or "bond" in msg:
            return None
        print("   claimTask preflight failed:", msg)
        return None

    rcpt = send_tx(orch.functions.claimTask(REQ_ID), value=bond_wei, min_gas=230_000)
    evs = orch.events.TaskClaimed().process_receipt(rcpt)
    if evs:
        return int(evs[-1]["args"]["idx"])

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

        if proven == n - 1:
            for i in range(n):
                acc = orch.functions.taskAcc(REQ_ID, i).call()
                if acc != 0: continue
                owner = orch.functions.taskOwner(REQ_ID, i).call()
                if owner == ZERO: continue
                try:
                    orch.functions.reassignTimedOut(REQ_ID, i).call({"from": addr})
                except Exception:
                    pass
                else:
                    try:
                        send_tx(orch.functions.reassignTimedOut(REQ_ID, i), min_gas=180_000)
                        print(f"   reassignTimedOut(last, {i})")
                    except Exception:
                        pass
            return

        for i in range(n):
            acc = orch.functions.taskAcc(REQ_ID, i).call()
            if acc != 0: continue
            owner = orch.functions.taskOwner(REQ_ID, i).call()
            if owner == ZERO: continue
            t = orch.functions.claimedAt(REQ_ID, i).call()
            if t == 0: continue
            remain = (t + ttl_sec) - now
            if remain > 0:
                print(f"   waiting TTL for idx {i}… {remain}s")
                continue
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
        except subprocess.CalledProcessError:
            print(f"   snark/proof failed for task {idx}; it will time out and be reassignable.")
        except Exception as e:
            print(f"   task {idx} failed: {e}")

    closed, bestAcc, winTaskCount, perWinTaskWei, perLoseTaskWei = orch.functions.getResult(REQ_ID).call()
    if closed:
        break

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

my_credit = int(orch.functions.credit(REQ_ID, addr).call())
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

bal1 = int(w3.eth.get_balance(addr))
delta = bal1 - bal0
print(f"   balance after:  {w3.from_wei(bal1,'ether')} ETH (Δ {w3.from_wei(delta,'ether')} ETH)")

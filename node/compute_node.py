#!/usr/bin/env python3
import json, os, subprocess, tempfile, pathlib, time, math
from typing import Optional, List, Tuple

from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount

# ───────── env
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0) or 0)

TRAIN_CSV = os.getenv("TRAIN_CSV",   "../client/train.csv")
HOLD_CSV  = os.getenv("HOLDOUT_CSV", "../client/dataset.csv")

ABI_PATH  = pathlib.Path("../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

# ───────── chain handles
w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)
orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR),
                       abi=json.load(open(ABI_PATH))["abi"])
addr = acct.address

# ───────── integer model constants (must match contract)
CAP = 15
TH  = 8

def quant01_to_q3(v) -> int:
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"bad number {v}")
    q = int(round(max(0.0, min(1.0, f)) * 3.0))
    return max(0, min(3, q))

def load_csv2(path: str):
    xs0: List[int] = []
    xs1: List[int] = []
    ys:  List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f.read().splitlines()):
            if not raw: continue
            line = raw.strip()
            if not line or line.startswith("#"): continue
            cells = [c for c in re_split(line) if c]
            if len(cells) < 3: continue
            if i == 0 and not cells[0].replace(".","",1).isdigit():  # header heuristic
                continue
            x0 = quant01_to_q3(cells[0])
            x1 = quant01_to_q3(cells[1])
            y  = int(cells[2])
            if y not in (0,1): raise ValueError(f"y must be 0/1 at row {i+1}")
            xs0.append(x0); xs1.append(x1); ys.append(y)
    if not xs0: raise RuntimeError(f"empty/invalid csv {path}")
    return xs0, xs1, ys

def re_split(s: str):
    # split on comma / semicolon / whitespace
    import re
    return re.split(r"[,;\s]+", s.strip())

# ───────── robust Merkle (sorted-pair)
def h256(b: bytes) -> bytes:
    return Web3.keccak(b)

def bytes32_of_int(x: int) -> bytes:
    return x.to_bytes(32, byteorder="big", signed=False)

def hash_pair_sorted(a: bytes, b: bytes) -> bytes:
    return h256(a + b) if a < b else h256(b + a)

def build_merkle_root_sorted(leaves: List[bytes]) -> bytes:
    if not leaves:
        return b"\x00"*32
    level = leaves[:]
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            if i+1 < len(level):
                nxt.append(hash_pair_sorted(level[i], level[i+1]))
            else:
                nxt.append(hash_pair_sorted(level[i], level[i]))
        level = nxt
    return level[0]

def build_merkle_layers_sorted(leaves: List[bytes]) -> List[List[bytes]]:
    layers = [leaves[:]]
    while len(layers[-1]) > 1:
        cur = layers[-1]
        nxt = []
        for i in range(0, len(cur), 2):
            if i+1 < len(cur): nxt.append(hash_pair_sorted(cur[i], cur[i+1]))
            else:              nxt.append(hash_pair_sorted(cur[i], cur[i]))
        layers.append(nxt)
    return layers

def merkle_proof_sorted(layers: List[List[bytes]], index: int) -> List[bytes]:
    path: List[bytes] = []
    idx = index
    for h in range(len(layers) - 1):
        cur = layers[h]
        sib = idx ^ 1
        if sib < len(cur): path.append(cur[sib])
        else:              path.append(cur[idx])
        idx //= 2
    return path

# transcript leaf: keccak(stepIndex, H(Ws), H(We))
def hashW7(W: Tuple[int,...]) -> bytes:
    assert len(W) == 7
    return h256(b"".join(bytes32_of_int(v) for v in W))

def leaf_for_step(step_idx: int, ws: Tuple[int,...], we: Tuple[int,...]) -> bytes:
    return h256(bytes32_of_int(step_idx) + hashW7(ws) + hashW7(we))

# training-sample leaf: keccak(index, x0, x1, y)
def leaf_for_sample(idx: int, x0: int, x1: int, y: int) -> bytes:
    return h256(b"".join([
        bytes32_of_int(idx), bytes32_of_int(x0), bytes32_of_int(x1), bytes32_of_int(y)
    ]))

# ───────── integer model: bucket, step, sat update
def lr_bucket(lr_ppm: int) -> int:
    return 1 + (1 if lr_ppm >= 50_000 else 0) + (1 if lr_ppm >= 100_000 else 0)

def ge(v: int, thr: int) -> int:
    return 1 if v >= thr else 0

def sat_update(x: int, inc: int, dec: int, delta: int, floor: int = 0) -> int:
    xcap = x + inc * delta
    if xcap > CAP: xcap = CAP
    sub = dec * delta
    if xcap - sub < floor: return floor
    return xcap - sub

def apply_one_step(W: Tuple[int,...], x0: int, x1: int, y: int, L: int) -> Tuple[int,...]:
    w0,w1,w2,w3,b0,b1,bO = W
    s0 = ge(w0 * x0 + w1 * x1 + b0, TH)
    s1 = ge(w2 * x0 + w3 * x1 + b1, TH)
    o  = 1 if (int(s0) - int(s1) + int(bO)) >= 1 else 0

    pos = (1 - o) * y
    neg = (1 - y) * o

    d0 = L * x0
    d1 = L * x1

    w0 = sat_update(w0, pos, neg, d0, 0)
    w1 = sat_update(w1, pos, neg, d1, 0)
    b0 = sat_update(b0, pos, neg, L,  0)

    w2 = sat_update(w2, neg, pos, d0, 0)
    w3 = sat_update(w3, neg, pos, d1, 0)
    b1 = sat_update(b1, neg, pos, L,  0)

    bO = sat_update(bO, pos, neg, L,  0)
    return (w0,w1,w2,w3,b0,b1,bO)

def train_collect(xs0: List[int], xs1: List[int], ys: List[int], lr_ppm: int, steps: int):
    assert len(xs0) == len(xs1) == len(ys)
    N = len(xs0)
    L = lr_bucket(lr_ppm)
    T = int(steps)

    # init weights
    W = (1,1,1,1, 0,0, 0)  # bO starts at 0 now

    pairs: List[Tuple[Tuple[int,...], Tuple[int,...]]] = []
    for e in range(T):
        for j in range(N):
            ws = W
            W  = apply_one_step(W, xs0[j], xs1[j], ys[j], L)
            pairs.append((ws, W))
    return W, pairs, N, T

def accuracy_on(W: Tuple[int,...], xs0: List[int], xs1: List[int], ys: List[int]) -> float:
    w0,w1,w2,w3,b0,b1,bO = W
    correct = 0
    for x0,x1,y in zip(xs0, xs1, ys):
        s0 = ge(w0 * x0 + w1 * x1 + b0, TH)
        s1 = ge(w2 * x0 + w3 * x1 + b1, TH)
        o  = 1 if (int(s0) - int(s1) + int(bO)) >= 1 else 0
        correct += int(o == y)
    return correct / len(xs0)

# ───────── tx helpers
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

def next_nonce() -> int:
    return w3.eth.get_transaction_count(acct.address, "pending")

def send_tx(fn_call, *, value: int = 0, min_gas: int = 220_000, headroom_num: int = 15, headroom_den: int = 10):
    try:
        est = int(fn_call.estimate_gas({"from": acct.address, "value": value}))
        gas_limit = max(min_gas, est * headroom_num // headroom_den)
    except Exception as e:
        print("   gas estimation failed, defaulting:", explain_web3_error(e))
        gas_limit = max(min_gas, 3_000_000)

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

# ───────── event topics (optional utils)
TOPIC_ONCHAIN_ACC = w3.keccak(text="OnChainAccChecked(uint256,uint256,uint256,uint256,uint256)").hex()
TOPIC_FINALIZED   = w3.keccak(text="ChallengesFinalized(uint256,uint256,bytes32,uint256)").hex()
TOPIC_ACCEPTED    = w3.keccak(text="ProofAccepted(uint256,uint256,address,uint256)").hex()

def chain_now() -> int:
    return int(w3.eth.get_block("latest")["timestamp"])

def space_len() -> int:
    return len(orch.functions.getSpace(REQ_ID).call())

def count_proven(n: int) -> int:
    return sum(1 for i in range(n) if orch.functions.taskAcc(REQ_ID, i).call() != 0)

def ensure_owner(idx: int) -> bool:
    return orch.functions.taskOwner(REQ_ID, idx).call().lower() == addr.lower()

# ───────── main
print(f"[worker:{addr}] starting on request {REQ_ID}…")
bal0 = int(w3.eth.get_balance(addr))
print(f"   balance before: {w3.from_wei(bal0,'ether')} ETH")

import re as _re
def _is_header_line(line: str) -> bool:
    return bool(_re.search(r"[A-Za-z]", line))

# Load data
tr_x0, tr_x1, tr_y = load_csv2(TRAIN_CSV)
te_x0, te_x1, te_y = load_csv2(HOLD_CSV)

# join / wait
try:
    closed, *_ = orch.functions.getResult(REQ_ID).call()
    if closed:
        print("   request is already closed; exiting.")
        raise SystemExit(0)
except Exception as e:
    print("   warning: getResult() failed (continuing):", explain_web3_error(e))

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

while True:
    try:
        needed, joined_, ready = orch.functions.lobbyCounts(REQ_ID).call()
        print(f"   lobby: {joined_}/{needed} ready={ready}")
        if ready: break
        time.sleep(1.0)
    except Exception as e:
        print("   lobbyCounts() failed:", explain_web3_error(e)); time.sleep(1.0)

# bond / ttl
try:
    bond_wei   = int(orch.functions.CLAIM_BOND_WEI().call())
    claim_ttl  = int(orch.functions.CLAIM_TTL().call())
    stall_ttl  = int(orch.functions.STALL_TTL().call())
except Exception:
    bond_wei  = int(Web3.to_wei(0.005, "ether"))
    claim_ttl = 10
    stall_ttl = 60

def try_claim_one() -> Optional[int]:
    try:
        orch.functions.claimTask(REQ_ID).call({"from": addr, "value": bond_wei})
    except Exception as e:
        msg = explain_web3_error(e)
        if any(x in msg for x in ["no tasks left", "closed", "not started", "bond", "holdout not set", "train root not set"]):
            return None
        print("   claimTask preflight failed:", msg); return None

    rcpt = send_tx(orch.functions.claimTask(REQ_ID), value=bond_wei, min_gas=230_000)
    evs = orch.events.TaskClaimed().process_receipt(rcpt)
    if evs: return int(evs[-1]["args"]["idx"])
    # fallback scan
    n = space_len()
    for i in range(n):
        owner = orch.functions.taskOwner(REQ_ID, i).call()
        acc   = orch.functions.taskAcc(REQ_ID, i).call()
        if owner.lower() == addr.lower() and acc == 0:
            return i
    return None

ZERO = "0x0000000000000000000000000000000000000000"

def maybe_reassign_timeouts():
    # Do NOT reassign tasks that we own (prevents self-sabotage / "task not yours")
    try:
        n = space_len()
        proven = count_proven(n)
        now = chain_now()
        majority = (proven * 2) > n
        ttl = claim_ttl if majority else stall_ttl

        if proven == n - 1:
            for i in range(n):
                if orch.functions.taskOwner(REQ_ID, i).call().lower() == addr.lower():  # skip mine
                    continue
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
            if orch.functions.taskOwner(REQ_ID, i).call().lower() == addr.lower():  # skip mine
                continue
            acc = orch.functions.taskAcc(REQ_ID, i).call()
            if acc != 0: continue
            owner = orch.functions.taskOwner(REQ_ID, i).call()
            if owner == ZERO: continue
            t = orch.functions.claimedAt(REQ_ID, i).call()
            if t == 0: continue
            remain = (t + ttl) - now
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

# Build training-sample Merkle (once)
samp_leaves = [leaf_for_sample(i, tr_x0[i], tr_x1[i], tr_y[i]) for i in range(len(tr_x0))]
samp_layers = build_merkle_layers_sorted(samp_leaves)
samp_root   = build_merkle_root_sorted(samp_leaves)

# ───────── do one task end-to-end
def do_task(idx: int, lr_ppm: int, steps: int):
    # TRAIN
    finalW, pairs, N, T = train_collect(tr_x0, tr_x1, tr_y, lr_ppm, steps)
    train_acc = accuracy_on(finalW, tr_x0, tr_x1, tr_y)
    test_acc  = accuracy_on(finalW, te_x0, te_x1, te_y)
    print(f"   [task {idx}] trained {T} epochs (bucket={lr_bucket(lr_ppm)}) → "
          f"train acc {100*train_acc:.2f}% | test acc {100*test_acc:.2f}%")

    # TRANSCRIPT
    leaves = [leaf_for_step(i, ws, we) for i, (ws,we) in enumerate(pairs)]
    tr_layers = build_merkle_layers_sorted(leaves)
    tr_root   = build_merkle_root_sorted(leaves)
    total     = len(leaves)
    print(f"   committed transcript root={Web3.to_hex(tr_root)} totalSteps={total}")

    if not ensure_owner(idx): 
        print("   lost ownership before commit; aborting task.")
        return

    # 2a) commit transcript
    try:
        # preflight
        current = orch.functions.trRoot(REQ_ID, idx).call()
        if int(current,16) == 0:
            send_tx(orch.functions.commitTranscript(REQ_ID, idx, Web3.to_hex(tr_root), total), min_gas=200_000)
        else:
            print("   transcript already committed on-chain.")
    except Exception as e:
        msg = explain_web3_error(e)
        if "already committed" in msg:
            print("   transcript already committed on-chain.")
        else:
            raise

    if not ensure_owner(idx): 
        print("   lost ownership before finalize; aborting task.")
        return

    # 2b) finalize K
    K = 3
    try:
        k_cur = orch.functions.challengeK(REQ_ID, idx).call()
        if int(k_cur) == 0:
            rcpt = send_tx(orch.functions.finalizeChallenges(REQ_ID, idx, K), min_gas=200_000)
            for log in rcpt.logs:
                if log["address"].lower() != ORCH_ADDR.lower(): continue
                if log["topics"][0].hex().lower() == TOPIC_FINALIZED.lower():
                    ev = orch.events.ChallengesFinalized().process_log(log)
                    sd = ev["args"]["seed"]
                    print(f"   finalized seed={Web3.to_hex(sd)}  challengeK={K}")
        else:
            print(f"   challenges already finalized, K={int(k_cur)}")
            K = int(k_cur)
    except Exception as e:
        print("   finalizeChallenges failed:", explain_web3_error(e)); return

    # 2c) answer each challenge
    for i in range(K):
        if not ensure_owner(idx): 
            print("   lost ownership mid-challenges; aborting task.")
            return
        step_idx = int(orch.functions.getChallenge(REQ_ID, idx, i).call())
        ws, we = pairs[step_idx]
        # training sample index & proof for this step
        samp_idx = step_idx % N
        proof_samp_nodes = merkle_proof_sorted(samp_layers, samp_idx)
        proof_samp_hexes = [Web3.to_hex(p) for p in proof_samp_nodes]
        # transcript proof
        proof_tr_nodes = merkle_proof_sorted(tr_layers, step_idx)
        proof_tr_hexes = [Web3.to_hex(p) for p in proof_tr_nodes]

        try:
            send_tx(
                orch.functions.bindFinalWeights(
                    REQ_ID, idx,
                    i,
                    list(ws), list(we),
                    samp_idx, tr_x0[samp_idx], tr_x1[samp_idx], tr_y[samp_idx],
                    proof_tr_hexes, proof_samp_hexes
                ),
                min_gas=400_000
            )
            print(f"   ✓ answered challenge i={i} (step={step_idx} using sample {samp_idx})")
        except Exception as e:
            print(f"   bindFinalWeights failed at i={i}:", explain_web3_error(e))
            return

    if not ensure_owner(idx):
        print("   lost ownership before submit; aborting task.")
        return

    # 3) submit result (hold-out accuracy rechecked on-chain)
    w0,w1,w2,w3,b0,b1,bO = finalW
    acc_bps = int(round(10000.0 * test_acc))
    rcpt = send_tx(orch.functions.submitResult(REQ_ID, idx, [w0,w1,w2,w3,b0,b1,bO], acc_bps), min_gas=250_000)
    # parse OnChainAccChecked
    for log in rcpt.logs:
        if log["address"].lower() != ORCH_ADDR.lower(): continue
        if log["topics"][0].hex().lower() == TOPIC_ONCHAIN_ACC.lower():
            ev = orch.events.OnChainAccChecked().process_log(log)
            args = ev["args"]
            print(f"   on-chain test: claimed {int(args['claimedAccBps'])/100:.2f}% "
                  f"vs recomputed {int(args['accOnChain'])/100:.2f}% on {int(args['rows'])} rows")
    # accepted?
    accepted = any(log["topics"][0].hex().lower() == TOPIC_ACCEPTED.lower()
                   for log in rcpt.logs if log["address"].lower() == ORCH_ADDR.lower())
    if accepted:
        print(f"   ✓ accepted on-chain (task {idx})")
    else:
        print(f"   ✗ not accepted (task {idx})")

# ───────── main loop
while True:
    claimed_any = False
    while True:
        idx = try_claim_one()
        if idx is None: break
        claimed_any = True
        lr_ppm, steps = orch.functions.getSpace(REQ_ID).call()[idx]
        print(f"   claimed task {idx} (lr={lr_ppm}, steps={steps}) with bond {w3.from_wei(bond_wei,'ether')} ETH")
        try:
            do_task(int(idx), int(lr_ppm), int(steps))
        except Exception as e:
            print(f"   task {idx} failed:", e)

    closed, bestAcc, winTaskCount, perWinTaskWei, perLoseTaskWei = orch.functions.getResult(REQ_ID).call()
    if closed: break
    if not claimed_any:
        maybe_reassign_timeouts()
    time.sleep(1.0)

# ───────── settlement summary
closed, bestAcc, winTaskCount, perWinTaskWei, perLoseTaskWei = orch.functions.getResult(REQ_ID).call()
space = orch.functions.getSpace(REQ_ID).call()

my_total = my_proven = my_win = my_lose = 0
for i in range(len(space)):
    if orch.functions.taskOwner(REQ_ID, i).call().lower() == addr.lower():
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

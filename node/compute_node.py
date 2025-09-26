#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, pathlib, time, math, re
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List, Tuple
from hexbytes import HexBytes

from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount

# ──────────────────────────────────────────────────────────────────────────────
# ENV / CHAIN
# ──────────────────────────────────────────────────────────────────────────────
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0) or 0)

# CSV paths (same defaults as the TS client)
TRAIN_CSV = os.getenv("TRAIN_CSV",   "client/train.csv")
HOLD_CSV  = os.getenv("HOLDOUT_CSV", os.getenv("DATASET_CSV", "client/dataset.csv"))

assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

# ABI path: allow override; otherwise search common locations
def _find_abi_path() -> pathlib.Path:
    env_path = os.getenv("ORCH_ABI_PATH")
    if env_path:
        p = pathlib.Path(env_path)
        if p.is_file():
            return p
    here = pathlib.Path.cwd()
    for p in [
        here / "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json",
        here.parent / "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json",
        here.parent.parent / "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json",
        here / "out/AiOrchestrator.sol/AiOrchestrator.json",
        here.parent / "out/AiOrchestrator.sol/AiOrchestrator.json",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError("Could not locate AiOrchestrator ABI. "
                            "Set ORCH_ABI_PATH or run `npx hardhat compile` in repo root.")

ABI_PATH = _find_abi_path()

def _resolve_csv(path: str) -> str:
    """Resolve CSV relative to repo root or node/ so TS and Python see the same file."""
    p = pathlib.Path(path)
    if p.is_file():
        return str(p)
    p2 = pathlib.Path("..") / path  # when launched from node/
    if p2.is_file():
        return str(p2)
    raise FileNotFoundError(f"CSV not found at '{path}' or '{p2}'")

TRAIN_CSV = _resolve_csv(TRAIN_CSV)
HOLD_CSV  = _resolve_csv(HOLD_CSV)

w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)
orch = w3.eth.contract(
    address=w3.to_checksum_address(ORCH_ADDR),
    abi=json.load(open(ABI_PATH, "r", encoding="utf-8"))["abi"]
)
addr = acct.address

# ──────────────────────────────────────────────────────────────────────────────
# MODEL (Integer MLP 2→4→1, 17 params) — EXACTLY mirrors AiOrchestrator.sol
# ──────────────────────────────────────────────────────────────────────────────
CAP_I = 127

def lr_bucket(lr_ppm: int) -> int:
    return 1 + (1 if lr_ppm >= 50_000 else 0) + (1 if lr_ppm >= 100_000 else 0)

def sgn(v: int) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)

def sat_i(x: int) -> int:
    if x >  CAP_I: return CAP_I
    if x < -CAP_I: return -CAP_I
    return x

def apply_one_step_mlp(W: Tuple[int, ...], x0: int, x1: int, y: int, L: int) -> Tuple[int, ...]:
    a = list(W)
    pre = [0,0,0,0]
    s   = [0,0,0,0]
    for j in range(4):
        base = j*2
        pre[j] = a[base+0]*x0 + a[base+1]*x1 + a[8 + j]
        s[j]   = 1 if pre[j] >= 0 else 0
    z = a[16]
    for j in range(4):
        z += a[12 + j] * s[j]
    p = 1 if z >= 0 else 0
    e = y - p
    if e != 0:
        Vold = [a[12 + j] for j in range(4)]
        for j in range(4):
            if s[j] != 0:
                a[12 + j] = sat_i(a[12 + j] + e * L * s[j])
        a[16] = sat_i(a[16] + e * L)
        for j in range(4):
            if s[j] != 0:
                sign_v = sgn(Vold[j])
                d = e * L * sign_v
                base = j*2
                a[base+0] = sat_i(a[base+0] + d * x0)
                a[base+1] = sat_i(a[base+1] + d * x1)
                a[8 + j]  = sat_i(a[8 + j]  + d)
    return tuple(a)

def init_weights() -> Tuple[int, ...]:
    return (1,1,  1,1,  1,1,  1,1,  0,0,0,0,  1,1,1,1,  0)

def forward(W: Tuple[int,...], x0: int, x1: int) -> int:
    a = W
    s = []
    for j in range(4):
        base = j*2
        pre = a[base+0]*x0 + a[base+1]*x1 + a[8+j]
        s.append(1 if pre >= 0 else 0)
    z = a[16]
    for j in range(4):
        z += a[12+j] * s[j]
    return 1 if z >= 0 else 0

def accuracy_on(W: Tuple[int,...], xs0: List[int], xs1: List[int], ys: List[int]) -> float:
    correct = 0
    for x0, x1, y in zip(xs0, xs1, ys):
        correct += int(forward(W, x0, x1) == y)
    return correct / len(xs0)

def train_collect(xs0: List[int], xs1: List[int], ys: List[int], lr_ppm: int, steps: int):
    N = len(xs0)
    L = lr_bucket(lr_ppm)
    T = int(steps)
    W = init_weights()
    pairs: List[Tuple[Tuple[int,...], Tuple[int,...]]] = []
    for _ in range(T):
        for j in range(N):
            ws = W
            W  = apply_one_step_mlp(W, xs0[j], xs1[j], ys[j], L)
            pairs.append((ws, W))
    return W, pairs, N, T

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MERKLE (sorted-pair)
# ──────────────────────────────────────────────────────────────────────────────
def quant01_to_q15_js(v) -> int:
    """
    Exact JS Math.round emulation into 0..15 after clamping to [0,1].
    """
    f = Decimal(str(v).strip())
    if f < 0: f = Decimal(0)
    if f > 1: f = Decimal(1)
    q = int((f * Decimal(15)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    if q < 0: q = 0
    if q > 15: q = 15
    return q

def _cells(line: str) -> List[str]:
    return re.split(r"[,;\s]+", line.strip())

def load_csv2(path: str):
    xs0: List[int] = []
    xs1: List[int] = []
    ys:  List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f.read().splitlines()):
            if not raw: continue
            line = raw.strip()
            if not line or line.startswith("#"): continue
            cells = [c for c in _cells(line) if c]
            if len(cells) < 3: continue
            # skip header if first cell looks non-numeric
            try: _ = float(cells[0])
            except: 
                if i == 0: continue
                else: raise
            x0 = quant01_to_q15_js(cells[0])
            x1 = quant01_to_q15_js(cells[1])
            y  = int(cells[2])
            if y not in (0,1): raise ValueError(f"y must be 0/1 at row {i+1}")
            xs0.append(x0); xs1.append(x1); ys.append(y)
    if not xs0: raise RuntimeError(f"empty/invalid csv {path}")
    return xs0, xs1, ys

def h256(b: bytes) -> bytes:
    return Web3.keccak(b)

def b32u(x: int) -> bytes:
    return int(x).to_bytes(32, byteorder="big", signed=False)

def hash_pair_sorted(a: bytes, b: bytes) -> bytes:
    return h256(a + b) if a < b else h256(b + a)

def build_merkle_layers_sorted(leaves: List[bytes]) -> List[List[bytes]]:
    if not leaves: return [[]]
    layers = [leaves[:]]
    while len(layers[-1]) > 1:
        cur = layers[-1]
        nxt: List[bytes] = []
        for i in range(0, len(cur), 2):
            if i+1 < len(cur): nxt.append(hash_pair_sorted(cur[i], cur[i+1]))
            else:              nxt.append(hash_pair_sorted(cur[i], cur[i]))
        layers.append(nxt)
    return layers

def build_merkle_root_sorted(leaves: List[bytes]) -> bytes:
    layers = build_merkle_layers_sorted(leaves)
    if not layers or not layers[-1]: return b"\x00"*32
    return layers[-1][0]

def merkle_proof_sorted(layers: List[List[bytes]], index: int) -> List[bytes]:
    if not layers or not layers[0]: return []
    path: List[bytes] = []
    idx = index
    for level in layers[:-1]:
        sib = idx ^ 1
        if sib < len(level): path.append(level[sib])
        else:                path.append(level[idx])
        idx //= 2
    return path

# Transcript leaf: keccak(stepIndex, H(Ws), H(We))
def hashW17(W: Tuple[int,...]) -> bytes:
    assert len(W) == 17
    # int256 encoding used only inside hashW; sign is intentional per contract
    return h256(b"".join(int(v).to_bytes(32, "big", signed=True) for v in W))

def leaf_for_step(step_idx: int, ws: Tuple[int,...], we: Tuple[int,...]) -> bytes:
    return h256(b32u(step_idx) + hashW17(ws) + hashW17(we))

# Training-sample leaf: keccak(index, x0, x1, y) — all uint256
def leaf_for_sample(idx: int, x0: int, x1: int, y: int) -> bytes:
    return h256(b32u(idx) + b32u(x0) + b32u(x1) + b32u(y))

# ──────────────────────────────────────────────────────────────────────────────
# TX helpers (no fee fields; local nonce; retry on races)
# ──────────────────────────────────────────────────────────────────────────────
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

_last_nonce: Optional[int] = None
def next_nonce() -> int:
    global _last_nonce
    if _last_nonce is None:
        _last_nonce = w3.eth.get_transaction_count(acct.address, "latest")
        return _last_nonce
    _last_nonce += 1
    return _last_nonce

def refresh_nonce_latest():
    global _last_nonce
    _last_nonce = w3.eth.get_transaction_count(acct.address, "latest")
    return _last_nonce

def send_tx(fn_call, *, value: int = 0, min_gas: int = 220_000, headroom_num: int = 15, headroom_den: int = 10):
    try:
        est = int(fn_call.estimate_gas({"from": acct.address, "value": value}))
        gas_limit = max(min_gas, est * headroom_num // headroom_den)
    except Exception as e:
        print("   gas estimation failed, defaulting:", explain_web3_error(e))
        gas_limit = max(min_gas, 3_000_000)
    while True:
        try:
            tx = fn_call.build_transaction({
                "from": acct.address,
                "gas":  gas_limit,
                # no gasPrice, no 1559 fields; node will fill
                "nonce": next_nonce(),
                "value": value,
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.rawTransaction)
            return w3.eth.wait_for_transaction_receipt(h)
        except Exception as e:
            msg = explain_web3_error(e).lower()
            if "nonce too low" in msg or "already known" in msg or "replacement" in msg or "queued" in msg:
                time.sleep(0.2)
                refresh_nonce_latest()
                continue
            raise RuntimeError(msg) from e

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
print(f"[worker:{addr}] starting on request {REQ_ID}…")
print(f"   using TRAIN_CSV={TRAIN_CSV}")
print(f"   using HOLD_CSV ={HOLD_CSV}")
bal0 = int(w3.eth.get_balance(addr))
print(f"   balance before: {w3.from_wei(bal0,'ether')} ETH")

# Load datasets and show our local training root (sanity/debug)
tr_x0, tr_x1, tr_y = load_csv2(TRAIN_CSV)
te_x0, te_x1, te_y = load_csv2(HOLD_CSV)

# print a couple of quantized rows so you can compare with the TS client if needed
for k in range(min(3, len(tr_x0))):
    print(f"   train row[{k}] => x0={tr_x0[k]} x1={tr_x1[k]} y={tr_y[k]}")

samp_leaves = [leaf_for_sample(i, tr_x0[i], tr_x1[i], tr_y[i]) for i in range(len(tr_x0))]
samp_layers = build_merkle_layers_sorted(samp_leaves)
samp_root   = build_merkle_root_sorted(samp_leaves)
print(f"   training root (computed by worker): {Web3.to_hex(samp_root)}")

from hexbytes import HexBytes

def _read_training_root_onchain(req_id: int) -> HexBytes:
    """
    Robustly read R[req_id].trainingRoot without relying on dynamic-field packing.
    Primary: use R(req_id)[5] (Solidity getter returns only static fields).
    Fallback: scan TrainingRootSet events.
    """
    # Primary path: tuple index 5
    try:
        tup = orch.functions.R(req_id).call()
        cand = tup[5]  # (client, bountyWei, minWorkers, started, closed, trainingRoot, ...)
        if isinstance(cand, (bytes, bytearray, HexBytes)):
            return HexBytes(cand)
        if isinstance(cand, str) and cand.startswith("0x") and len(cand) == 66:
            return HexBytes(cand)
    except Exception:
        pass

    # Fallback: event scan
    try:
        evs = orch.events.TrainingRootSet().get_logs(
            fromBlock=0, toBlock="latest",
            argument_filters={"id": req_id}
        )
        if evs:
            return HexBytes(evs[-1]["args"]["root"])
    except Exception:
        pass

    # Default: zero
    return HexBytes("0x" + "00"*32)

# --- use it here ---
onchain_root = _read_training_root_onchain(REQ_ID)
if HexBytes(onchain_root) != HexBytes(samp_root):
    print("   ✗ Training root mismatch:")
    print(f"     on-chain: {onchain_root.hex()}")
    print(f"     worker  : {HexBytes(samp_root).hex()}")
    print("     Fix: re-run `make req` (prints NEW request id) after any CSV/quantization change,")
    print("          then start the worker with that request id. Ensure both sides use 0..15 half-up quantization.")
    raise SystemExit(1)

# Lobby join
try:
    if not orch.functions.joined(REQ_ID, addr).call():
        send_tx(orch.functions.joinLobby(REQ_ID))
        print("   ✓ joined lobby")
    else:
        print("   already joined lobby")
except Exception as e:
    print("   joinLobby failed:", explain_web3_error(e))
    raise SystemExit(1)

# Wait for start
while True:
    needed, joined_, ready = orch.functions.lobbyCounts(REQ_ID).call()
    print(f"   lobby: {joined_}/{needed} ready={ready}")
    if ready: break
    time.sleep(1.0)

# bond parameters
try:
    bond_wei   = int(orch.functions.CLAIM_BOND_WEI().call())
    claim_ttl  = int(orch.functions.CLAIM_TTL().call())
    stall_ttl  = int(orch.functions.STALL_TTL().call())
except Exception:
    bond_wei = int(Web3.to_wei(0.005, "ether")); claim_ttl = 10; stall_ttl = 60

def space_len() -> int:
    return len(orch.functions.getSpace(REQ_ID).call())

def count_proven(n: int) -> int:
    return sum(1 for i in range(n) if orch.functions.taskAcc(REQ_ID, i).call() != 0)

ZERO = "0x0000000000000000000000000000000000000000"

def maybe_reassign_timeouts():
    try:
        n = space_len()
        proven = count_proven(n)
        now = int(w3.eth.get_block("latest")["timestamp"])
        majority = (proven * 2) > n
        ttl = claim_ttl if majority else stall_ttl
        for i in range(n):
            owner = orch.functions.taskOwner(REQ_ID, i).call()
            if owner.lower() == addr.lower():  # never reassign our own task
                continue
            acc = orch.functions.taskAcc(REQ_ID, i).call()
            if acc != 0: continue
            if owner == ZERO:
                continue
            t = orch.functions.claimedAt(REQ_ID, i).call()
            if t == 0 or now < (t + ttl):
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

def try_claim_one() -> Optional[int]:
    try:
        orch.functions.claimTask(REQ_ID).call({"from": addr, "value": bond_wei})
    except Exception as e:
        msg = explain_web3_error(e)
        if any(x in msg for x in ["no tasks left", "closed", "not started", "bond", "holdout not set", "train root not set"]):
            return None
        print("   claimTask preflight failed:", msg)
        return None
    rcpt = send_tx(orch.functions.claimTask(REQ_ID), value=bond_wei, min_gas=230_000)
    evs = orch.events.TaskClaimed().process_receipt(rcpt)
    if evs:
        return int(evs[-1]["args"]["idx"])
    # fallback (scan)
    n = space_len()
    for i in range(n):
        owner = orch.functions.taskOwner(REQ_ID, i).call()
        acc   = orch.functions.taskAcc(REQ_ID, i).call()
        if owner.lower() == addr.lower() and acc == 0:
            return i
    return None

def do_task(idx: int, lr_ppm: int, steps: int):
    finalW, pairs, N, T = train_collect(tr_x0, tr_x1, tr_y, lr_ppm, steps)
    train_acc = accuracy_on(finalW, tr_x0, tr_x1, tr_y)
    test_acc  = accuracy_on(finalW, te_x0, te_x1, te_y)
    print(f"   [task {idx}] trained {T} epochs (bucket={lr_bucket(lr_ppm)}) → "
          f"train acc {100*train_acc:.2f}% | test acc {100*test_acc:.2f}%")

    leaves = [leaf_for_step(i, ws, we) for i, (ws,we) in enumerate(pairs)]
    tr_layers = build_merkle_layers_sorted(leaves)
    tr_root   = build_merkle_root_sorted(leaves)
    total     = len(leaves)
    print(f"   committed transcript root={Web3.to_hex(tr_root)} totalSteps={total}")

    try:
        current = orch.functions.trRoot(REQ_ID, idx).call()
        zero = (isinstance(current, (bytes, bytearray, HexBytes)) and int.from_bytes(current, "big") == 0) \
               or (isinstance(current, str) and current == "0x" + "00"*32)
        if zero:
            send_tx(orch.functions.commitTranscript(REQ_ID, idx, HexBytes(tr_root), total), min_gas=200_000)
        else:
            print("   transcript already committed on-chain.")
    except Exception as e:
        msg = explain_web3_error(e)
        if "already committed" in msg:
            print("   transcript already committed on-chain.")
        else:
            raise

    K = 3
    try:
        k_cur = int(orch.functions.challengeK(REQ_ID, idx).call())
        if k_cur == 0:
            send_tx(orch.functions.finalizeChallenges(REQ_ID, idx, K), min_gas=200_000)
            print(f"   finalized K={K}")
        else:
            print(f"   challenges already finalized, K={k_cur}")
            K = k_cur
    except Exception as e:
        print("   finalizeChallenges failed:", explain_web3_error(e))
        return

    for i in range(K):
        step_idx = int(orch.functions.getChallenge(REQ_ID, idx, i).call())
        ws, we = pairs[step_idx]
        samp_idx = step_idx % N
        proof_samp_nodes = merkle_proof_sorted(samp_layers, samp_idx)
        proof_tr_nodes   = merkle_proof_sorted(tr_layers, step_idx)
        try:
            send_tx(
                orch.functions.bindFinalWeights(
                    REQ_ID, idx, i,
                    list(ws), list(we),
                    samp_idx, tr_x0[samp_idx], tr_x1[samp_idx], tr_y[samp_idx],
                    [HexBytes(p) for p in proof_tr_nodes],
                    [HexBytes(p) for p in proof_samp_nodes],
                ),
                min_gas=480_000
            )
            print(f"   ✓ answered challenge i={i} (step={step_idx} using sample={samp_idx})")
        except Exception as e:
            print("   bindFinalWeights failed at i=%d:" % i, explain_web3_error(e))
            return

    acc_bps = int(round(10000.0 * test_acc))
    try:
        send_tx(
            orch.functions.submitResult(
                REQ_ID, idx,
                list(finalW),
                acc_bps
            ),
            min_gas=260_000
        )
        print("   ✓ submitted result")
    except Exception as e:
        print("   submitResult failed:", explain_web3_error(e))
        return

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

    closed, *_ = orch.functions.getResult(REQ_ID).call()
    if closed: break
    if not claimed_any:
        maybe_reassign_timeouts()
    time.sleep(1.0)

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
print(f"[worker:{addr}] settled. credit={w3.from_wei(my_credit, 'ether')} ETH, wins={my_win}, losses={my_lose}")

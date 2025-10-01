#!/usr/bin/env python3
import json, os, pathlib, time, re, subprocess, uuid
from typing import Optional, List, Tuple
from decimal import Decimal, ROUND_HALF_UP
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

TRAIN_CSV = os.getenv("TRAIN_CSV",   "client/train.csv")
HOLD_CSV  = os.getenv("HOLDOUT_CSV", "client/dataset.csv")

SNARKJS = os.getenv("SNARKJS_BIN", "npx snarkjs")

def _env_artifact(*names: str, default: str) -> str:
    """
    Read an artifact path from env and sanitize it:
    - trim quotes/outer whitespace
    - collapse accidental spaces next to '/', '_' or '.' (common Makefile slip)
    """
    val = None
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            val = v
            break
    if val is None:
        val = default
    s = val.strip().strip('"').strip("'")
    s = re.sub(r'\s+(?=[_/\.])|(?<=[_/\.])\s+', '', s)
    return s

# Prefer ZK_* envs; fall back to ACC_*; finally fall back to defaults in ../circuits
ACC_WASM = _env_artifact(
    "ZK_ACC_WASM", "ACC_WASM",
    default="../circuits/MlpHoldoutAcc_256_js/MlpHoldoutAcc_256.wasm"
)
ACC_ZKEY = _env_artifact(
    "ZK_ACC_ZKEY", "ACC_ZKEY",
    default="../circuits/MlpHoldoutAcc_256_final.zkey"
)

# Normalize to absolute paths so cwd changes don’t break us
ACC_WASM = str(pathlib.Path(ACC_WASM).resolve())
ACC_ZKEY = str(pathlib.Path(ACC_ZKEY).resolve())
print(f"   zk artifacts: wasm={ACC_WASM}  zkey={ACC_ZKEY}")

# Claim/reassign tuning
ENABLE_REASSIGN = os.getenv("ENABLE_REASSIGN", "1") == "1"
CLAIM_BURST = int(os.getenv("CLAIM_BURST", "3"))
CLAIM_BURST_BACKOFF_MS = int(os.getenv("CLAIM_BURST_BACKOFF_MS", "200"))
AUTO_WITHDRAW = os.getenv("AUTO_WITHDRAW", "1") == "1"

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_csv(path: str) -> str:
    p = pathlib.Path(path)
    if p.is_file(): return str(p)
    p2 = pathlib.Path("..") / path
    if p2.is_file(): return str(p2)
    return str(p)

def _find_abi_path() -> pathlib.Path:
    env_path = os.getenv("ORCH_ABI_PATH")
    cands = []
    if env_path: cands.append(pathlib.Path(env_path))
    here = pathlib.Path.cwd()
    cands += [
        here / "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json",
        here.parent / "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json",
        here / "out/AiOrchestrator.sol/AiOrchestrator.json",
        here.parent / "out/AiOrchestrator.sol/AiOrchestrator.json",
    ]
    for p in cands:
        if p.is_file(): return p
    raise FileNotFoundError("Could not locate AiOrchestrator ABI (run `pnpm hardhat compile`).")

assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)

ABI_PATH = _find_abi_path()
with open(ABI_PATH, "r", encoding="utf-8") as f:
    orch_abi = json.load(f)["abi"]

orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR), abi=orch_abi)
addr = acct.address

ZERO = "0x0000000000000000000000000000000000000000"
def eth(wei: int) -> str: return f"{w3.from_wei(int(wei), 'ether')} ETH"

# ──────────────────────────────────────────────────────────────────────────────
# MODEL 2→4→1 (int) — mirrors AiOrchestrator.sol
# ──────────────────────────────────────────────────────────────────────────────
CAP_I = 127

def lr_bucket(lr_ppm: int) -> int:
    L = 1 + (int(lr_ppm) // 20_000)
    return 8 if L > 8 else (1 if L < 1 else L)

def sgn(v: int) -> int: return 1 if v > 0 else (-1 if v < 0 else 0)
def sat_i(x: int) -> int:
    if x >  CAP_I: return CAP_I
    if x < -CAP_I: return -CAP_I
    return x

def apply_one_step_mlp(W: Tuple[int, ...], x0: int, x1: int, y: int, L: int) -> Tuple[int, ...]:
    a = list(W)
    s = [0,0,0,0]
    for j in range(4):
        base = j*2
        pre = a[base+0]*x0 + a[base+1]*x1 + a[8+j]
        s[j] = 1 if pre >= 0 else 0
    z = a[16]
    for j in range(4): z += a[12 + j] * s[j]
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
                d = e * L * sgn(Vold[j])
                base = j*2
                a[base+0] = sat_i(a[base+0] + d * x0)
                a[base+1] = sat_i(a[base+1] + d * x1)
                a[8 + j]  = sat_i(a[8 + j]  + d)
    return tuple(a)

def _map_byte_signed(b: int, limit: int) -> int:
    return int(b % (2*limit + 1)) - limit

def _u256(x: int) -> bytes:
    return int(x).to_bytes(32, byteorder="big", signed=False)

def _det_init_w(req_id: int, idx: int, training_root: bytes) -> Tuple[int,...]:
    seed_input = b"init/" + _u256(req_id) + _u256(idx) + bytes(training_root)
    seed = Web3.keccak(seed_input)
    out: List[int] = [0]*17
    limits = [3]*8 + [6]*4 + [2]*4 + [2]
    counter = 0
    buf = Web3.keccak(seed + _u256(counter))
    pos = 0
    for k in range(17):
        if pos >= 32:
            counter += 1
            buf = Web3.keccak(seed + _u256(counter))
            pos = 0
        out[k] = _map_byte_signed(buf[pos], limits[k]); pos += 1
    return tuple(out)

def forward(W: Tuple[int,...], x0: int, x1: int) -> int:
    s = []
    for j in range(4):
        base = j*2
        pre = W[base+0]*x0 + W[base+1]*x1 + W[8+j]
        s.append(1 if pre >= 0 else 0)
    z = W[16]
    for j in range(4): z += W[12+j] * s[j]
    return 1 if z >= 0 else 0

def accuracy_on(W: Tuple[int,...], xs0: List[int], xs1: List[int], ys: List[int]) -> float:
    correct = 0
    for x0, x1, y in zip(xs0, xs1, ys):
        correct += int(forward(W, x0, x1) == y)
    return correct / len(xs0)

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MERKLE (sorted‑pair). 0..15 JS‑half‑up quantization
# ──────────────────────────────────────────────────────────────────────────────
def quant01_to_q(v) -> int:
    f = Decimal(str(v).strip())
    if f < 0: f = Decimal(0)
    if f > 1: f = Decimal(1)
    q = int((f * Decimal(15)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    return 0 if q < 0 else (15 if q > 15 else q)

def _cells(line: str) -> List[str]:
    return re.split(r"[,;\s]+", line.strip())

def load_csv2(path: str):
    xs0: List[int] = []; xs1: List[int] = []; ys: List[int] = []
    with open(_resolve_csv(path), "r", encoding="utf-8") as f:
        for i, raw in enumerate(f.read().splitlines()):
            if not raw: continue
            line = raw.strip()
            if not line or line.startswith("#"): continue
            cells = [c for c in _cells(line) if c]
            if len(cells) < 3: continue
            if i == 0 and (not cells[0].replace(".","",1).isdigit()):  # header
                continue
            x0 = quant01_to_q(cells[0]); x1 = quant01_to_q(cells[1]); y = int(cells[2])
            if y not in (0,1): raise ValueError(f"y must be 0/1 at row {i+1}")
            xs0.append(x0); xs1.append(x1); ys.append(y)
    if not xs0: raise RuntimeError(f"empty/invalid csv {path}")
    return xs0, xs1, ys

def h256(b: bytes) -> bytes: return Web3.keccak(b)
def b32u(x: int) -> bytes: return int(x).to_bytes(32, "big", signed=False)
def b32i(x: int) -> bytes: return int(x).to_bytes(32, "big", signed=True)

def hash_pair_sorted(a: bytes, b: bytes) -> bytes:
    return h256(a + b) if a < b else h256(b + a)

def build_merkle_layers_sorted(leaves: List[bytes]) -> List[List[bytes]]:
    if not leaves: return [[]]
    layers = [leaves[:]]
    while len(layers[-1]) > 1:
        cur = layers[-1]; nxt: List[bytes] = []
        for i in range(0, len(cur), 2):
            if i+1 < len(cur): nxt.append(hash_pair_sorted(cur[i], cur[i+1]))
            else:              nxt.append(hash_pair_sorted(cur[i], cur[i]))
        layers.append(nxt)
    return layers

def build_merkle_root_sorted(leaves: List[bytes]) -> bytes:
    layers = build_merkle_layers_sorted(leaves)
    return b"\x00"*32 if not layers or not layers[-1] else layers[-1][0]

def merkle_proof_sorted(layers: List[List[bytes]], index: int) -> List[bytes]:
    if not layers or not layers[0]: return []
    path: List[bytes] = []; idx = index
    for level in layers[:-1]:
        sib = idx ^ 1
        path.append(level[sib] if sib < len(level) else level[idx])
        idx //= 2
    return path

def leaf_for_step(step_idx: int, ws: Tuple[int,...], we: Tuple[int,...]) -> bytes:
    hWs = h256(b"".join(b32i(v) for v in ws))
    hWe = h256(b"".join(b32i(v) for v in we))
    return h256(b32u(step_idx) + hWs + hWe)

def leaf_for_sample(idx: int, x0: int, x1: int, y: int) -> bytes:
    return h256(b32u(idx) + b32u(x0) + b32u(x1) + b32u(y))

# ──────────────────────────────────────────────────────────────────────────────
# TX helpers
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

def _fresh_nonce() -> int:
    return w3.eth.get_transaction_count(acct.address, "pending")

def send_tx(fn_call, *, value: int = 0, min_gas: int = 220_000, headroom_num: int = 15, headroom_den: int = 10):
    try:
        est = int(fn_call.estimate_gas({"from": acct.address, "value": value}))
        gas_limit = max(min_gas, est * headroom_num // headroom_den)
    except Exception as e:
        print("   gas estimation failed, defaulting:", explain_web3_error(e))
        gas_limit = max(min_gas, 3_000_000)

    attempts = 0
    while True:
        attempts += 1
        try:
            nonce = _fresh_nonce()
            tx = fn_call.build_transaction({
                "from":    acct.address,
                "gas":     gas_limit,
                "nonce":   nonce,
                "value":   value,
                "chainId": w3.eth.chain_id,
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.rawTransaction)
            rcpt = w3.eth.wait_for_transaction_receipt(h)
            return rcpt
        except Exception as e:
            msg = explain_web3_error(e).lower()
            retriable = (
                "nonce too low" in msg or "already known" in msg or "known transaction" in msg or
                "nonce too high" in msg or "can't be queued" in msg or
                "replacement" in msg or "underpriced" in msg or "fee cap" in msg
            )
            if retriable and attempts < 24:
                time.sleep(0.08)
                continue
            raise RuntimeError(msg) from e

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
print(f"[worker:{addr}] starting on request {REQ_ID}…")
print(f"   using TRAIN_CSV={_resolve_csv(TRAIN_CSV)}")
print(f"   using HOLD_CSV ={_resolve_csv(HOLD_CSV)}")
bal0 = int(w3.eth.get_balance(addr))
print(f"   balance before: {eth(bal0)}")

# Load datasets
tr_x0, tr_x1, tr_y = load_csv2(TRAIN_CSV)
te_x0, te_x1, te_y = load_csv2(HOLD_CSV)

# Build Merkle for training
samp_leaves = [leaf_for_sample(i, tr_x0[i], tr_x1[i], tr_y[i]) for i in range(len(tr_x0))]
samp_layers = build_merkle_layers_sorted(samp_leaves)
samp_root   = build_merkle_root_sorted(samp_leaves)
print(f"   training root (computed by worker): {Web3.to_hex(samp_root)}")

# Build hold‑out Merkle (root‑mode)
hold_leaves = [leaf_for_sample(i, te_x0[i], te_x1[i], te_y[i]) for i in range(len(te_x0))]
hold_layers = build_merkle_layers_sorted(hold_leaves)

# Lobby join
try:
    if not orch.functions.joined(REQ_ID, addr).call():
        send_tx(orch.functions.joinLobby(REQ_ID))
        print("   ✓ joined lobby")
    else:
        print("   already joined lobby")
except Exception as e:
    print("   joinLobby failed:", explain_web3_error(e)); raise SystemExit(1)

# Wait for start
while True:
    try:
        needed, joined_, ready = orch.functions.lobbyCounts(REQ_ID).call()
        print(f"   lobby: {joined_}/{needed} ready={ready}")
        if ready: break
    except Exception:
        pass
    time.sleep(1.0)

# bond parameters
try:
    bond_wei   = int(orch.functions.CLAIM_BOND_WEI().call())
except Exception:
    bond_wei = int(Web3.to_wei(0.005, "ether"))

def list_free_task_indices() -> List[int]:
    n = len(orch.functions.getSpace(REQ_ID).call())
    return [i for i in range(n)
            if orch.functions.taskOwner(REQ_ID, i).call() == ZERO
            and int(orch.functions.taskAcc(REQ_ID, i).call()) == 0]

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
    if not evs:
        print("   TaskClaimed event missing; skipping.")
        return None
    idx = int(evs[-1]["args"]["idx"])

    owner = orch.functions.taskOwner(REQ_ID, idx).call()
    if owner.lower() != addr.lower():
        print(f"   lost claim for idx={idx} (owner changed to {owner}); skipping.")
        return None
    return idx

# Helpers to pack limbs (Python mirror of Circom)
def _pack_bits_256(bits: List[int]) -> List[int]:
    limbs = []
    for g in range(8):
        base = g * 32
        limb = 0
        for k in range(32):
            limb |= (int(bits[base + k]) & 1) << k
        limbs.append(limb)
    return limbs

def _pack_u4_256(vals: List[int]) -> List[int]:
    limbs = []
    for g in range(8):
        base = g * 32
        limb = 0
        for k in range(32):
            limb |= (int(vals[base + k]) & 15) << (4 * k)
        limbs.append(limb)
    return limbs

# Always serialize numeric values to decimal strings (avoid IEEE-754 rounding)
def _s(x: int) -> str: return str(int(x))
def _s_arr(xs: List[int]) -> List[str]: return [str(int(v)) for v in xs]

def do_task(idx: int, lr_ppm: int, steps: int):
    # Deterministic init using the client‑set training root (must match!)
    W0 = _det_init_w(REQ_ID, idx, samp_root)

    # Train and collect transcript pairs
    L = lr_bucket(lr_ppm)
    W = W0
    pairs: List[Tuple[Tuple[int,...], Tuple[int,...]]] = []
    for _ in range(int(steps)):
        for j in range(len(tr_x0)):
            ws = W
            W  = apply_one_step_mlp(W, tr_x0[j], tr_x1[j], tr_y[j], L)
            pairs.append((ws, W))
    finalW = W

    train_acc = accuracy_on(finalW, tr_x0, tr_x1, tr_y)
    test_acc  = accuracy_on(finalW, te_x0, te_x1, te_y)
    print(f"   [task {idx}] trained (bucket={lr_bucket(lr_ppm)}) → "
          f"train {100*train_acc:.2f}% | holdout {100*test_acc:.2f}%")

    # Commit transcript root
    leaves = [leaf_for_step(i, ws, we) for i, (ws,we) in enumerate(pairs)]
    tr_layers = build_merkle_layers_sorted(leaves)
    tr_root   = build_merkle_root_sorted(leaves)
    total     = len(leaves)
    print(f"   committed transcript root={Web3.to_hex(tr_root)} totalSteps={total}")

    owner = orch.functions.taskOwner(REQ_ID, idx).call()
    if owner.lower() != addr.lower():
        print(f"   ownership lost before commit (owner {owner}); skipping task {idx}.")
        return

    try:
        current = orch.functions.trRoot(REQ_ID, idx).call()
        zero = (isinstance(current, (bytes, bytearray, HexBytes)) and int.from_bytes(current, "big") == 0) \
               or (isinstance(current, str) and current in ("0x" + "00"*32, "0x"))
        if zero:
            send_tx(orch.functions.commitTranscript(REQ_ID, idx, HexBytes(tr_root), total), min_gas=200_000)
        else:
            print("   transcript already committed on-chain.")
    except Exception as e:
        msg = explain_web3_error(e)
        if "already committed" in msg: print("   transcript already committed on-chain.")
        else: raise

    # Finalize & answer training challenges
    Ktr = 3
    try:
        k_cur = int(orch.functions.challengeK(REQ_ID, idx).call())
        if k_cur == 0:
            send_tx(orch.functions.finalizeChallenges(REQ_ID, idx, Ktr), min_gas=200_000)
            print(f"   finalized K={Ktr}")
        else:
            print(f"   challenges already finalized, K={k_cur}")
            Ktr = k_cur
    except Exception as e:
        print("   finalizeChallenges failed:", explain_web3_error(e)); return

    for i in range(Ktr):
        step_idx = int(orch.functions.getChallenge(REQ_ID, idx, i).call())
        ws, we = pairs[step_idx]
        samp_idx = step_idx % len(tr_x0)
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
            print("   bindFinalWeights failed at i=%d:" % i, explain_web3_error(e)); return

    # Prepare hold‑out sampling
    H_len = int(orch.functions.datasetLength(REQ_ID).call())
    Kcfg  = int(orch.functions.HOLDOUT_K().call())
    K = min(Kcfg, H_len)

    idxs: List[int] = []
    hx0:  List[int] = []
    hx1:  List[int] = []
    hy:   List[int] = []
    mask: List[int] = []
    proofs_concat: List[HexBytes] = []
    sizes: List[int] = []

    correct = 0
    for i in range(K):
        sidx = int(orch.functions.getHoldoutChallenge(REQ_ID, idx, i).call())
        idxs.append(sidx)
        x0 = te_x0[sidx]; x1 = te_x1[sidx]; y = te_y[sidx]
        hx0.append(x0); hx1.append(x1); hy.append(y); mask.append(1)
        pnodes = merkle_proof_sorted(hold_layers, sidx)
        sizes.append(len(pnodes))
        proofs_concat.extend([HexBytes(p) for p in pnodes])
        if forward(finalW, x0, x1) == y: correct += 1

    # Pad to HOLDOUT_K (masked rows)
    while len(hx0) < Kcfg:
        idxs.append(0)
        hx0.append(0); hx1.append(0); hy.append(0); mask.append(0)
        sizes.append(0)

    acc_bps = (correct * 10000) // K

    # Pack public limbs
    mask_p = _pack_bits_256(mask)
    y_p    = _pack_bits_256(hy)
    x0_p   = _pack_u4_256(hx0)
    x1_p   = _pack_u4_256(hx1)

    # Per‑attempt isolated temp directory (avoid cross‑process clobber)
    def _unique_tmp_dir(base: pathlib.Path, req_id: int, task_idx: int) -> pathlib.Path:
        rnd = uuid.uuid4().hex[:8]
        d = base / f"proof-{req_id}-{task_idx}-{os.getpid()}-{int(time.time()*1e6)}-{rnd}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _prove_and_verify_once(tmp_dir: pathlib.Path) -> Tuple[bool, Optional[Tuple[list, list, list]], Optional[list]]:
        """
        Run wtns+prove once in tmp_dir. Return (ok, (a,b,c), public_signals) or (False, None, None)
        on mismatch or failure.
        """
        # Build an input.json where *all* numbers are decimal strings
        input_json = {
            # PUBLIC inputs
            "acc_bps": _s(acc_bps),
            "w_abs":   _s_arr([abs(w) for w in finalW]),
            "w_sign":  _s_arr([1 if w < 0 else 0 for w in finalW]),
            "mask_p":  _s_arr(mask_p),
            "x0_p":    _s_arr(x0_p),
            "x1_p":    _s_arr(x1_p),
            "y_p":     _s_arr(y_p),
            # PRIVATE witness arrays
            "mask": _s_arr(mask),
            "x0":   _s_arr(hx0),
            "x1":   _s_arr(hx1),
            "y":    _s_arr(hy),
        }
        with open(tmp_dir / "input.json", "w") as f:
            json.dump(input_json, f)

        def _run(cmd: str):
            print("   [zk] $", cmd)
            c = subprocess.run(cmd, shell=True, cwd=str(tmp_dir), capture_output=True, text=True)
            if c.returncode != 0:
                print(c.stdout); print(c.stderr)
                raise RuntimeError(f"zk cmd failed: {cmd}")
            return c

        cmd_wtns  = f'{SNARKJS} wtns calculate "{ACC_WASM}" "input.json" "witness.wtns"'
        cmd_prove = f'{SNARKJS} groth16 prove "{ACC_ZKEY}" "witness.wtns" "proof.json" "public.json"'

        _run(cmd_wtns)
        _run(cmd_prove)

        # Read proof & public signals from THIS tmp_dir
        with open(tmp_dir / "proof.json", "r") as f:
            pr = json.load(f)

        def _hex2int(x): return int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)

        a = [_hex2int(pr["pi_a"][0]), _hex2int(pr["pi_a"][1])]
        b = [
            [_hex2int(pr["pi_b"][0][1]), _hex2int(pr["pi_b"][0][0])],  # Fp2 order swap c1,c0
            [_hex2int(pr["pi_b"][1][1]), _hex2int(pr["pi_b"][1][0])],
        ]
        c = [_hex2int(pr["pi_c"][0]), _hex2int(pr["pi_c"][1])]

        with open(tmp_dir / "public.json", "r") as f:
            pub_raw = json.load(f)
        if isinstance(pub_raw, list):
            public_signals = [int(s) for s in pub_raw]
        else:
            arr = pub_raw.get("publicSignals", pub_raw.get("pubSignals"))
            if arr is None:
                print("   [zk] ERROR: public.json missing publicSignals/pubSignals")
                return (False, None, None)
            public_signals = [int(s) for s in arr]

        expected_pub = 1 + 17 + 17 + 8 + 8 + 8 + 8  # 67
        print(f"   [zk] public signals len = {len(public_signals)} (expected {expected_pub})")
        if len(public_signals) != expected_pub:
            print("   [zk] ERROR: public signals length mismatch.")
            return (False, None, None)

        # Build the exact input vector (contract ABI order)
        input_vec = []
        input_vec.append(int(acc_bps))
        input_vec += [abs(int(w)) for w in finalW]               # w_abs[17]
        input_vec += [1 if w < 0 else 0 for w in finalW]         # w_sign[17]
        input_vec += mask_p                                      # mask_p[8]
        input_vec += x0_p                                        # x0_p[8]
        input_vec += x1_p                                        # x1_p[8]
        input_vec += y_p                                         # y_p[8]

        # Debug mismatch (should not happen now that we stringify)
        mismatch_at = next((i for i, (a0, b0) in enumerate(zip(input_vec, public_signals)) if int(a0) != int(b0)), None)
        if mismatch_at is not None:
            lo = max(0, mismatch_at - 3)
            hi = min(len(input_vec), mismatch_at + 4)
            print(f"   [zk] MISMATCH at index {mismatch_at}: input_vec={input_vec[mismatch_at]} public={public_signals[mismatch_at]}")
            print(f"        context [{lo}:{hi}]:")
            print(f"        input_vec slice : {input_vec[lo:hi]}")
            print(f"        public.json slice: {public_signals[lo:hi]}")
            print(f"        mask_p[0..2]={mask_p[:3]}  x0_p[0..2]={x0_p[:3]}  x1_p[0..2]={x1_p[:3]}  y_p[0..2]={y_p[:3]}")
            return (False, None, None)

        # Verifier preflight
        try:
            ver_addr = orch.functions.accVerifier().call()
            ver_abi = [{
                "inputs":[
                    {"internalType":"uint256[2]","name":"a","type":"uint256[2]"},
                    {"internalType":"uint256[2][2]","name":"b","type":"uint256[2][2]"},
                    {"internalType":"uint256[2]","name":"c","type":"uint256[2]"},
                    {"internalType":"uint256[67]","name":"input","type":"uint256[67]"}
                ],
                "name":"verifyProof","outputs":[{"internalType":"bool","name":"","type":"bool"}],
                "stateMutability":"view","type":"function"
            }]
            acc = w3.eth.contract(address=w3.to_checksum_address(ver_addr), abi=ver_abi)
            print("   [zk] verifier preflight call…")
            ok = acc.functions.verifyProof(a, b, c, input_vec).call()
            print(f"   [zk] verifier preflight result: {ok}")
            if not ok:
                print("   [zk] Proof invalid against verifier.")
                return (False, None, None)
        except Exception as e:
            print("   [zk] verifier preflight reverted:", explain_web3_error(e))
            return (False, None, None)

        return (True, (a, b, c), input_vec)

    # Prove with a few retries (fresh tmp dir each time)
    attempts = int(os.getenv("ZK_PROVE_MAX_ATTEMPTS", "3"))
    base_tmp = pathlib.Path("tmp"); base_tmp.mkdir(parents=True, exist_ok=True)

    a = b = c = None
    input_vec = None
    ok_any = False
    for attempt in range(1, attempts+1):
        tmp_dir = _unique_tmp_dir(base_tmp, REQ_ID, idx)
        print(f"   [zk] proving attempt {attempt}/{attempts} in {tmp_dir}")
        try:
            ok, proof_tuple, pub_vec = _prove_and_verify_once(tmp_dir)
            if ok and proof_tuple:
                (a, b, c) = proof_tuple
                input_vec = pub_vec
                ok_any = True
                if os.getenv("ZK_TMP_KEEP", "0") != "1":
                    try:
                        for p in tmp_dir.glob("*"): p.unlink()
                        tmp_dir.rmdir()
                    except Exception:
                        pass
                break
            else:
                print("   [zk] attempt failed; will retry with a fresh directory.")
        except Exception as e:
            print("   [zk] attempt error:", e)

    if not ok_any:
        print("   [zk] giving up after retries; will let TTL/timeout reassign this task.")
        return

    # Submit to orchestrator
    try:
        send_tx(
            orch.functions.submitResultZK(
                REQ_ID, idx,
                list(finalW),
                int(acc_bps),
                idxs, mask, hx0, hx1, hy,
                proofs_concat, sizes,
                a, b, c
            ),
            min_gas=900_000
        )
        print(f"   ✓ submitted result (zk) — K={K} → {acc_bps} bps")
    except Exception as e:
        print("   submitResultZK failed:", explain_web3_error(e))


# ──────────────────────────────────────────────────────────────────────────────
# CLAIM/WORK LOOP
# ──────────────────────────────────────────────────────────────────────────────
while True:
    idx = None
    for attempt in range(CLAIM_BURST):
        idx = try_claim_one()
        if idx is not None: break
        time.sleep(CLAIM_BURST_BACKOFF_MS / 1000.0)

    if idx is not None:
        hp = orch.functions.getSpace(REQ_ID).call()[idx]
        lr_ppm, steps = int(hp[0]), int(hp[1])
        print(f"   claimed task {idx} (lr={lr_ppm}, steps={steps}) with bond {eth(int(bond_wei))}")
        try: do_task(idx, lr_ppm, steps)
        except Exception as e: print(f"   task {idx} failed:", e)
        continue

    if ENABLE_REASSIGN:
        try:
            n = len(orch.functions.getSpace(REQ_ID).call())
            now = int(w3.eth.get_block("latest")["timestamp"])
            claim_ttl = int(orch.functions.CLAIM_TTL().call())
            stall_ttl = int(orch.functions.STALL_TTL().call())
            progress_ttl = int(orch.functions.PROGRESS_TTL().call())
            proven = sum(1 for i in range(n) if orch.functions.taskAcc(REQ_ID, i).call() != 0)
            majority = (proven * 2) > n
            base = claim_ttl if majority else stall_ttl
            for i in range(n):
                owner = orch.functions.taskOwner(REQ_ID, i).call()
                if owner and owner.lower() == addr.lower(): continue
                if orch.functions.taskAcc(REQ_ID, i).call() != 0: continue
                tr = orch.functions.trRoot(REQ_ID, i).call()
                if isinstance(tr, (bytes, bytearray, HexBytes)):
                    if int.from_bytes(tr, "big") != 0: continue
                elif isinstance(tr, str) and tr not in ("0x"+"00"*32, "0x"):
                    continue
                t0 = int(orch.functions.claimedAt(REQ_ID, i).call())
                lp = int(orch.functions.lastProgressAt(REQ_ID, i).call())
                if t0 == 0: continue
                has_progress = (lp != 0 and lp > t0)
                time_ok = (now >= t0 + base) or (has_progress and now >= lp + progress_ttl)
                if not time_ok: continue
                try: orch.functions.reassignTimedOut(REQ_ID, i).call({"from": addr})
                except Exception: continue
                try: send_tx(orch.functions.reassignTimedOut(REQ_ID, i), min_gas=180_000); print(f"   reassignTimedOut({i})")
                except Exception: pass
        except Exception:
            pass

    closed, *_ = orch.functions.getResult(REQ_ID).call()
    if closed: break
    time.sleep(1.0)

# Settlement + optional withdraw
closed, bestAcc, winTaskCount, perWinnerTaskWei, perLoserTaskWei = orch.functions.getResult(REQ_ID).call()
my_total, my_proven, my_wins, my_losses = orch.functions.taskStatsOf(REQ_ID, addr).call()
my_credit_before = int(orch.functions.credit(REQ_ID, addr).call())

print(f"[worker:{addr}] stats: total={my_total} proven={my_proven} wins={my_wins} losses={my_losses}")
print(f"   settlement: bestAcc={bestAcc} bps | perWin={eth(perWinnerTaskWei)} perLose={eth(perLoserTaskWei)}")
print(f"   credit on-chain (pre-withdraw): {eth(my_credit_before)}")

if AUTO_WITHDRAW and my_credit_before > 0:
    try: send_tx(orch.functions.withdraw(REQ_ID), min_gas=120_000)
    except Exception as e: print("   withdraw failed:", explain_web3_error(e))

print(f"[worker:{addr}] settled.")

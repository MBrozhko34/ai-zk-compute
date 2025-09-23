#!/usr/bin/env python3
"""
Track B+ worker with deterministic init (by steps) and robust guards.
Prints TRAIN vs TEST (hold-out) accuracy and performs Track-B+ flow.
"""

import json, os, subprocess, tempfile, pathlib, time
from typing import Optional, List, Tuple, Union

from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hexbytes import HexBytes

# ───────── env
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0) or 0)

DATASET_CSV = os.getenv("DATASET_CSV", "../client/dataset.csv")

ZKEY      = pathlib.Path("../circuits/xor_final.zkey")
VK_JSON   = pathlib.Path("../circuits/verification_key.json")
CIRC_DIR  = pathlib.Path("../circuits/XorCircuit_js")
ABI_PATH  = pathlib.Path("../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

# ───────── chain handles
w3: Web3 = Web3(HTTPProvider(RPC_URL))
acct: LocalAccount = Account.from_key(PRIV)
orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR),
                       abi=json.load(open(ABI_PATH))["abi"])
addr = acct.address

# ───────── utilities

def is_zero_bytes32(v: Union[bytes, HexBytes, str, int]) -> bool:
    if isinstance(v, int):
        return v == 0
    if isinstance(v, (bytes, bytearray, HexBytes)):
        return int.from_bytes(bytes(v), "big") == 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("0x"): s = s[2:]
        if s == "": return True
        return int(s, 16) == 0
    # unknown type — be conservative
    return False

# robust Groth16 calldata parser (unchanged)
def parse_groth16_calldata(raw: str):
    txt = raw.strip()
    def _try(s):
        try: return json.loads(s)
        except: return None
    parsed = _try(txt) or _try(f"[{txt}]")
    if parsed is None: raise RuntimeError("cannot parse groth16 calldata")
    if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(x, list) for x in parsed):
        a_raw, b_raw, c_raw, inputs_raw = parsed
    elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list) and len(parsed[0]) == 4:
        a_raw, b_raw, c_raw, inputs_raw = parsed[0]
    else:
        raise RuntimeError("unexpected groth16 calldata shape")

    def _to_int(x):
        if isinstance(x, int): return x
        if isinstance(x, str):
            s = x.strip()
            return int(s, 16) if s.lower().startswith("0x") else int(s)
        raise TypeError(f"bad numeric {type(x)}")
    a = [_to_int(x) for x in a_raw]
    b = [[_to_int(x) for x in row] for row in b_raw]
    c = [_to_int(x) for x in c_raw]
    inputs = [_to_int(x) for x in inputs_raw]
    if len(a)!=2 or len(c)!=2 or len(b)!=2 or any(len(row)!=2 for row in b):
        raise RuntimeError("bad (a,b,c) shapes")
    return a, b, c, inputs

# training dynamics (unchanged)
def step1(v: int) -> int: return 1 if v >= 1 else 0     # Hidden-0 gate
def step2(v: int) -> int: return 1 if v >= 2 else 0     # Hidden-1 gate
def lr_bucket(lr_ppm: int) -> int: return 1 + (1 if lr_ppm >= 50_000 else 0) + (1 if lr_ppm >= 100_000 else 0)
def sat_add3_floor(x: int, inc: int, dec: int, delta: int, floor: int = 0) -> int:
    x_cap = x + inc * delta
    if x_cap > 3: x_cap = 3
    sub = dec * delta
    if x_cap - sub < floor: return floor
    return x_cap - sub

def seeded_init(lr_ppm: int, steps: int) -> Tuple[int,int,int,int,int,int,int]:
    # steps >= 60: good XOR init (bO=0). else: bO=1 (degenerate).
    w0=w1=w2=w3=1; b0=0; b1=0; bO=(0 if steps >= 60 else 1)
    return (w0,w1,w2,w3,b0,b1,bO)

def train_collect(lr_ppm: int, steps: int, max_epochs: int = 300):
    w0,w1,w2,w3,b0,b1,bO = seeded_init(lr_ppm, steps)
    X = [(0,0),(1,0),(0,1),(1,1)]; Y = [0,1,1,0]
    L = lr_bucket(lr_ppm)
    T = min(int(steps), max_epochs)
    pairs: List[Tuple[Tuple[int,...], Tuple[int,...]]] = []
    for _e in range(T):
        for (x0,x1), y in zip(X, Y):
            ws = (w0,w1,w2,w3,b0,b1,bO)
            s0 = step1(w0*x0 + w1*x1 + b0)
            s1 = step2(w2*x0 + w3*x1 + b1)
            o  = 1 if (s0 - s1 + bO) >= 1 else 0
            pos = (1 - o) * y
            neg = (1 - y) * o
            d0 = L * x0; d1 = L * x1
            w0 = sat_add3_floor(w0, pos, neg, d0, 0)
            w1 = sat_add3_floor(w1, pos, neg, d1, 0)
            b0 = sat_add3_floor(b0, pos, neg, L,  0)
            w2 = sat_add3_floor(w2, neg, pos, d0, 0)
            w3 = sat_add3_floor(w3, neg, pos, d1, 0)
            b1 = sat_add3_floor(b1, neg, pos, L,  0)
            bO = sat_add3_floor(bO, pos, neg, L,  0)
            we = (w0,w1,w2,w3,b0,b1,bO)
            pairs.append((ws, we))
    correct=0
    for (x0,x1), y in zip(X, Y):
        s0 = step1(w0*x0 + w1*x1 + b0)
        s1 = step2(w2*x0 + w3*x1 + b1)
        o  = 1 if (s0 - s1 + bO) >= 1 else 0
        correct += int(o == y)
    acc_bps = correct * 2500
    finalW  = (w0,w1,w2,w3,b0,b1,bO)
    return acc_bps, finalW, pairs

# local hold-out reading (for printing only)
def load_holdout_csv(path: str) -> List[Tuple[int,int,int]]:
    rows: List[Tuple[int,int,int]] = []
    try:
        with open(path, "r") as f:
            for i, raw in enumerate(f.read().splitlines()):
                line = (raw or "").strip()
                if not line or line.startswith("#"): continue
                parts = [c for c in line.replace(",", " ").replace(";", " ").split() if c]
                if len(parts) < 3: continue
                if i == 0 and (not parts[0].isdigit() or not parts[1].isdigit() or not parts[2].isdigit()):
                    continue
                a,b,c = int(parts[0]), int(parts[1]), int(parts[2])
                if a in (0,1) and b in (0,1) and c in (0,1):
                    rows.append((a,b,c))
    except Exception:
        rows = [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]
    if not rows:
        rows = [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]
    return rows

def acc_on(weights: Tuple[int,...], rows: List[Tuple[int,int,int]]) -> float:
    w0,w1,w2,w3,b0,b1,bO = weights
    correct = 0
    for (x0,x1,y) in rows:
        s0 = step1(w0*x0 + w1*x1 + b0)
        s1 = step2(w2*x0 + w3*x1 + b1)
        o  = 1 if (s0 - s1 + bO) >= 1 else 0
        correct += int(o == y)
    return (100.0 * correct) / max(1, len(rows))

# Merkle helpers (unchanged)
def h256(b: bytes) -> bytes: return Web3.keccak(b)
def bytes32_of_int(x: int) -> bytes: return x.to_bytes(32, "big", signed=False)
def hashW7(W: Tuple[int,...]) -> bytes: return h256(b"".join(bytes32_of_int(v) for v in W))
def leaf_for_step(step_idx: int, ws: Tuple[int,...], we: Tuple[int,...]) -> bytes:
    return h256(bytes32_of_int(step_idx) + hashW7(ws) + hashW7(we))
def hash_pair_sorted(a: bytes, b: bytes) -> bytes: return h256(a + b) if a < b else h256(b + a)
def build_merkle_root(leaves: List[bytes]) -> bytes:
    level = leaves[:]
    if not level: return b"\x00"*32
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            if i+1 < len(level): nxt.append(hash_pair_sorted(level[i], level[i+1]))
            else:                 nxt.append(hash_pair_sorted(level[i], level[i]))
        level = nxt
    return level[0]
def merkle_proof_sorted(leaves: List[bytes], index: int) -> List[bytes]:
    layers = [leaves[:]]
    while len(layers[-1]) > 1:
        cur = layers[-1]; nxt=[]
        for i in range(0, len(cur), 2):
            if i+1 < len(cur): nxt.append(hash_pair_sorted(cur[i], cur[i+1]))
            else:              nxt.append(hash_pair_sorted(cur[i], cur[i]))
        layers.append(nxt)
    path: List[bytes] = []
    idx = index
    for h in range(len(layers) - 1):
        cur = layers[h]; sib = idx ^ 1
        path.append(cur[sib] if sib < len(cur) else cur[idx])
        idx //= 2
    return path

# tx utils (unchanged)
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

TOPIC_ONCHAIN_ACC = w3.keccak(text="OnChainAccChecked(uint256,uint256,uint256,uint256,uint256)").hex()
TOPIC_FINALIZED   = w3.keccak(text="ChallengesFinalized(uint256,uint256,bytes32,uint256)").hex()
TOPIC_ACCEPTED    = w3.keccak(text="ProofAccepted(uint256,uint256,address,uint256)").hex()
TOPIC_BAD         = w3.keccak(text="BadProof(uint256,uint256,address,uint256)").hex()

def print_onchain_acc(rcpt):
    for log in rcpt.logs:
        if log["address"].lower() != ORCH_ADDR.lower(): continue
        if log["topics"][0].hex().lower() == TOPIC_ONCHAIN_ACC.lower():
            ev = orch.events.OnChainAccChecked().process_log(log)
            args = ev["args"]
            print(f"   on-chain inference: claimed {int(args['claimedAccBps'])/100:.2f}% "
                  f"vs recomputed {int(args['accOnChain'])/100:.2f}% on {int(args['rows'])} rows")

def print_accept_or_slash(rcpt, idx: int):
    accepted = False
    for log in rcpt.logs:
        if log["address"].lower() != ORCH_ADDR.lower(): continue
        t0 = log["topics"][0].hex().lower()
        if t0 == TOPIC_ACCEPTED.lower(): accepted = True
        if t0 == TOPIC_BAD.lower():      accepted = False
    if accepted: print(f"   ✓ accepted on-chain (task {idx})")
    else:        print(f"   ✗ rejected/slashed on-chain (task {idx})")

# helpers
def chain_now() -> int:
    return int(w3.eth.get_block("latest")["timestamp"])
def space_len() -> int:
    return len(orch.functions.getSpace(REQ_ID).call())
def count_proven(n: int) -> int:
    return sum(1 for i in range(n) if orch.functions.taskAcc(REQ_ID, i).call() != 0)
def answered_mask(idx: int) -> int:
    return int(orch.functions.answeredMask(REQ_ID, idx).call())

# ───────── main
bal0 = int(w3.eth.get_balance(addr))
print(f"[worker:{addr}] starting on request {REQ_ID}…")
print(f"   balance before: {w3.from_wei(bal0,'ether')} ETH")

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
        if any(x in msg for x in ["no tasks left", "closed", "not started", "bond", "holdout not set"]):
            return None
        print("   claimTask preflight failed:", msg); return None

    rcpt = send_tx(orch.functions.claimTask(REQ_ID), value=bond_wei, min_gas=230_000)
    evs = orch.events.TaskClaimed().process_receipt(rcpt)
    if evs: return int(evs[-1]["args"]["idx"])
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
        majority = (proven * 2) > n
        ttl = claim_ttl if majority else stall_ttl

        # If only one remains, do NOT reassign if we own it.
        if proven == n - 1:
            for i in range(n):
                acc = orch.functions.taskAcc(REQ_ID, i).call()
                if acc != 0:
                    continue
                owner = orch.functions.taskOwner(REQ_ID, i).call()
                if owner == ZERO or owner.lower() == addr.lower():   # <-- skip our own
                    continue
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

        # General case: still never reassign our own task; respect TTL.
        for i in range(n):
            acc = orch.functions.taskAcc(REQ_ID, i).call()
            if acc != 0: continue
            owner = orch.functions.taskOwner(REQ_ID, i).call()
            if owner == ZERO or owner.lower() == addr.lower():       # <-- skip our own
                continue
            t = orch.functions.claimedAt(REQ_ID, i).call()
            if t == 0: continue
            remain = (t + ttl) - now
            if remain > 0:
                # Optional: print only occasionally to reduce spam
                # print(f"   waiting TTL for idx {i}… {remain}s")
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


def do_task(idx: int, lr_ppm: int, steps: int):
    owner = orch.functions.taskOwner(REQ_ID, idx).call()
    if owner.lower() != addr.lower():
        print(f"   [task {idx}] not mine anymore; skipping.")
        return
    if int(orch.functions.taskAcc(REQ_ID, idx).call()) != 0:
        print(f"   [task {idx}] already proven; skipping.")
        return

    # 1) Train with transcript
    acc_bps, finalW, pairs = train_collect(lr_ppm, steps)
    train_acc_pct = acc_bps / 100.0
    hold = load_holdout_csv(DATASET_CSV)
    test_acc_pct = acc_on(finalW, hold)
    print(f"   [task {idx}] trained {min(steps,300)} epochs (bucket={lr_bucket(lr_ppm)})"
          f" → train acc {train_acc_pct:.2f}% | test acc {test_acc_pct:.2f}%")

    # 2) Build transcript Merkle (sorted)
    leaves = [leaf_for_step(i, ws, we) for i, (ws,we) in enumerate(pairs)]
    root   = build_merkle_root(leaves)
    total  = len(leaves)
    print(f"   committed transcript root={Web3.to_hex(root)} totalSteps={total}")

    # 2a) commit transcript if not already done
    try:
        current_root = orch.functions.trRoot(REQ_ID, idx).call()
        if is_zero_bytes32(current_root):
            send_tx(orch.functions.commitTranscript(REQ_ID, idx, root, total), min_gas=200_000)
        else:
            print("   transcript already committed on-chain.")
    except Exception as e:
        msg = explain_web3_error(e)
        if "already committed" not in msg and "task not yours" not in msg:
            raise

    # 2b) finalize challenges (K=3) if not already done
    try:
        Kcur = int(orch.functions.challengeK(REQ_ID, idx).call())
        if Kcur == 0:
            rcpt = send_tx(orch.functions.finalizeChallenges(REQ_ID, idx, 3), min_gas=200_000)
            for log in rcpt.logs:
                if log["address"].lower() != ORCH_ADDR.lower(): continue
                if log["topics"][0].hex().lower() == TOPIC_FINALIZED.lower():
                    ev = orch.events.ChallengesFinalized().process_log(log)
                    sd = ev["args"]["seed"]
                    print(f"   finalized seed={Web3.to_hex(sd)}  challengeK=3")
            Kcur = 3
        else:
            print(f"   challenges already finalized, K={Kcur}")
    except Exception as e:
        msg = explain_web3_error(e)
        if "already finalized" not in msg and "task not yours" not in msg:
            raise
        Kcur = int(orch.functions.challengeK(REQ_ID, idx).call())

    # 2c) answer challenges that remain
    for i in range(Kcur):
        if (answered_mask(idx) >> i) & 1:
            continue
        step_idx = int(orch.functions.getChallenge(REQ_ID, idx, i).call())
        ws, we = pairs[step_idx]
        proof_nodes = merkle_proof_sorted(leaves, step_idx)
        proof_hexes = [Web3.to_hex(p) for p in proof_nodes]
        send_tx(orch.functions.bindFinalWeights(REQ_ID, idx, i, list(ws), list(we), proof_hexes), min_gas=350_000)
        print(f"   ✓ answered challenge i={i} (step={step_idx})")

    # 3) SNARK witness
    w0,w1,w2,w3,b0,b1,bO = finalW
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp/"input.json").write_text(json.dumps({
            "lr": lr_ppm, "steps": steps, "acc_bps": acc_bps,
            "w0_pub": w0, "w1_pub": w1, "w2_pub": w2, "w3_pub": w3,
            "b0_pub": b0, "b1_pub": b1, "bO_pub": bO
        }, indent=2))

        subprocess.check_call([
            "node", str(CIRC_DIR/"generate_witness.js"),
            str(CIRC_DIR/"XorCircuit.wasm"),
            str(tmp/"input.json"), str(tmp/"witness.wtns")
        ])
        subprocess.check_call([
            "snarkjs", "groth16", "prove",
            str(ZKEY), str(tmp/"witness.wtns"),
            str(tmp/"proof.json"), str(tmp/"public.json")
        ])
        print("   verifying off-chain (groth16)…")
        subprocess.check_call([
            "snarkjs", "groth16", "verify",
            str(VK_JSON), str(tmp/"public.json"), str(tmp/"proof.json")
        ])
        print("   ✓ off-chain verification OK")

        raw = subprocess.check_output([
            "snarkjs", "zkey", "export", "soliditycalldata",
            str(tmp/"public.json"), str(tmp/"proof.json")
        ]).decode()

    a, b, c, inputs = parse_groth16_calldata(raw)
    if len(inputs) != 10:
        raise RuntimeError("expected 10 public inputs")

    owner_now = orch.functions.taskOwner(REQ_ID, idx).call()
    if owner_now.lower() != addr.lower():
        print(f"   [task {idx}] lost ownership before submit; aborting.")
        return
    if not orch.functions.trainingChecksPassed(REQ_ID, idx).call():
        print(f"   [task {idx}] training checks incomplete on-chain; aborting.")
        return

    rcpt = send_tx(orch.functions.submitProofOrSlash(REQ_ID, idx, a, b, c, inputs), min_gas=700_000)
    print("   ✓ SNARK submitted; contract rechecked accuracy")
    print_onchain_acc(rcpt)
    print_accept_or_slash(rcpt, idx)

# ───────── main loop over tasks
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

#!/usr/bin/env python3
"""Claim one task, train 2-2-1 XOR net, prove & submit (Groth16)."""
import json, os, subprocess, tempfile, pathlib
from web3 import Web3, HTTPProvider
from eth_account import Account
import torch, torch.nn as nn

# ───────── env
RPC_URL   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ORCH_ADDR = os.getenv("ORCH_ADDR")
PRIV      = os.getenv("PRIVATE_KEY")
REQ_ID    = int(os.getenv("REQUEST_ID", 0))

ZKEY      = pathlib.Path("../circuits/xor_final.zkey")
VK_JSON   = pathlib.Path("../circuits/verification_key.json")
CIRC_DIR  = pathlib.Path("../circuits/XorCircuit_js")
ABI_PATH  = pathlib.Path("../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
assert ORCH_ADDR and PRIV, "ORCH_ADDR & PRIVATE_KEY required"

# ───────── helpers (mirror circuit)
bin1   = lambda v: 1 if v >= 0 else 0       # clamp to {0,1}
clamp3 = lambda x: min(max(int(x), 0), 3)   # 0..3 for Step()

def xor_forward(x0, x1, wIH, bH, bO):
    h0 = clamp3(wIH[0]*x0 + wIH[1]*x1 + bH[0])
    h1 = clamp3(wIH[2]*x0 + wIH[3]*x1 + bH[1])
    s0 = 1 if h0 >= 1 else 0
    s1 = 1 if h1 >= 1 else 0
    o  = clamp3(s0 - s1 + bO)
    return 1 if o >= 1 else 0

# ───────── chain handles
w3   = Web3(HTTPProvider(RPC_URL))
acct = Account.from_key(PRIV)
orch = w3.eth.contract(address=w3.to_checksum_address(ORCH_ADDR),
                       abi=json.load(open(ABI_PATH))["abi"])

# ── 1) claim task
claim_tx = orch.functions.claimTask(REQ_ID).build_transaction({
    "from":   acct.address,
    "gas":    200_000,
    "gasPrice": w3.eth.gas_price,
    "nonce":  w3.eth.get_transaction_count(acct.address)
})
rcpt = w3.eth.wait_for_transaction_receipt(
    w3.eth.send_raw_transaction(acct.sign_transaction(claim_tx).rawTransaction)
)
idx = orch.events.TaskClaimed().process_receipt(rcpt)[0].args.idx
lr_ppm, steps = orch.functions.getSpace(REQ_ID).call()[idx]
print(f"[worker] idx={idx}, lr_ppm={lr_ppm}, steps={steps}")

# ── 2) train small MLP on XOR
X = torch.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

net = nn.Sequential(nn.Linear(2,2), nn.Sigmoid(),
                    nn.Linear(2,1), nn.Sigmoid())
opt = torch.optim.SGD(net.parameters(), lr_ppm/1_000_000)
crit = nn.BCELoss()
for _ in range(steps):
    opt.zero_grad(); crit(net(X),y).backward(); opt.step()

# ── 3) binarise & compute circuit-compatible accuracy
wIH = [bin1(v) for v in net[0].weight.detach().round().flatten().tolist()]  # 4 ints in {0,1}
bH  = [bin1(v) for v in net[0].bias.detach().round().tolist()]              # 2 ints in {0,1}
bO  = 1                                                                      # avoid negatives

preds   = [xor_forward(*xy, wIH, bH, bO) for xy in [(0,0),(1,0),(0,1),(1,1)]]
correct = sum(int(p==t) for p,t in zip(preds, [0,1,1,0]))
acc_bps = correct * 2500
print(f"   circuit-compatible acc = {correct}/4 → {acc_bps/100}%")

# ── 4) witness & proof
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    (tmp/"input.json").write_text(json.dumps({
        "lr": lr_ppm,
        "steps": steps,
        "acc_bps": acc_bps,
        "wIH": wIH,
        "bH":  bH,
        "bO":  bO
    }, indent=2))

    # generate witness and proof
    subprocess.check_call([
        "node", str(CIRC_DIR/"generate_witness.js"),
        str(CIRC_DIR/"XorCircuit.wasm"),
        str(tmp/"input.json"),
        str(tmp/"witness.wtns")
    ])
    subprocess.check_call([
        "snarkjs", "groth16", "prove", str(ZKEY),
        str(tmp/"witness.wtns"),
        str(tmp/"proof.json"),
        str(tmp/"public.json")
    ])

    # (important) verify locally before sending on-chain
    print("   verifying off-chain with snarkjs…")
    subprocess.check_call([
        "snarkjs", "groth16", "verify",
        str(VK_JSON), str(tmp/"public.json"), str(tmp/"proof.json")
    ])
    print("   ✓ off-chain verification OK")

    proof          = json.loads((tmp/"proof.json").read_text())
    public_signals = [int(x) for x in json.loads((tmp/"public.json").read_text())]

# ── 5) format proof for Solidity (SWAP pi_b LIMBS!) & submit
# a, c are direct; b needs inner-limb swap for the Solidity verifier
a = [int(proof["pi_a"][0]), int(proof["pi_a"][1])]
b = [
    [int(proof["pi_b"][0][1]), int(proof["pi_b"][0][0])],
    [int(proof["pi_b"][1][1]), int(proof["pi_b"][1][0])]
]
c = [int(proof["pi_c"][0]), int(proof["pi_c"][1])]

print("   public signals (lr, steps, acc_bps):", public_signals)

tx = orch.functions.submitProof(REQ_ID, a, b, c, public_signals).build_transaction({
    "from": acct.address,
    "gas":  600_000,
    "gasPrice": w3.eth.gas_price,
    "nonce": w3.eth.get_transaction_count(acct.address)
})
h = w3.eth.send_raw_transaction(acct.sign_transaction(tx).rawTransaction)
print("✓ submitProof tx", h.hex())

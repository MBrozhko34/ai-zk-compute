#!/usr/bin/env python3
"""Join lobby, wait for start, claim task, train XOR, prove (Groth16), auto-wait for rewards."""
import json, os, subprocess, tempfile, pathlib, time
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
bin1   = lambda v: 1 if v >= 0 else 0        # clamp to {0,1}
clamp3 = lambda x: min(max(int(x), 0), 3)    # 0..3 for Step()

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

def send_tx(tx):
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.rawTransaction)
    return w3.eth.wait_for_transaction_receipt(h)

addr = acct.address
print(f"[worker:{addr}] joining lobby for request {REQ_ID}…")

# 0) join the lobby (blocks on-chain until included)
join = orch.functions.joinLobby(REQ_ID).build_transaction({
    "from": addr, "gas": 150_000, "gasPrice": w3.eth.gas_price,
    "nonce": w3.eth.get_transaction_count(addr),
})
send_tx(join)

# 0b) poll until lobby is ready
while True:
    needed, joined, ready = orch.functions.lobbyCounts(REQ_ID).call()
    print(f"   lobby: {joined}/{needed} ready={ready}")
    if ready: break
    time.sleep(2)

# 1) claim assigned task index
claim = orch.functions.claimTask(REQ_ID).build_transaction({
    "from": addr, "gas": 150_000, "gasPrice": w3.eth.gas_price,
    "nonce": w3.eth.get_transaction_count(addr),
})
rcpt = send_tx(claim)
# decode topic if you want; simpler: ask contract who you are
idx = orch.functions.taskOf(REQ_ID, addr).call()
lr_ppm, steps = orch.functions.getSpace(REQ_ID).call()[idx]
print(f"[worker:{addr}] idx={idx}, lr_ppm={lr_ppm}, steps={steps}")

# 2) train small MLP on XOR
X = torch.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

net = nn.Sequential(nn.Linear(2,2), nn.Sigmoid(),
                    nn.Linear(2,1), nn.Sigmoid())
opt = torch.optim.SGD(net.parameters(), lr_ppm/1_000_000)
crit = nn.BCELoss()
for _ in range(steps):
    opt.zero_grad(); crit(net(X),y).backward(); opt.step()

# 3) binarise & compute circuit-compatible accuracy
wIH = [bin1(v) for v in net[0].weight.detach().round().flatten().tolist()]  # 4 ints in {0,1}
bH  = [bin1(v) for v in net[0].bias.detach().round().tolist()]              # 2 ints in {0,1}
bO  = 1                                                                      # avoid negatives

preds   = [xor_forward(*xy, wIH, bH, bO) for xy in [(0,0),(1,0),(0,1),(1,1)]]
correct = sum(int(p==t) for p,t in zip(preds, [0,1,1,0]))
acc_bps = correct * 2500
print(f"   circuit-compatible acc = {correct}/4 → {acc_bps/100}%")

# 4) witness & proof
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

submit = orch.functions.submitProof(REQ_ID, a, b, c, public_signals).build_transaction({
    "from": addr, "gas": 600_000, "gasPrice": w3.eth.gas_price,
    "nonce": w3.eth.get_transaction_count(addr),
})
rcpt = send_tx(submit)
print(f"   ✓ proof submitted in tx {rcpt['transactionHash'].hex()}")

# 6) wait for auto-settlement & print rewards
while True:
    closed, bestAcc, winnersCount, perWinnerWei = orch.functions.getResult(REQ_ID).call()
    if closed:
        winners = [orch.functions.winnerAt(REQ_ID, i).call() for i in range(winnersCount)]
        print(f"[worker:{addr}] settled! bestAcc={bestAcc/100:.2f}% winners={winnersCount} perWinner={w3.from_wei(perWinnerWei,'ether')} ETH")
        if addr in winners:
            print(f"   🎉 You are a WINNER. Payout: {w3.from_wei(perWinnerWei,'ether')} ETH")
        else:
            print(f"   You did not win this round. Payout: {w3.from_wei(perWinnerWei,'ether')} ETH")
        break
    print("   waiting for others to submit & settle…")
    time.sleep(3)
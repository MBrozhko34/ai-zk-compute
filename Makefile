-include .env

# ──────────────────────────────────────────────────────────────────────────────
# General / deps
# ──────────────────────────────────────────────────────────────────────────────
.PHONY: setup deps clear nde depl req exp worker zk-ptau zk-circuit zk-setup zk-verifier zk-all

setup: deps

deps:
	pnpm install
	cd node && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

clear:
	pnpm hardhat clean
	rm -rf circuits/XorCircuit_js circuits/*.zkey circuits/*.r1cs circuits/*.sym \
	       contracts/Groth16Verifier.sol contracts/PlonkVerifier.sol \
	       zk/* contracts/AccVerifier.sol

clean:
	# Hardhat artifacts/cache
	-pnpm hardhat clean

	# --- Circom build outputs (safe: does NOT touch *.circom or *.ptau) ---
	# wasm dirs created by circom (e.g., XorCircuit_js, MlpHoldoutAcc_256_js)
	-rm -rf circuits/*_js

	# top-level files emitted by circom/snarkjs:
	#   *.r1cs, *.sym, *.zkey, *.wtns, *.wasm, verification_key.json
	# (keeps pot*.ptau and any *.circom)
	-find circuits -maxdepth 1 -type f \( \
		-name "*.r1cs" -o -name "*.sym"  -o -name "*.zkey" -o \
		-name "*.wtns" -o -name "*.wasm" -o -name "verification_key.json" \
	\) -delete

	# If snarkjs left any logs/tmp (harmless if none exist)
	-find circuits -maxdepth 1 -type f \( -name "*.tmp" -o -name "*.log" \) -delete

	# --- Solidity verifiers produced by the builds ---
	-rm -f contracts/Groth16Verifier.sol \
	      contracts/PlonkVerifier.sol \
	      contracts/AccVerifier.sol

	# Optional: wipe local zk scratch dir if you keep artifacts there
	-rm -rf zk/*

# ──────────────────────────────────────────────────────────────────────────────
# (legacy XOR helpers kept for reference; not used by worker anymore)
# ──────────────────────────────────────────────────────────────────────────────
groth-circuit:
	cd circuits && \
		circom XorCircuit.circom --r1cs --wasm --sym -l ./lib && \
		snarkjs groth16 setup XorCircuit.r1cs pot18_final.ptau xor_000.zkey && \
		snarkjs zkey contribute xor_000.zkey xor_final.zkey --name="initial setup" -e="random entropy" && \
		snarkjs zkey export verificationkey xor_final.zkey verification_key.json && \
		snarkjs zkey export solidityverifier xor_final.zkey ../contracts/Groth16Verifier.sol

# ──────────────────────────────────────────────────────────────────────────────
# Hardhat / deploy / request
# ──────────────────────────────────────────────────────────────────────────────
nde:
	pnpm hardhat compile
	npx hardhat node

depl:
	pnpm hardhat run scripts/deploy.ts --network localhost

req:
	ts-node client/open_request.ts

exp:
	@echo 'export RPC_URL=http://127.0.0.1:8545'
	@echo 'export ORCH_ADDR=<DEPLOYED_ORCHESTRATOR_ADDRESS>'
	@echo 'export REQUEST_ID=0'
	@echo 'export PRIVATE_KEY=<ONE_OF_HARDHAT_KEYS>'

# ──────────────────────────────────────────────────────────────────────────────
# ZK (Groth16) for the MLP hold-out accuracy circuit
#   circuits/MlpHoldoutAcc_256.circom  →  zk/MlpHoldoutAcc_256_js/MlpHoldoutAcc_256.wasm
# ──────────────────────────────────────────────────────────────────────────────
CIRCOM     := npx circom
SNARKJS    := npx snarkjs

# --- zk (Groth16) for MLP hold-out accuracy ---
# ── ZK (Groth16) artifacts for the MLP holdout circuit ───────────────────────
# --- zk (Groth16) for MLP hold-out accuracy ---
# Circuit basename (no spaces!)
ACC_NAME := $(strip MlpHoldoutAcc_256)
ACC_DIR  := circuits
ACC_WASM := $(ACC_DIR)/$(ACC_NAME)_js/$(ACC_NAME).wasm
ACC_ZKEY := $(ACC_DIR)/$(ACC_NAME)_final.zkey
ACC_CIR  := circuits/$(ACC_NAME).circom
ACC_R1CS := $(ACC_DIR)/$(ACC_NAME).r1cs
PTAU     := $(ACC_DIR)/pot18_final.ptau
ACC_VER  := $(ACC_DIR)/AccVerifier.sol

zk-ptau:
	mkdir -p $(ACC_DIR)
	snarkjs powersoftau new bn128 15 $(PTAU).tmp -v && \
	snarkjs powersoftau contribute $(PTAU).tmp $(PTAU) --name="init" -v && \
	rm -f $(PTAU).tmp


mlp-zk:
	circom circuits/MlpHoldoutAcc_256.circom --r1cs --wasm --sym -o circuits -l node_modules -l circuits/lib && \
	cd circuits && \
		snarkjs groth16 setup MlpHoldoutAcc_256.r1cs pot18_final.ptau MlpHoldoutAcc_256_pk.zkey && \
		snarkjs zkey contribute MlpHoldoutAcc_256_pk.zkey MlpHoldoutAcc_256_final.zkey --name="acc" -e="random entropy" -v && \
		snarkjs zkey export verificationkey MlpHoldoutAcc_256_final.zkey verification_key.json && \
		snarkjs zkey export solidityverifier MlpHoldoutAcc_256_final.zkey ../contracts/AccVerifier.sol

# ──────────────────────────────────────────────────────────────────────────────
# Worker – ensures zk artifacts exist (MLP), then runs the Python node
# ──────────────────────────────────────────────────────────────────────────────
worker:
	@[ -n "$(RPC_URL)" ] || (echo "RPC_URL missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(ORCH_ADDR)" ] || (echo "ORCH_ADDR missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(REQUEST_ID)" ] || (echo "REQUEST_ID missing (pass e.g. REQUEST_ID=0)"; exit 1)
	@[ -n "$(PRIVATE_KEY)" ] || (echo "Usage: make worker PRIVATE_KEY=<hex> [REQUEST_ID=N]"; exit 1)
	@[ -x "node/.venv/bin/python" ] || (echo "Python venv missing. Run 'make deps' first."; exit 1)
	@[ -f "$(ACC_ZKEY)" ] && [ -f "$(ACC_WASM)" ] || $(MAKE) mlp-zk
	@[ -f "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json" ] || pnpm hardhat compile
	cd node && \
		RPC_URL="$(RPC_URL)" ORCH_ADDR="$(ORCH_ADDR)" REQUEST_ID="$(REQUEST_ID)" PRIVATE_KEY="$(PRIVATE_KEY)" \
		ACC_WASM="../$(ACC_WASM)" ACC_ZKEY="../$(ACC_ZKEY)" \
		ZK_ACC_WASM="../$(ACC_WASM)" ZK_ACC_ZKEY="../$(ACC_ZKEY)" \
		.venv/bin/python compute_node.py


startup: clear clean mlp-zk nde
	@echo "✅ startup complete: ran 'clear' → 'clean' → 'nde'"

client-side: depl req

tidy: clear clean
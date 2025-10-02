-include .env

# ──────────────────────────────────────────────────────────────────────────────
# General / deps
# ──────────────────────────────────────────────────────────────────────────────
.PHONY: deps clear clean nde depl req exp worker zk-ptau mlp-zk

deps:
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

	-rm -rf node/tmp/proof-0-*


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
	@echo "✅ startup complete: ran 'clear' → 'clean' → 'mlp-zk' → 'nde'"

run: depl req
	$(MAKE) REQ_ID=0 workers-12

tidy: clear clean

# --------------------------------------------------------------------
# QoL: allow REQ_ID alias (compat with earlier commands)
# --------------------------------------------------------------------
REQUEST_ID ?= $(REQ_ID)

# --------------------------------------------------------------------
# Local Hardhat dev private keys (NEVER use on real networks)
# We start workers from #2 to avoid colliding with the deployer/client.
# --------------------------------------------------------------------
PK2  := 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
PK3  := 0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a
PK4  := 0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6
PK5  := 0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a
PK6  := 0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba
PK7  := 0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e
PK8  := 0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356
PK9  := 0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97
PK10 := 0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6
PK11 := 0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897
PK12 := 0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82
PK13 := 0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd

# helper to error early if REQUEST_ID missing
define _require_rid
	@[ -n "$(REQUEST_ID)" ] || { echo "REQUEST_ID (or REQ_ID) is required"; exit 1; }
endef

.PHONY: workers-3 workers-6 workers-9 workers-12 stop-workers

workers-1:
	@echo "▶ Spawning 3 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK2) & \
	wait; echo "workers-1 finished"

workers-1-2:
	@echo "▶ Spawning 3 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK3) & \
	wait; echo "workers-1 finished"

# Spawn 3 workers (accounts #2..#4)
workers-3:
	$(call _require_rid)
	@echo "▶ Spawning 3 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK2) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK3) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK4) & \
	wait; echo "✅ workers-3 finished"

# Spawn 6 workers (accounts #2..#7)
workers-6:
	$(call _require_rid)
	@echo "▶ Spawning 6 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK2) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK3) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK4) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK5) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK6) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK7) & \
	wait; echo "✅ workers-6 finished"

# Spawn 9 workers (accounts #2..#10)
workers-9:
	$(call _require_rid)
	@echo "▶ Spawning 9 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK2)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK3)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK4)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK5)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK6)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK7)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK8)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK9)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK10) & \
	wait; echo "✅ workers-9 finished"

# Spawn 12 workers (accounts #2..#13)
workers-12:
	$(call _require_rid)
	@echo "▶ Spawning 12 workers for REQUEST_ID=$(REQUEST_ID)"; \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK2)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK3)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK4)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK5)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK6)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK7)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK8)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK9)  & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK10) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK11) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK12) & \
	$(MAKE) --no-print-directory worker REQUEST_ID=$(REQUEST_ID) PRIVATE_KEY=$(PK13) & \
	wait; echo "✅ workers-12 finished"

# Kill any backgrounded Python workers (best-effort)
stop-workers:
	-@pkill -f "compute_node\.py" || true

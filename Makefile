-include .env
setup: deps circuit

deps:
	pnpm install
	cd node && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

clear:
	pnpm hardhat clean
	rm -rf circuits/XorCircuit_js circuits/*.zkey circuits/*.r1cs circuits/*.sym contracts/Groth16Verifier.sol contracts/PlonkVerifier.sol

plonk-circuit:
	cd circuits && \
		circom XorCircuit.circom --r1cs --wasm --sym -l ./lib && \
		snarkjs plonk setup XorCircuit.r1cs pot18_final.ptau xor_final.zkey && \
		snarkjs zkey export verificationkey xor_final.zkey verification_key.json && \
		snarkjs zkey export solidityverifier xor_final.zkey ../contracts/PlonkVerifier.sol

groth-circuit:
	cd circuits && \
		circom XorCircuit.circom --r1cs --wasm --sym -l ./lib && \
		snarkjs groth16 setup XorCircuit.r1cs pot18_final.ptau xor_000.zkey && \
		snarkjs zkey contribute xor_000.zkey xor_final.zkey --name="initial setup" -e="random entropy" && \
		snarkjs zkey export verificationkey xor_final.zkey verification_key.json && \
		snarkjs zkey export solidityverifier xor_final.zkey ../contracts/Groth16Verifier.sol

nde:
	pnpm hardhat compile
	npx hardhat node

depl:
	pnpm hardhat run scripts/deploy.ts --network localhost

req:
	ts-node client/open_request.ts

exp:
	export RPC_URL=http://127.0.0.1:8545
	export ORCH_ADDR=0x261D8c5e9742e6f7f1076Fa1F560894524e19cad
	export REQUEST_ID=0
	export PRIVATE_KEY=0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d

.PHONY: worker
worker:
	@[ -n "$(RPC_URL)" ] || (echo "RPC_URL missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(ORCH_ADDR)" ] || (echo "ORCH_ADDR missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(REQUEST_ID)" ] || (echo "REQUEST_ID missing (pass e.g. REQUEST_ID=0)"; exit 1)
	@[ -n "$(PRIVATE_KEY)" ] || (echo "Usage: make worker PRIVATE_KEY=<hex> [REQUEST_ID=N]"; exit 1)
	@[ -x "node/.venv/bin/python" ] || (echo "Python venv missing. Run 'make deps' first."; exit 1)
	@[ -f "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json" ] || pnpm hardhat compile
	@([ -f "circuits/xor_final.zkey" ] && [ -f "circuits/XorCircuit_js/XorCircuit.wasm" ]) || $(MAKE) circuit
	cd node && \
		RPC_URL="$(RPC_URL)" ORCH_ADDR="$(ORCH_ADDR)" REQUEST_ID="$(REQUEST_ID)" PRIVATE_KEY="$(PRIVATE_KEY)" \
		.venv/bin/python compute_node.py

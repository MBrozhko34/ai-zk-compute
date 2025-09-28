-include .env
setup: deps circuit

deps:
	pnpm install
	cd node && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

clear:
	pnpm hardhat clean
	rm -rf circuits/XorCircuit_js circuits/*.zkey circuits/*.r1cs circuits/*.sym contracts/Groth16Verifier.sol contracts/PlonkVerifier.sol

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

.PHONY: workerp
workerp:
	@[ -n "$(RPC_URL)" ] || (echo "RPC_URL missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(ORCH_ADDR)" ] || (echo "ORCH_ADDR missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(REQUEST_ID)" ] || (echo "REQUEST_ID missing (pass e.g. REQUEST_ID=0)"; exit 1)
	@[ -n "$(PRIVATE_KEY)" ] || (echo "Usage: make worker PRIVATE_KEY=<hex> [REQUEST_ID=N]"; exit 1)
	@[ -x "node/.venv/bin/python" ] || (echo "Python venv missing. Run 'make deps' first."; exit 1)
	@[ -f "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json" ] || pnpm hardhat compile
	cd node && \
		RPC_URL="$(RPC_URL)" ORCH_ADDR="$(ORCH_ADDR)" REQUEST_ID="$(REQUEST_ID)" PRIVATE_KEY="$(PRIVATE_KEY)" \
		.venv/bin/python compute_node.py


.PHONY: spawn
spawn:
	@[ -n "$(RPC_URL)" ] || (echo "RPC_URL missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(ORCH_ADDR)" ] || (echo "ORCH_ADDR missing (set in .env or pass inline)"; exit 1)
	@[ -n "$(REQUEST_ID)" ] || (echo "REQUEST_ID missing (pass e.g. REQUEST_ID=0)"; exit 1)
	@[ -x "node/.venv/bin/python" ] || (echo "Python venv missing. Run 'make deps' first."; exit 1)
	@[ -f "artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json" ] || pnpm hardhat compile
	@keys="\
0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d \
0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a \
0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6 \
0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a \
0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba \
0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e \
0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356 \
0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97 \
0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6 \
0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897 \
0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82 \
0xa267530f49f8280200edf313ee7af6b827f2a8bce2897751d06a843f644967b1 \
0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd \
0xc526ee95bf44d8fc405a158bb884d9d1238d99f0612e9f33d006bb0789009aaa \
0x8166f546bab6da521a8369cab06c5d2b9e46670292d85c875ee9ec20e84ffb61 \
0xea6c44ac03bff858b476bba40716402b03e41b8e97e276d1baec7c37d42484a0 \
0x689af8efa8c651a91ad287602527f3af2fe9f6501a7ac4b061667b5a93e037fd \
0xde9be858da4a475276426320d5e9262ecfc3ba460bfac56360bfa6c4c28b4ee0 \
0xdf57089febbacf7ba0bc227dafbffa9fc08a93fdc68e1e42411a14efcf23656e"; \
	for k in $$keys; do \
		echo "→ starting worker for $$k"; \
		$(MAKE) -s workerp PRIVATE_KEY=$$k REQUEST_ID=$(REQUEST_ID) & \
	done; \
	wait

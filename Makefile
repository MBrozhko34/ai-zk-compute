-include .env
setup: deps circuit

deps:
	pnpm install
	cd node && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

clear:
	pnpm hardhat clean
	rm -rf circuits/XorCircuit_js circuits/*.zkey circuits/*.r1cs circuits/*.sym contracts/Groth16Verifier.sol

circuit:
	cd circuits && \
		circom XorCircuit.circom --r1cs --wasm --sym -l ./lib && \
		snarkjs groth16 setup XorCircuit.r1cs powersOfTau28_hez_final_10.ptau xor_000.zkey && \
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







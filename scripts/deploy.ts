import hre from "hardhat";
import { ethers } from "ethers";
import * as fs from "fs";
import * as path from "path";
import * as dotenv from "dotenv";
dotenv.config();

/**
 * Resolve the Groth16 verifier artifact regardless of the contract name
 * that snarkjs emitted (e.g., "Groth16Verifier" vs "Verifier").
 */
async function getVerifierFactory(signer: any) {
  // Try the most common fully-qualified & bare names first.
  const candidates = [
    "contracts/AccVerifier.sol:Groth16Verifier",
    "contracts/AccVerifier.sol:Verifier",
    "Groth16Verifier",
    "Verifier",
    "AccVerifier",
  ];

  for (const name of candidates) {
    try {
      const art = await hre.artifacts.readArtifact(name);
      console.log(`Using verifier artifact: ${art.contractName} (source: ${name})`);
      return hre.ethers.getContractFactoryFromArtifact(art, signer);
    } catch {
      /* keep trying */
    }
  }

  // Fallback: scan artifacts for AccVerifier.sol and pick the one that has verifyProof
  const dir = path.join(__dirname, "..", "artifacts", "contracts", "AccVerifier.sol");
  if (fs.existsSync(dir)) {
    const files = fs.readdirSync(dir).filter(f => f.endsWith(".json"));
    for (const f of files) {
      try {
        const p = path.join(dir, f);
        const art = JSON.parse(fs.readFileSync(p, "utf8"));
        if (art?.abi && art?.bytecode) {
          const hasVerify = (art.abi as any[]).some(
            (x) =>
              x?.type === "function" &&
              x?.name === "verifyProof" &&
              Array.isArray(x?.inputs) &&
              x.inputs.length === 4
          );
          if (hasVerify) {
            console.log(`Using verifier artifact (scan): ${art.contractName} (${f})`);
            return hre.ethers.getContractFactoryFromArtifact(art, signer);
          }
        }
      } catch {
        /* ignore bad json */
      }
    }
  }

  throw new Error(
    "Verifier artifact not found. Run `make mlp-zk` (to regenerate contracts/AccVerifier.sol) and then `npx hardhat compile`."
  );
}

async function main() {
  /* 0. env */
  const { RPC_URL, PRIVATE_KEY } = process.env;
  if (!RPC_URL || !PRIVATE_KEY) throw new Error("RPC_URL & PRIVATE_KEY in .env");

  // Ensure artifacts exist (especially after regenerating AccVerifier.sol)
  await hre.run("compile");

  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PRIVATE_KEY, provider);
  console.log("Deployer:", wallet.address);

  /* 1. current mined nonce */
  const baseNonce = await wallet.getNonce();

  /* 2. Groth16 verifier (name-agnostic) */
  const Verifier = await getVerifierFactory(wallet);
  const verifier = await Verifier.deploy({ nonce: baseNonce });
  await verifier.waitForDeployment();
  const verifierAddr = await verifier.getAddress();
  console.log("Verifier →", verifierAddr);

  /* 3. AiOrchestrator (nonce = baseNonce + 1) */
  const Orch = await hre.ethers.getContractFactory("AiOrchestrator", wallet);
  const orch = await Orch.deploy(verifierAddr, { nonce: baseNonce + 1 });
  await orch.waitForDeployment();
  const orchAddr = await orch.getAddress();
  console.log("Orchestrator →", orchAddr);

  /* 4. reminder */
  console.log(`\nAdd to .env:\nORCH_ADDR=${orchAddr}\n`);
}

main().catch((e) => {
  console.error(e?.shortMessage || e);
  process.exitCode = 1;
});

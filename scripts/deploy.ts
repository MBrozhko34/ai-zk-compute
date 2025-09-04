// scripts/deploy.ts
import hre from "hardhat";
import { ethers } from "ethers";
import * as dotenv from "dotenv";
dotenv.config();

async function main() {
  /* 0. env */
  const { RPC_URL, PRIVATE_KEY } = process.env;
  if (!RPC_URL || !PRIVATE_KEY) throw new Error("RPC_URL & PRIVATE_KEY in .env");

  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PRIVATE_KEY, provider);
  console.log("Deployer:", wallet.address);

  /* 1. current mined nonce */
  const baseNonce = await wallet.getNonce();

  /* 2. PlonkVerifier */
  const Verifier = await hre.ethers.getContractFactory("PlonkVerifier", wallet);
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
  console.error(e);
  process.exitCode = 1;
});

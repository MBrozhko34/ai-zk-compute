// scripts/deploy.ts
import hre from "hardhat";
import { ethers as Ethers } from "ethers";
import * as dotenv from "dotenv";
dotenv.config();

async function main() {
  const { RPC_URL, PRIVATE_KEY } = process.env;
  if (!RPC_URL || !PRIVATE_KEY) {
    throw new Error("RPC_URL & PRIVATE_KEY must be set in .env");
  }

  // Signer set up from your .env (so it's the same address your worker will use)
  const provider = new Ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new Ethers.Wallet(PRIVATE_KEY, provider);
  console.log("Deployer:", wallet.address);

  // Optional: keep a stable nonce, but it's fine to omit and let ethers set it.
  const nextNonce = await wallet.getNonce();

  // IMPORTANT: AiOrchestrator has a zero-arg constructor in the no-Groth16 path.
  const Orch = await hre.ethers.getContractFactory("AiOrchestrator", wallet);
  const orch = await Orch.deploy({ nonce: nextNonce }); // <-- pass ONLY overrides
  await orch.waitForDeployment();
  const orchAddr = await orch.getAddress();

  console.log("AiOrchestrator →", orchAddr);
  console.log(`\nAdd to .env:\nORCH_ADDR=${orchAddr}\n`);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

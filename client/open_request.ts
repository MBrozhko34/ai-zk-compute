import { ethers } from "ethers";
import * as dotenv from "dotenv";
import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

dotenv.config();

if (!process.env.ORCH_ADDR || !process.env.ORCH_ADDR.startsWith("0x")) {
  throw new Error(`ORCH_ADDR env var missing or malformed: "${process.env.ORCH_ADDR}"`);
}

(async () => {
  /* ── provider + signer ─────────────────────────────── */
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL!);
  const wallet   = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);
  const orch     = new ethers.Contract(process.env.ORCH_ADDR!, OrchAbi.abi, wallet);

  /* ── hyper‑parameter grid ──────────────────────────── */
  const grid = [
    { lr:  50_000, steps: 900 },
    { lr: 100_000, steps: 700 },
    { lr: 150_000, steps: 500 }
  ];

  /* each proof earns 0.01 ETH; deposit total bounty up‑front */
  const rewardWei = ethers.parseEther("0.01");
  const bounty    = rewardWei * BigInt(grid.length);

  /* ── open the request (new ABI: no reward argument) ── */
  const tx = await orch.openRequest(
    "ipfs://Qm...datasetCID",  // dataset location
    grid,                       // full hp grid
    { value: bounty }           // ETH bounty
  );
  const rc = await tx.wait();
  console.log("request id", rc!.logs[0].args!.id.toString());
})();
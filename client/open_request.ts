import { ethers } from "ethers";
import * as dotenv from "dotenv";
import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

dotenv.config();

if (!process.env.ORCH_ADDR || !process.env.ORCH_ADDR.startsWith("0x")) {
  throw new Error(`ORCH_ADDR env var missing or malformed: "${process.env.ORCH_ADDR}"`);
}

(async () => {
  /* Provider + signer */
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL!);
  const wallet   = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);
  const orch     = new ethers.Contract(process.env.ORCH_ADDR!, OrchAbi.abi, wallet);

  /* Expanded hyper-parameter grid (lr is ppm) */
  const grid = [
    { lr:   1_000, steps:  30 },  // 0.001 × 30
    { lr:   5_000, steps:  60 },  // 0.005 × 60
    { lr:  20_000, steps:  40 },  // 0.02  × 40
    { lr:  50_000, steps:  80 },  // 0.05  × 80
    { lr:  80_000, steps: 120 },  // 0.08  × 120
    { lr: 150_000, steps:  20 },  // 0.15  × 20
    { lr: 150_000, steps: 300 },  // 0.15  × 300
    { lr:     500, steps:   5 },  // 0.0005 × 5
  ];

  /* Bounty scales with grid size */
  const perTask = ethers.parseEther("0.01");
  const bounty  = perTask * BigInt(grid.length);

  const tx = await orch.openRequest("ipfs://...datasetCID", grid, /* minWorkers */ 2, { value: bounty });

  const rc = await tx.wait();
  const log = rc!.logs[0] as any;
  console.log("request id", log.args!.id.toString());
})();

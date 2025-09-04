import { ethers } from "ethers";
import * as dotenv from "dotenv";
import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

dotenv.config();

const ORCH_ADDR = process.env.ORCH_ADDR;
const RPC_URL   = process.env.RPC_URL;
const PK        = process.env.PRIVATE_KEY;

if (!ORCH_ADDR || !ORCH_ADDR.startsWith("0x")) {
  throw new Error(`ORCH_ADDR env var missing or malformed: "${ORCH_ADDR}"`);
}
if (!RPC_URL) throw new Error("RPC_URL missing");
if (!PK)      throw new Error("PRIVATE_KEY missing");

(async () => {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PK, provider);
  const orch     = new ethers.Contract(ORCH_ADDR, OrchAbi.abi, wallet);

  // Struct[] matches: struct HyperParam { uint256 lr; uint256 steps; }
  const grid: Array<{ lr: bigint | number; steps: bigint | number }> = [
    { lr:   1_000, steps:  30 },
    { lr:   5_000, steps:  60 },
    { lr:  20_000, steps:  40 },
    { lr:  50_000, steps:  80 },
    { lr:  80_000, steps: 120 },
    { lr: 150_000, steps:  20 },
    { lr: 150_000, steps: 300 },
    { lr:     500, steps:   5 },
  ];

  const perTask = ethers.parseEther("0.01");
  const bounty  = perTask * BigInt(grid.length);
  const minWorkers = 2;

  // ✅ 3 args + overrides (matches your Groth16-only contract)
  const tx = await orch.openRequest("ipfs://...datasetCID", grid, minWorkers, {
    value: bounty,
  });
  const receipt = await tx.wait();

  // Parse our event safely (don’t assume logs[0])
  let openedId: bigint | null = null;
  for (const log of receipt.logs) {
    if (log.address.toLowerCase() !== (orch.target as string).toLowerCase()) continue;
    try {
      const parsed = orch.interface.parseLog(log);
      if (parsed?.name === "RequestOpened") {
        const a: any = parsed.args;
        openedId = a.id as bigint;
        console.log("request id", openedId.toString());
        console.log("minWorkers", a.minWorkers?.toString?.() ?? minWorkers.toString());
        console.log("taskCount", a.taskCount?.toString?.() ?? grid.length.toString());
        console.log("bountyWei", a.bountyWei?.toString?.() ?? bounty.toString());
        break;
      }
    } catch {}
  }

  // Fallback if event not found (rare in HH local)
  if (openedId === null) {
    const nextId: bigint = await orch.nextId();
    openedId = nextId - 1n;
    console.log("(fallback) request id", openedId.toString());
    console.log("minWorkers", minWorkers.toString());
    console.log("taskCount", grid.length.toString());
    console.log("bountyWei", bounty.toString());
  }
})();

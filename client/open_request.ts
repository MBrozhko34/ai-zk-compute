// client/open_request.ts
import { ethers, NonceManager } from "ethers";
import * as dotenv from "dotenv";
import * as fs from "fs";
import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

dotenv.config();

const ORCH_ADDR = process.env.ORCH_ADDR;
const RPC_URL   = process.env.RPC_URL;
const PK        = process.env.PRIVATE_KEY;
const CSV_PATH  = process.env.DATASET_CSV || "client/dataset.csv";

if (!ORCH_ADDR || !ORCH_ADDR.startsWith("0x")) {
  throw new Error(`ORCH_ADDR env var missing or malformed: "${ORCH_ADDR}"`);
}
if (!RPC_URL) throw new Error("RPC_URL missing");
if (!PK)      throw new Error("PRIVATE_KEY missing");

function loadCsv3(path: string): { x0: bigint[]; x1: bigint[]; y: bigint[] } {
  const txt = fs.readFileSync(path, "utf8");
  const x0: bigint[] = [], x1: bigint[] = [], y: bigint[] = [];
  const isInt = (s: string) => /^-?\d+$/.test(s.trim());

  const lines = txt.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw) continue;
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cells = line.split(/[,;\s]+/).filter(Boolean);
    if (cells.length < 3) continue;

    if (!isInt(cells[0]) || !isInt(cells[1]) || !isInt(cells[2])) {
      if (i === 0) continue; // header
      console.warn(`Skipping non-numeric row ${i + 1}: ${line}`);
      continue;
    }

    const a = BigInt(cells[0]);
    const b = BigInt(cells[1]);
    const c = BigInt(cells[2]);
    if ((a !== 0n && a !== 1n) || (b !== 0n && b !== 1n) || (c !== 0n && c !== 1n)) {
      throw new Error(`Row ${i + 1}: values must be 0/1, got (${cells[0]}, ${cells[1]}, ${cells[2]})`);
    }

    x0.push(a); x1.push(b); y.push(c);
  }

  if (x0.length === 0) throw new Error(`Empty/invalid CSV at ${path}`);
  if (x0.length !== x1.length || x0.length !== y.length) {
    throw new Error(`CSV lengths mismatch: x0=${x0.length} x1=${x1.length} y=${y.length}`);
  }
  return { x0, x1, y };
}

(async () => {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PK, provider);
  const managed  = new NonceManager(wallet);                 // <-- NEW
  const orch     = new ethers.Contract(ORCH_ADDR, OrchAbi.abi, managed); // <-- use managed

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

  const tx = await orch.openRequest("file://dataset.csv", grid, minWorkers, { value: bounty });
  const receipt = await tx.wait();

  let openedId: bigint | null = null;
  for (const log of receipt.logs) {
    if (log.address.toLowerCase() !== (orch.target as string).toLowerCase()) continue;
    try {
      const parsed = orch.interface.parseLog(log);
      if (parsed?.name === "RequestOpened") { openedId = parsed.args.id as bigint; break; }
    } catch {}
  }
  if (openedId === null) {
    const nextId: bigint = await orch.nextId();
    openedId = nextId - 1n;
  }
  console.log("request id", openedId.toString());

  const { x0, x1, y } = loadCsv3(CSV_PATH);
  console.log(`pushing HOLD-OUT dataset of ${x0.length} rows from ${CSV_PATH}…`);
  await (await orch.setHoldoutDataset(openedId, x0, x1, y)).wait();
  console.log("✓ hold-out dataset set");
})().catch((e) => {
  // small QoL: show ethers JSON-RPC error message cleanly
  console.error(e?.info?.error?.message || e?.shortMessage || e);
  process.exitCode = 1;
});

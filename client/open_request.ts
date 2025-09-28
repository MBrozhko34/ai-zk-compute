// client/open_request.ts
import { ethers, NonceManager } from "ethers";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";
import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

dotenv.config();

const ORCH_ADDR   = process.env.ORCH_ADDR!;
const RPC_URL     = process.env.RPC_URL!;
const PK          = process.env.PRIVATE_KEY!;
const TRAIN_CSV   = process.env.TRAIN_CSV   || "client/train.csv";
const HOLDOUT_CSV = process.env.HOLDOUT_CSV || "client/dataset2.csv";  // your bigger dataset
const HOLDOUT_MODE = (process.env.HOLDOUT_MODE || "array").toLowerCase(); // "array" | "root"

if (!ORCH_ADDR || !ORCH_ADDR.startsWith("0x")) throw new Error(`ORCH_ADDR missing/malformed: ${ORCH_ADDR}`);
if (!RPC_URL) throw new Error("RPC_URL missing");
if (!PK)      throw new Error("PRIVATE_KEY missing");

// ----- CSV → quantized arrays (x0,x1 in 0..15; y in {0,1})
function quant01ToQ(s: string): bigint {
  const v = Number.parseFloat(s);
  if (!Number.isFinite(v)) throw new Error(`bad float "${s}"`);
  const clamped = Math.min(1, Math.max(0, v));
  const q = Math.round(clamped * 15);
  return BigInt(q);
}
function isHeaderRow(cells: string[]): boolean {
  // treat row as header if any of the first 3 cells is not a number
  return cells.slice(0, 3).some((c) => Number.isNaN(Number.parseFloat(c)));
}
function loadCsvQuant(pathStr: string): { x0: bigint[]; x1: bigint[]; y: bigint[] } {
  const txt = fs.readFileSync(pathStr, "utf8");
  const x0: bigint[] = [], x1: bigint[] = [], y: bigint[] = [];
  const lines = txt.split(/\r?\n/);
  let seenHeader = false;
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]; if (!raw) continue;
    const line = raw.trim(); if (!line || line.startsWith("#")) continue;
    const cells = line.split(/[,;\s]+/).filter(Boolean);
    if (cells.length < 3) continue;

    if (!seenHeader && isHeaderRow(cells)) { seenHeader = true; continue; }

    // features as floats 0..1 → quantize to 0..15
    try {
      const a = quant01ToQ(cells[0]);
      const b = quant01ToQ(cells[1]);
      const c = BigInt(parseInt(cells[2], 10));
      if (c !== 0n && c !== 1n) throw new Error(`y must be 0/1, got ${cells[2]}`);
      x0.push(a); x1.push(b); y.push(c);
    } catch (e) {
      throw new Error(`Row ${i + 1}: ${String(e)}`);
    }
  }
  if (x0.length === 0) throw new Error(`Empty/invalid CSV at ${pathStr}`);
  if (x0.length !== x1.length || x0.length !== y.length) {
    throw new Error(`CSV lengths mismatch: x0=${x0.length} x1=${x1.length} y=${y.length}`);
  }
  return { x0, x1, y };
}

// ----- Merkle (sorted-pair) exactly like contract: keccak( sorted(leafA, leafB) )
function leafForSample(idx: number, x0: bigint, x1: bigint, y: bigint): string {
  return ethers.solidityPackedKeccak256(
    ["uint256","uint256","uint256","uint256"],
    [BigInt(idx), x0, x1, y]
  );
}
function hashPairSorted(aHex: string, bHex: string): string {
  const a = ethers.getBytes(aHex);
  const b = ethers.getBytes(bHex);
  const [lo, hi] = (Buffer.compare(Buffer.from(a), Buffer.from(b)) <= 0) ? [a, b] : [b, a];
  return ethers.keccak256(ethers.concat([lo, hi]));
}
function merkleRootSorted(leaves: string[]): string {
  if (leaves.length === 0) return ethers.ZeroHash;
  let cur = leaves.slice();
  while (cur.length > 1) {
    const nxt: string[] = [];
    for (let i = 0; i < cur.length; i += 2) {
      const a = cur[i];
      const b = (i + 1 < cur.length) ? cur[i + 1] : cur[i];
      nxt.push(hashPairSorted(a, b));
    }
    cur = nxt;
  }
  return cur[0];
}

(async () => {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PK, provider);
  const managed  = new NonceManager(wallet);
  const orch     = new ethers.Contract(ORCH_ADDR, OrchAbi.abi, managed);

  // ---- Bigger hyper-parameter grid (edit as you like)
  const grid: Array<{ lr: bigint | number; steps: bigint | number }> = [
    // lr in ppm (will bucket on-chain), moderate step counts
    { lr:  5_000,  steps:  50 }, { lr:  5_000,  steps: 150 }, { lr:  5_000,  steps: 300 },
    { lr: 20_000,  steps:  50 }, { lr: 20_000,  steps: 150 }, { lr: 20_000,  steps: 300 },
    { lr: 40_000,  steps:  50 }, { lr: 40_000,  steps: 150 }, { lr: 40_000,  steps: 300 },
    { lr: 80_000,  steps:  50 }, { lr: 80_000,  steps: 150 }, { lr: 80_000,  steps: 300 },
    { lr:100_000,  steps:  50 }, { lr:100_000,  steps: 150 }, { lr:100_000,  steps: 300 },
    { lr:150_000,  steps:  50 }, { lr:150_000,  steps: 150 }, { lr:150_000,  steps: 300 },
  ];

  // One request funds all tasks equally
  const perTask = ethers.parseEther("0.01");
  const bounty  = perTask * BigInt(grid.length);
  const minWorkers = 2;

  const datasetCid = "file://" + path.basename(HOLDOUT_CSV);
  const tx = await orch.openRequest(datasetCid, grid, minWorkers, { value: bounty });
  const rc = await tx.wait();

  // derive id from event (fallback to nextId-1)
  let id: bigint | null = null;
  for (const log of rc.logs) {
    if (log.address.toLowerCase() !== (orch.target as string).toLowerCase()) continue;
    try {
      const parsed = orch.interface.parseLog(log);
      if (parsed?.name === "RequestOpened") { id = parsed.args.id as bigint; break; }
    } catch {}
  }
  if (id === null) {
    const nextId: bigint = await orch.nextId();
    id = nextId - 1n;
  }
  console.log("request id", id.toString());

  // ---- Training commitment (R0)
  const tr = loadCsvQuant(TRAIN_CSV);
  const trLeaves = tr.x0.map((_, i) => leafForSample(i, tr.x0[i], tr.x1[i], tr.y[i]));
  const trRoot = merkleRootSorted(trLeaves);
  console.log(`training set: len=${tr.x0.length} root=${trRoot}`);
  await (await orch.setTrainingDatasetRoot(id, trRoot, tr.x0.length)).wait();
  console.log("✓ training root set");

  // ---- Hold-out (array mode by default; or Merkle root if HOLDOUT_MODE=root)
  const te = loadCsvQuant(HOLDOUT_CSV);
  console.log(`hold-out set has ${te.x0.length} rows (${HOLDOUT_MODE} mode)`);

  const teLeaves = te.x0.map((_, i) => leafForSample(i, te.x0[i], te.x1[i], te.y[i]));
  const teRoot = merkleRootSorted(teLeaves);
  await (await orch.setHoldoutDatasetRoot(id, teRoot, te.x0.length)).wait();
  console.log("✓ hold-out root set");
})().catch((e) => {
  console.error(e?.info?.error?.message || e?.shortMessage || e);
  process.exitCode = 1;
});

// client/open_request.ts
import { ethers } from "ethers";
import * as dotenv from "dotenv";
import * as fs from "fs";
dotenv.config();

import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

const ORCH_ADDR = process.env.ORCH_ADDR!;
const RPC_URL   = process.env.RPC_URL!;
const PK        = process.env.PRIVATE_KEY!;
const TRAIN_CSV = process.env.TRAIN_CSV   || "client/train.csv";
const HOLD_CSV  = process.env.HOLDOUT_CSV || process.env.DATASET_CSV || "client/dataset.csv";

if (!ORCH_ADDR || !ORCH_ADDR.startsWith("0x")) throw new Error(`ORCH_ADDR malformed: ${ORCH_ADDR}`);
if (!RPC_URL) throw new Error("RPC_URL missing");
if (!PK)      throw new Error("PRIVATE_KEY missing");

type Sample = { x0: number; x1: number; y: number };

function quant01ToQ3(v: string | number): number {
  const f = typeof v === "number" ? v : parseFloat(v.trim());
  if (!Number.isFinite(f)) throw new Error(`bad number: ${v}`);
  let q = Math.round(Math.min(1, Math.max(0, f)) * 3);
  if (q < 0) q = 0; if (q > 3) q = 3;
  return q;
}

function loadCsv2(path: string): Sample[] {
  const txt = fs.readFileSync(path, "utf8");
  const out: Sample[] = [];
  const lines = txt.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw) continue;
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cells = line.split(/[,;\s]+/).filter(Boolean);
    if (cells.length < 3) continue;
    if (i === 0 && (isNaN(Number(cells[0])) || /[A-Za-z]/.test(cells[0]))) continue; // header
    const x0 = quant01ToQ3(cells[0]);
    const x1 = quant01ToQ3(cells[1]);
    const y  = Number(cells[2]);
    if (![0,1].includes(y)) throw new Error(`Row ${i+1}: y must be 0/1`);
    out.push({ x0, x1, y });
  }
  if (out.length === 0) throw new Error(`Empty/invalid CSV at ${path}`);
  return out;
}

// Sorted-pair Merkle (directionless) utilities — must match contract
function keccakPacked(types: string[], values: any[]): string {
  return ethers.keccak256(ethers.solidityPacked(types, values));
}
function hashPairSorted(a: string, b: string): string {
  return a.toLowerCase() < b.toLowerCase()
    ? keccakPacked(["bytes32","bytes32"], [a, b])
    : keccakPacked(["bytes32","bytes32"], [b, a]);
}
function merkleRootSorted(leaves: string[]): string {
  if (leaves.length === 0) return "0x" + "00".repeat(32);
  let level = leaves.slice();
  while (level.length > 1) {
    const next: string[] = [];
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) next.push(hashPairSorted(level[i], level[i+1]));
      else                      next.push(hashPairSorted(level[i], level[i]));
    }
    level = next;
  }
  return level[0];
}

function buildTrainingRoot(samples: Sample[]): string {
  const leaves = samples.map((s, i) =>
    keccakPacked(["uint256","uint256","uint256","uint256"], [i, s.x0, s.x1, s.y])
  );
  return merkleRootSorted(leaves);
}

(async () => {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PK, provider);
  const orch     = new ethers.Contract(ORCH_ADDR, OrchAbi.abi, wallet);

  // Load datasets
  const train = loadCsv2(TRAIN_CSV);
  const hold  = loadCsv2(HOLD_CSV);
  const trainRoot = buildTrainingRoot(train);
  console.log(`training set: ${train.length} rows, root=${trainRoot}`);
  console.log(`hold-out set: ${hold.length} rows from ${HOLD_CSV}`);

  // Hyperparam grid (tune to get variability)
  const grid: Array<{ lr: bigint | number; steps: bigint | number }> = [
    { lr:   1_000, steps:  20 },
    { lr:   5_000, steps:  40 },
    { lr:  20_000, steps:  60 },
    { lr:  50_000, steps:  80 },
    { lr:  80_000, steps: 120 },
    { lr: 150_000, steps:  20 },
    { lr: 150_000, steps: 200 },
    { lr:     500, steps:   8 },
  ];

  const perTask = ethers.parseEther("0.01");
  const bounty  = perTask * BigInt(grid.length);
  const minWorkers = 2;

  // 1) open
  const tx = await orch.openRequest("file://train+holdout.csv", grid, minWorkers, { value: bounty });
  const receipt = await tx.wait();

  // Extract request id
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

  // 2) set training root
  await (await orch.setTrainingDatasetRoot(openedId, trainRoot, train.length)).wait();
  console.log("✓ training root set");

  // 3) push hold-out arrays
  const hx0 = hold.map(s => BigInt(s.x0));
  const hx1 = hold.map(s => BigInt(s.x1));
  const hy  = hold.map(s => BigInt(s.y));
  await (await orch.setHoldoutDataset(openedId, hx0, hx1, hy)).wait();
  console.log("✓ hold-out dataset set");
})();

// client/open_request.ts
import { ethers, TransactionReceipt, Log, EventLog } from "ethers";
import * as dotenv from "dotenv";
import * as fs from "fs";
dotenv.config();

import OrchAbi from "../artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json";

const PK = (process.env.CLIENT_PRIVATE_KEY || process.env.PRIVATE_KEY || "").trim();
const ORCH_ADDR = (process.env.ORCH_ADDR || "").trim();
const RPC_URL   = (process.env.RPC_URL   || "").trim();
const TRAIN_CSV = process.env.TRAIN_CSV   || "client/train.csv";
const HOLD_CSV  = process.env.HOLDOUT_CSV || process.env.DATASET_CSV || "client/dataset.csv";

if (!PK || !PK.startsWith("0x")) throw new Error("CLIENT_PRIVATE_KEY or PRIVATE_KEY missing/malformed");
if (!ORCH_ADDR || !ORCH_ADDR.startsWith("0x")) throw new Error("ORCH_ADDR malformed");
if (!RPC_URL) throw new Error("RPC_URL missing");

type Sample = { x0: number; x1: number; y: number };

// Quantize floats in [0,1] to **0..15** (round-half-up) — MUST match worker & contract XMAX=15
function quant01ToQ15(v: string | number): number {
  const f = typeof v === "number" ? v : parseFloat(String(v).trim());
  if (!Number.isFinite(f)) throw new Error(`bad number: ${v}`);
  let q = Math.round(Math.min(1, Math.max(0, f)) * 15);
  if (q < 0) q = 0; if (q > 15) q = 15;
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
    // skip header if first cell non-numeric
    if (i === 0 && (isNaN(Number(cells[0])) || /[A-Za-z]/.test(cells[0]))) continue;
    const x0 = quant01ToQ15(cells[0]);
    const x1 = quant01ToQ15(cells[1]);
    const y  = Number(cells[2]);
    if (![0, 1].includes(y)) throw new Error(`Row ${i + 1}: y must be 0/1`);
    out.push({ x0, x1, y });
  }
  if (out.length === 0) throw new Error(`Empty/invalid CSV at ${path}`);
  return out;
}

// Sorted‑pair Merkle helpers — must match contract
function keccakPacked(types: string[], values: any[]): string {
  return ethers.keccak256(ethers.solidityPacked(types, values));
}
function hashPairSorted(a: string, b: string): string {
  return a.toLowerCase() < b.toLowerCase()
    ? keccakPacked(["bytes32", "bytes32"], [a, b])
    : keccakPacked(["bytes32", "bytes32"], [b, a]);
}
function merkleRootSorted(leaves: string[]): string {
  if (leaves.length === 0) return "0x" + "00".repeat(32);
  let level = leaves.slice();
  while (level.length > 1) {
    const next: string[] = [];
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) next.push(hashPairSorted(level[i], level[i + 1]));
      else next.push(hashPairSorted(level[i], level[i]));
    }
    level = next;
  }
  return level[0];
}
function buildTrainingRoot(samples: Sample[]): string {
  const leaves = samples.map((s, i) =>
    keccakPacked(["uint256", "uint256", "uint256", "uint256"], [i, s.x0, s.x1, s.y]),
  );
  return merkleRootSorted(leaves);
}

// ─────────────────────────────────────────────────────────────
// Local nonce sender: explicit nonces, no gas fields, with retry
// ─────────────────────────────────────────────────────────────
function explainError(e: any): string {
  const msg = (e?.shortMessage || e?.message || "").toLowerCase();
  const code = e?.code || "";
  const dataMsg = e?.info?.error?.message || e?.info?.error?.data?.message || "";
  return `${code} ${msg} ${dataMsg}`.trim();
}

async function makeLocalNonceSender(
  wallet: ethers.Wallet,
  provider: ethers.JsonRpcProvider,
  to: string
) {
  let next = await provider.getTransactionCount(wallet.address, "latest");

  async function waitReceiptStrict(hash: string): Promise<TransactionReceipt> {
    const MAX_MS = 60_000;
    const start = Date.now();
    while (true) {
      const got = await provider.waitForTransaction(hash, null, 5_000);
      if (got) return got;
      const direct = await provider.getTransactionReceipt(hash);
      if (direct) return direct;
      if (Date.now() - start > MAX_MS) throw new Error(`tx ${hash} not mined within ${MAX_MS}ms`);
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  async function send(data: string, value: bigint = 0n): Promise<TransactionReceipt> {
    while (true) {
      try {
        const tx = await wallet.sendTransaction({ to, data, value, nonce: next });
        const rcpt0 = await tx.wait();
        const rcpt  = rcpt0 ?? (await waitReceiptStrict(tx.hash));
        next += 1;
        return rcpt;
      } catch (e: any) {
        const msg = explainError(e);
        if (
          e?.code === "NONCE_EXPIRED" ||
          msg.includes("nonce too low") ||
          msg.includes("already known") ||
          msg.includes("can't be queued")
        ) {
          next = await provider.getTransactionCount(wallet.address, "latest");
          continue;
        }
        if (msg.includes("cannot send both gasprice and maxfeepergas")) {
          continue;
        }
        throw e;
      }
    }
  }
  return { send };
}

(async () => {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PK, provider);
  const orch     = new ethers.Contract(ORCH_ADDR, OrchAbi.abi, wallet);
  const orchAddr = ORCH_ADDR;

  // Warn if client key == worker key (nonce races)
  const wk = (process.env.PRIVATE_KEY || "").trim();
  if (wk && wk.startsWith("0x")) {
    try {
      if (new ethers.Wallet(wk).address.toLowerCase() === wallet.address.toLowerCase()) {
        console.warn("⚠️  CLIENT_PRIVATE_KEY and PRIVATE_KEY are the same. Use distinct keys to avoid nonce races.");
      }
    } catch {}
  }

  const sender = await makeLocalNonceSender(wallet, provider, orchAddr);

  // Load datasets (quantize 0..15)
  const train = loadCsv2(TRAIN_CSV);
  const hold  = loadCsv2(HOLD_CSV);

  const trainRoot = buildTrainingRoot(train);
  const holdRoot  = buildTrainingRoot(hold);  // same leaf rule (i, x0, x1, y)
  console.log(`training set: ${train.length} rows, root=${trainRoot}`);
  console.log(`hold-out set: ${hold.length} rows from ${HOLD_CSV}`);

  // Hyperparam grid (you can enlarge freely now)
  const grid: Array<{ lr: bigint | number; steps: bigint | number }> = [
    { lr:   5_000,  steps:  1 },  // L=1, very underfit
    { lr:   5_000,  steps:  2 },  // L=1
    { lr:  20_000,  steps:  4 },  // L=1
    { lr:  50_000,  steps:  6 },  // L=2
    { lr:  80_000,  steps:  8 },  // L=2
    { lr: 100_000,  steps: 10 },  // L=3
    { lr: 120_000,  steps: 12 },  // L=3
    { lr: 150_000,  steps: 16 },  // L=3
  ];

  const perTask = ethers.parseEther("0.01");
  const bounty  = perTask * BigInt(grid.length);
  const minWorkers = 2;

  // 1) openRequest
  const dataOpen = orch.interface.encodeFunctionData("openRequest", [
    "file://train+holdout.csv",
    grid,
    minWorkers,
  ]);
  const rcptOpen: TransactionReceipt = await sender.send(dataOpen, bounty);

  // Recover id
  const topicReqOpened = ethers.id("RequestOpened(uint256,uint256,uint256,uint256)");
  let openedId: bigint | null = null;
  const logs: Array<Log | EventLog> = (rcptOpen.logs ?? []) as Array<Log | EventLog>;
  for (const log of logs) {
    if (log.address.toLowerCase() !== orchAddr.toLowerCase()) continue;
    const topic0 = (log as any).topics?.[0]?.toLowerCase?.() || "";
    if (topic0 !== topicReqOpened.toLowerCase()) continue;
    const [id] = ethers.AbiCoder.defaultAbiCoder().decode(
      ["uint256","uint256","uint256","uint256"], (log as any).data
    );
    openedId = id as bigint; break;
  }
  if (openedId === null) throw new Error("Could not recover RequestOpened.id from logs");
  console.log("request id", openedId.toString());

  // 2) setTrainingDatasetRoot
  const dataRoot = orch.interface.encodeFunctionData("setTrainingDatasetRoot", [
    openedId, trainRoot, train.length,
  ]);
  await sender.send(dataRoot);

  // 3) NEW: setHoldoutDatasetRoot (no more array upload!)
  const dataHoldRoot = orch.interface.encodeFunctionData("setHoldoutDatasetRoot", [
    openedId, holdRoot, hold.length,
  ]);
  await sender.send(dataHoldRoot);

  console.log("✓ training root and hold-out root set");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

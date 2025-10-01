// utils.ts
// -----------------------------------------------------------------------------
// Test helper utilities for ai-zk-compute
// - Avoid Mocha's global `run()` by using `sh()` for shelling out.
// - Ensures MLP ZK artifacts exist, compiles contracts, starts a HH node,
//   deploys the Groth16 verifier + AiOrchestrator, commits dataset roots,
//   and spawns/stops Python workers.
// -----------------------------------------------------------------------------

import { spawn, SpawnOptions, ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";
import {
  JsonRpcProvider,
  Wallet,
  Contract,
  ContractFactory,
  NonceManager,
  parseEther,
  solidityPackedKeccak256,
} from "ethers";

const ROOT = path.resolve(__dirname, "../.."); // project root
const NODE_DIR = path.join(ROOT, "node");

function pjoin(...p: string[]) {
  return path.join(ROOT, ...p);
}
function exists(rel: string) {
  return fs.existsSync(pjoin(rel));
}

/** Shell out to a process (renamed from `run` → `sh` to avoid Mocha global). */
export async function sh(
  cmd: string,
  args: string[],
  opts: SpawnOptions & { quiet?: boolean } = {},
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, {
      cwd: ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
      ...opts,
    });
    let out = "";
    let err = "";
    if (p.stdout) p.stdout.on("data", (d) => { out += d.toString(); if (!opts.quiet) process.stdout.write(d); });
    if (p.stderr) p.stderr.on("data", (d) => { err += d.toString(); if (!opts.quiet) process.stderr.write(d); });
    p.on("error", reject);
    p.on("close", (code) => resolve({ code, stdout: out, stderr: err }));
  });
}

// -----------------------------------------------------------------------------
// Ensure local deps / artifacts / compilation
// -----------------------------------------------------------------------------

export async function ensureDeps() {
  if (!exists("node/.venv/bin/python")) {
    await sh("make", ["deps"]);
  }
}

export async function ensureCircuits() {
  const has =
    exists("circuits/MlpHoldoutAcc_256_final.zkey") &&
    exists("circuits/MlpHoldoutAcc_256_js/MlpHoldoutAcc_256.wasm");
  if (!has) {
    await sh("make", ["mlp-zk"]);
  }
}

export async function ensureCompiled() {
  await sh("pnpm", ["hardhat", "compile"]);
}

// -----------------------------------------------------------------------------
// Hardhat node lifecycle
// -----------------------------------------------------------------------------

export async function startNode(): Promise<{
  kill: () => void;
  url: string;
  accounts: { addr: string; pk: string }[];
}> {
  const node = spawn(
    "npx",
    ["hardhat", "node", "--hostname", "127.0.0.1", "--port", "8545"],
    { cwd: ROOT, env: process.env, stdio: ["ignore", "pipe", "pipe"] }
  );

  if (node.stdout) node.stdout.setEncoding("utf8");
  if (node.stderr) node.stderr.setEncoding("utf8");

  await new Promise<void>((resolve, reject) => {
    const to = setTimeout(() => reject(new Error("hardhat node start timeout")), 15000);
    node.on("error", reject);
    if (node.stdout) node.stdout.on("data", (d: string) => {
      process.stdout.write(d);
      if (d.includes("Started HTTP and WebSocket JSON-RPC server")) {
        clearTimeout(to);
        resolve();
      }
    });
    if (node.stderr) node.stderr.on("data", (d: string) => process.stderr.write(d));
  });

  const url = "http://127.0.0.1:8545";

  // Full default HH dev set: #1 is deployer/client; workers use #2..#13
  const accounts = [
    { addr: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266", pk: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80" }, // #1 (deployer)
    { addr: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8", pk: "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d" }, // #2
    { addr: "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", pk: "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a" }, // #3
    { addr: "0x90F79bf6EB2c4f870365E785982E1f101E93b906", pk: "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6" }, // #4
    { addr: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65", pk: "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a" }, // #5
    { addr: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc", pk: "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba" }, // #6
    { addr: "0x976EA74026E726554dB657fA54763abd0C3a0aa9", pk: "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e" }, // #7
    { addr: "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955", pk: "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356" }, // #8
    { addr: "0x23618e81E3f5cdF7F54C3d65f7FBc0aBf5B21E8f", pk: "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97" }, // #9
    { addr: "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720", pk: "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6" }, // #10
    { addr: "0xBcd4042DE499D14e55001CcbB24a551F3b954096", pk: "0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897" }, // #11
    { addr: "0x71bE63f3384f5fb98995898A86B02Fb2426c5788", pk: "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82" }, // #12
    { addr: "0xFABB0ac9d68B0B445fB7357272Ff202C5651694a", pk: "0xa267530f49f8280200edf313ee7af6b827f2a8bce2897751d06a843f644967b1" }, // #13
  ];

  return {
    kill: () => { try { node.kill("SIGINT"); } catch {} },
    url,
    accounts,
  };
}

// -----------------------------------------------------------------------------
// Deploy verifier + orchestrator
// -----------------------------------------------------------------------------

function readJson(p: string) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

/** Handle all naming variants we might get from `snarkjs zkey export solidityverifier`. */
function loadVerifierArtifact() {
  const candidates = [
    // Preferred names in your repo
    pjoin("artifacts/contracts/AccVerifier.sol/AccVerifier.json"),
    pjoin("artifacts/contracts/AccVerifier.sol/Verifier.json"),
    // What you actually have right now (see your screenshot)
    pjoin("artifacts/contracts/AccVerifier.sol/Groth16Verifier.json"),
    // Rare legacy layout
    pjoin("artifacts/contracts/Groth16Verifier.sol/Groth16Verifier.json"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return readJson(c);
  }
  throw new Error(
    "Verifier artifact not found.\n" +
    candidates.map((c) => " - " + c).join("\n") +
    "\nFix: run 'make mlp-zk' and 'pnpm hardhat compile' (the loader accepts Groth16Verifier.json too)."
  );
}

export async function deployAll(url: string) {
  const provider = new JsonRpcProvider(url);

  // Use NonceManager to avoid nonce races.
  const base = new Wallet(
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    provider
  );
  const wallet = new NonceManager(base);

  // Verifier compiled from snarkjs export (any of the accepted names above)
  const verArt = loadVerifierArtifact();
  const orchArt = readJson(
    pjoin("artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json")
  );

  const Ver = new ContractFactory(verArt.abi, verArt.bytecode, wallet);
  const ver = await Ver.deploy();
  await ver.waitForDeployment();

  const Orch = new ContractFactory(orchArt.abi, orchArt.bytecode, wallet);
  const orch = await Orch.deploy(await ver.getAddress());
  await orch.waitForDeployment();

  return {
    provider,
    wallet: wallet as unknown as Wallet,
    verAddr: await ver.getAddress(),
    orchAddr: await orch.getAddress(),
    orchAbi: orchArt.abi,
  };
}

// -----------------------------------------------------------------------------
// CSV → quantize → Merkle roots → set roots on-chain
// -----------------------------------------------------------------------------

type Row = { x0: number; x1: number; y: number };

function parseCsv(fp: string): Row[] {
  const raw = fs.readFileSync(fp, "utf8");
  const lines = raw.split(/\r?\n/);
  const rows: Row[] = [];
  for (const line of lines) {
    const s = line.trim();
    if (!s || s.startsWith("#")) continue;
    const parts = s.split(/[,;\s]+/).filter(Boolean);
    if (parts.length < 3) continue;

    const [a, b, c] = parts.slice(0, 3);
    const nx0 = Number(a), nx1 = Number(b), ny = Number(c);
    if (!Number.isFinite(nx0) || !Number.isFinite(nx1) || !Number.isFinite(ny)) {
      continue; // header or malformed row
    }

    // Quantize features to 4-bit nibbles (half-up on [0,1]); labels to {0,1}
    const q = (v: number) => {
      if (v <= 1.0000001 && v >= 0) return Math.min(15, Math.max(0, Math.floor(v * 15 + 0.5)));
      return Math.min(15, Math.max(0, Math.round(v)));
    };
    const x0 = q(nx0);
    const x1 = q(nx1);
    const y = ny > 0 ? 1 : 0;

    rows.push({ x0, x1, y });
  }
  return rows;
}

function leaf(i: number, r: Row): string {
  // abi.encodePacked(uint256 i, uint8 x0, uint8 x1, uint8 y)
  return solidityPackedKeccak256(
    ["uint256", "uint8", "uint8", "uint8"],
    [BigInt(i), r.x0, r.x1, r.y]
  );
}

function sortedPairParent(a: string, b: string): string {
  const [lo, hi] = (a.toLowerCase() <= b.toLowerCase()) ? [a, b] : [b, a];
  return solidityPackedKeccak256(["bytes32", "bytes32"], [lo, hi]);
}

function merkleRoot(leaves: string[]): string {
  if (leaves.length === 0) return "0x" + "00".repeat(64);
  let level = leaves.slice();
  while (level.length > 1) {
    const next: string[] = [];
    for (let i = 0; i < level.length; i += 2) {
      const a = level[i];
      const b = (i + 1 < level.length) ? level[i + 1] : level[i];
      next.push(sortedPairParent(a, b));
    }
    level = next;
  }
  return level[0];
}

/**
 * Compute roots for client/train.csv and client/dataset.csv and
 * commit them on-chain for the given request id.
 *
 * Required by the contract before claims are allowed ("holdout not set").
 * See Section 5.3 ("setTrainingDatasetRoot, setHoldoutDatasetRoot"). :contentReference[oaicite:2]{index=2}
 */
export async function commitDatasets(
  orch: Contract,
  client: Wallet,
  id: number,
  trainCsv = pjoin("client/train.csv"),
  holdoutCsv = pjoin("client/dataset.csv"),
) {
  const T = parseCsv(trainCsv);
  const H = parseCsv(holdoutCsv);

  const tLeaves = T.map((r, i) => leaf(i, r));
  const hLeaves = H.map((r, i) => leaf(i, r));

  const tRoot = merkleRoot(tLeaves);
  const hRoot = merkleRoot(hLeaves);

  const O = orch.connect(client) as any;

  if (O.setTrainingDatasetRoot && O.setHoldoutDatasetRoot) {
    await (await O.setTrainingDatasetRoot(id, tRoot, T.length)).wait();
    await (await O.setHoldoutDatasetRoot(id, hRoot, H.length)).wait();
  } else {
    throw new Error(
      "Expected setTrainingDatasetRoot/setHoldoutDatasetRoot in ABI. " +
      "Update the contract or provide array-mode helpers."
    );
  }
}

// -----------------------------------------------------------------------------
// Worker process management
// -----------------------------------------------------------------------------

export function spawnWorker({
  url,
  orch,
  requestId,
  priv,
}: {
  url: string;
  orch: string;
  requestId: number;
  priv: string;
}) {
  const env = {
    ...process.env,
    RPC_URL: url,
    ORCH_ADDR: orch,
    REQUEST_ID: String(requestId),
    PRIVATE_KEY: priv,
    // Explicit MLP artifacts for the Python node
    ACC_WASM: path.join(ROOT, "circuits", "MlpHoldoutAcc_256_js", "MlpHoldoutAcc_256.wasm"),
    ACC_ZKEY: path.join(ROOT, "circuits", "MlpHoldoutAcc_256_final.zkey"),
    ZK_ACC_WASM: path.join(ROOT, "circuits", "MlpHoldoutAcc_256_js", "MlpHoldoutAcc_256.wasm"),
    ZK_ACC_ZKEY: path.join(ROOT, "circuits", "MlpHoldoutAcc_256_final.zkey"),
  };
  const p = spawn(
    path.join(NODE_DIR, ".venv", "bin", "python"),
    ["compute_node.py"],
    {
      cwd: NODE_DIR,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    }
  );
  if (p.stdout) p.stdout.on("data", (d) => process.stdout.write(d));
  if (p.stderr) p.stderr.on("data", (d) => process.stderr.write(d));
  return p;
}

function waitForExit(p: ChildProcess, timeoutMs = 5000) {
  return new Promise<void>((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    const t = setTimeout(finish, timeoutMs);
    p.once("exit", () => { clearTimeout(t); finish(); });
    p.once("close", () => { clearTimeout(t); finish(); });
  });
}

export async function stopWorker(p: ChildProcess, timeoutMs = 4000) {
  try {
    const pid = p.pid;
    if (typeof pid !== "number") {
      try { p.kill("SIGINT"); } catch {}
      await waitForExit(p, timeoutMs);
      return;
    }
    try { process.kill(-pid, "SIGINT"); } catch {}
    try { p.kill("SIGINT"); } catch {}

    const t = setTimeout(() => {
      try { process.kill(-pid, "SIGKILL"); } catch {}
      try { p.kill("SIGKILL"); } catch {}
    }, timeoutMs);

    await waitForExit(p, timeoutMs + 1000);
    clearTimeout(t);
  } catch {
    // swallow
  }
}

// -----------------------------------------------------------------------------
// On-chain helpers
// -----------------------------------------------------------------------------

/** Open a request and return its id. */
export async function openRequest(
  orch: Contract,
  client: Wallet,
  minWorkers: number,
  bountyEth: string,
  grid: { lr: bigint; steps: bigint }[],
) {
  const O = orch.connect(client) as any;
  const nextId: bigint = await O.nextId();
  const tx = await O.openRequest("cid://xor", grid, minWorkers, {
    value: parseEther(bountyEth),
  });
  await tx.wait();
  return Number(nextId);
}

export async function waitClosed(orch: Contract, id: number, maxMs = 180000) {
  const O = orch as any;
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    const r = await O.getResult(id);
    const closed = (r?.closed ?? r?.[0]) as boolean;
    if (closed) return r;
    await new Promise((res) => setTimeout(res, 1000));
  }
  throw new Error("Timeout waiting for settlement");
}

export async function showCredits(orch: Contract, id: number, addrs: string[]) {
  const O = orch as any;
  for (const a of addrs) {
    const c = await O.credit(id, a);
    console.log("credit", a, String(c));
  }
}

export function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

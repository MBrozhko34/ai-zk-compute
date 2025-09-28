// test/e2e/utils.ts
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";
import fs from "fs";
import { ethers } from "ethers";

export type Worker = { pk: string; addr: string };
export const HARDHAT_RPC = "http://127.0.0.1:8545";

// First 10 keys from your Makefile / Hardhat defaults
export const ACCOUNTS: Worker[] = [
  ["0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"],
  ["0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a", "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"],
  ["0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", "0x90F79bf6EB2c4f870365E785982E1f101E93b906"],
  ["0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"],
  ["0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"],
  ["0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", "0x976EA74026E726554dB657fA54763abd0C3a0aa9"],
  ["0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356", "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955"],
  ["0xdbda1821b80551c9d65939329250298aa3472ba22feea9210cf5d620ea67b97", "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f"],
  ["0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6", "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"],
  ["0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897", "0xBcd4042DE499D14e55001CcbB24a551F3b954096"],
].map(([pk, addr]) => ({ pk, addr }));

export function startHardhatNode(): ChildProcessWithoutNullStreams {
  const child = spawn("npx", ["hardhat", "node"], { stdio: "pipe" });
  let ready = false;
  child.stdout.on("data", (b) => {
    const s = String(b);
    if (!ready && s.toLowerCase().includes("started http")) ready = true;
  });
  // crude wait; tests will still retry RPC calls
  return child;
}

export function stop(child: ChildProcessWithoutNullStreams) {
  try { child.kill("SIGKILL"); } catch {}
}

export function writeCSV(file: string, rows: Array<[number, number, number]>) {
  const dir = path.dirname(file);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const body = rows.map(([a,b,c]) => `${a} ${b} ${c}`).join("\n");
  fs.writeFileSync(file, body);
}

// round-half-up quantize to 0..15
export function q15(x: number): number {
  const f = Math.max(0, Math.min(1, x));
  let q = Math.round(f * 15);
  if (q < 0) q = 0; if (q > 15) q = 15;
  return q;
}

export function makeDataset(nTrain = 64, nHold = 16, seed = 42) {
  const rnd = (function() {
    let s = seed >>> 0;
    return () => ((s = (1664525 * s + 1013904223) >>> 0) / 0xffffffff);
  })();
  function gen(n: number) {
    const rows: Array<[number, number, number]> = [];
    for (let i = 0; i < n; i++) {
      const x0 = q15(rnd());
      const x1 = q15(rnd());
      // linear rule with noise
      const y = ((x0 + 0.6 * x1 + (rnd() < 0.08 ? (rnd() < 0.5 ? -5 : +5) : 0)) > 12) ? 1 : 0;
      rows.push([x0, x1, y ? 1 : 0]);
    }
    return rows;
  }
  return { train: gen(nTrain), hold: gen(nHold) };
}

export async function deployOrch(
  abiPath: string,
  rpc = HARDHAT_RPC
): Promise<{
  orch: ethers.Contract;
  addr: string;
  wallet: ethers.Wallet;
  provider: ethers.JsonRpcProvider;
}> {
  const json = JSON.parse(fs.readFileSync(abiPath, "utf8"));
  const bytecode = json.bytecode as `0x${string}`;
  const abi = json.abi;

  const provider = new ethers.JsonRpcProvider(rpc);
  const wallet = new ethers.Wallet(ACCOUNTS[0].pk, provider);

  const factory = new ethers.ContractFactory(abi, bytecode, wallet);
  const deployed = await factory.deploy();
  await deployed.waitForDeployment();
  const addr = await deployed.getAddress();

  // Re-wrap as a canonical ethers.Contract (clean typings)
  const orch = new ethers.Contract(addr, abi, wallet);

  return { orch, addr, wallet, provider };
}


export async function openRequestDirect(
  orch: ethers.Contract,
  client: ethers.Wallet,
  train: Array<[number, number, number]>,
  hold: Array<[number, number, number]>,
  grid: Array<{ lr: number; steps: number }>,
  bountyPerTaskEth = "0.01",
  minWorkers = 2
) {
  function leaf(i: number, s: [number, number, number]) {
    const [x0, x1, y] = s;
    return ethers.keccak256(
      ethers.solidityPacked(
        ["uint256", "uint256", "uint256", "uint256"],
        [i, x0, x1, y]
      )
    );
  }
  function merkleRoot(leaves: string[]) {
    if (leaves.length === 0) return "0x" + "00".repeat(32);
    let level = leaves.slice();
    while (level.length > 1) {
      const next: string[] = [];
      for (let i = 0; i < level.length; i += 2) {
        const a = level[i];
        const b = i + 1 < level.length ? level[i + 1] : level[i];
        const pair =
          a.toLowerCase() < b.toLowerCase()
            ? ethers.keccak256(
                ethers.solidityPacked(["bytes32", "bytes32"], [a, b])
              )
            : ethers.keccak256(
                ethers.solidityPacked(["bytes32", "bytes32"], [b, a])
              );
        next.push(pair);
      }
      level = next;
    }
    return level[0];
  }

  const trainLeaves = train.map((s, i) => leaf(i, s));
  const holdLeaves = hold.map((s, i) => leaf(i, s));
  const trainRoot = merkleRoot(trainLeaves);
  const holdRoot = merkleRoot(holdLeaves);

  const perTask = ethers.parseEther(bountyPerTaskEth);
  const bounty = perTask * BigInt(grid.length);

  // bind a signer once
  const orchWithClient = orch.connect(client);

  // ---- call functions via getFunction(...) to satisfy TS ----
  const openRequestFn = orchWithClient.getFunction("openRequest");
  const setTrainRootFn = orchWithClient.getFunction("setTrainingDatasetRoot");
  const setHoldRootFn = orchWithClient.getFunction("setHoldoutDatasetRoot");

  // open
  const txOpen = await openRequestFn(
    "file://train+hold.csv",
    grid.map((g) => ({ lr: g.lr, steps: g.steps })),
    minWorkers,
    { value: bounty }
  );
  const rcptOpen = await txOpen.wait();

  // extract id from event
  const event = rcptOpen!.logs.find(
    (l: any) => (l as any).fragment?.name === "RequestOpened"
  ) as any;
  const id: bigint = event?.args?.[0] ?? 0n;

  // set roots (no more TS error)
  await (await setTrainRootFn(id, trainRoot, train.length)).wait();
  await (await setHoldRootFn(id, holdRoot, hold.length)).wait();

  return Number(id);
}


export function spawnWorker(opts: {
  orchAddr: string; requestId: number; pk: string;
  trainCsv: string; holdCsv: string; cwd?: string; extraEnv?: Record<string,string>;
}) {
  const p = spawn(
    process.platform === "win32" ? "python" : "python3",
    [ "compute_node.py" ],
    {
      cwd: opts.cwd ?? path.resolve(process.cwd(), "node"),
      env: {
        ...process.env,
        RPC_URL: HARDHAT_RPC,
        ORCH_ADDR: opts.orchAddr,
        REQUEST_ID: String(opts.requestId),
        PRIVATE_KEY: opts.pk,
        TRAIN_CSV: opts.trainCsv,
        HOLDOUT_CSV: opts.holdCsv,
        ENABLE_REASSIGN: "1",
        AUTO_WITHDRAW: "0", // tests prefer to inspect credit on-chain; workers won’t withdraw here
        ...(opts.extraEnv ?? {}),
      },
      stdio: "pipe",
    }
  );
  return p;
}

export async function waitForClosed(orch: ethers.Contract, id: number, timeoutMs = 120_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const [closed] = await orch.getResult(id);
    if (closed) return true;
    await new Promise(r => setTimeout(r, 500));
  }
  return false;
}

import { describe, it, beforeEach, afterEach } from "mocha";
import type { ChildProcess } from "child_process";
import { Contract, Wallet } from "ethers";
import {
  ensureDeps,
  ensureCircuits,
  ensureCompiled,
  startNode,
  deployAll,
  spawnWorker,
  stopWorker,
  openRequest,
  waitClosed,
  showCredits,
  sleep,
} from "./utils";

const GRID = [
  { lr: 500n, steps: 5n },
  { lr: 1000n, steps: 30n },
  { lr: 20000n, steps: 40n },
  { lr: 50000n, steps: 80n },
  { lr: 80000n, steps: 120n },
  { lr: 150000n, steps: 20n },
  { lr: 150000n, steps: 300n },
  { lr: 250000n, steps: 10n }, // ensure UNIQUE, no duplicates
];

describe("E2E: AiOrchestrator + Python workers + snarkjs", function () {
  this.timeout(300_000);

  let url: string;
  let nodeKill: () => void;
  let accounts: { addr: string; pk: string }[];
  let orchAddr: string;
  let orchAbi: any;
  let orch: Contract;
  let client: Wallet;

  let workers: ChildProcess[] = [];
  let timers: NodeJS.Timeout[] = [];
  const addTimer = (t: NodeJS.Timeout) => { timers.push(t); return t; };

  beforeEach(async () => {
    await ensureDeps();
    await ensureCircuits();
    await ensureCompiled();

    const node = await startNode();
    url = node.url;
    nodeKill = node.kill;
    accounts = node.accounts;

    const { wallet, orchAddr: oa, orchAbi: abi } = await deployAll(url);
    client = wallet as unknown as Wallet; // NonceManager-wrapped signer is fine here
    orchAddr = oa;
    orchAbi = abi;
    orch = new Contract(orchAddr, orchAbi, client);

    workers = [];
    timers = [];
  });

  afterEach(async () => {
    for (const t of timers) { try { clearTimeout(t); } catch {} }
    timers = [];

    for (const p of workers) {
      try { await stopWorker(p); } catch {}
    }
    workers = [];

    try { nodeKill(); } catch {}
    await sleep(200);
  });

  it("no dropouts (minWorkers=4)", async () => {
    const id = await openRequest(orch, client, 4, "1.0", GRID);
    workers.push(
      spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[0].pk }),
      spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[1].pk }),
      spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[2].pk }),
      spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[3].pk }),
    );

    const res = await waitClosed(orch, id);
    const bestAcc = Number(((res as any).bestAcc ?? (res as any)[1])) / 100;
    console.log("settled bestAcc=", bestAcc, "%");
    await showCredits(orch, id, accounts.slice(0, 4).map((a) => a.addr));
  });

  it("dropout + last-task fast-path (minWorkers=2) + late joiner", async () => {
    const id = await openRequest(orch, client, 2, "1.0", GRID);
    const p1 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[1].pk });
    const p2 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[2].pk });
    workers.push(p1, p2);

    addTimer(setTimeout(() => { try { p2.kill("SIGINT"); } catch {} }, 5000));
    addTimer(setTimeout(() => {
      const p3 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[3].pk });
      workers.push(p3);
    }, 15000));

    const res = await waitClosed(orch, id);
    const bestAcc = Number(((res as any).bestAcc ?? (res as any)[1])) / 100;
    console.log("settled bestAcc=", bestAcc, "%");
    await showCredits(orch, id, accounts.slice(0, 4).map((a) => a.addr));
  });
});

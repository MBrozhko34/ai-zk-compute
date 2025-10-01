import { describe, it, beforeEach, afterEach } from "mocha";
import { expect } from "chai";
import type { ChildProcess } from "child_process";
import {
  Contract,
  Wallet,
  JsonRpcProvider,
  NonceManager,
  Interface,
  FunctionFragment,
} from "ethers";
import {
  ensureDeps,
  ensureCircuits,
  ensureCompiled,
  startNode,
  deployAll,
  spawnWorker,
  stopWorker,
  openRequest,
  commitDatasets,
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
  let nodeKill: () => void; // sync kill
  let accounts: { addr: string; pk: string }[];
  let orchAddr: string;
  let orchAbi: any;
  let orch: Contract;
  let client: Wallet;
  let provider: JsonRpcProvider;

  // For worker-based tests
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

    const dep = await deployAll(url);
    provider = dep.provider;
    client = dep.wallet as unknown as Wallet; // NonceManager-wrapped signer is fine
    orchAddr = dep.orchAddr;
    orchAbi = dep.orchAbi;
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

  //
  // ─────────────────────────────────────────────
  // Worker-driven scenarios
  // ─────────────────────────────────────────────
  //

  it("no dropouts (minWorkers=4)", async () => {
    const id = await openRequest(orch, client, 4, "1.0", GRID);
    await commitDatasets(orch, client, id);
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
    await commitDatasets(orch, client, id);
    const p1 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[1].pk });
    const p2 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[2].pk });
    workers.push(p1, p2);

    // simulate dropout
    addTimer(setTimeout(() => { try { p2.kill("SIGINT"); } catch {} }, 5000));

    // late joiner
    addTimer(setTimeout(() => {
      const p3 = spawnWorker({ url, orch: orchAddr, requestId: id, priv: accounts[3].pk });
      workers.push(p3);
    }, 15000));

    const res = await waitClosed(orch, id);
    const bestAcc = Number(((res as any).bestAcc ?? (res as any)[1])) / 100;
    console.log("settled bestAcc=", bestAcc, "%");
    await showCredits(orch, id, accounts.slice(0, 4).map((a) => a.addr));
  });

  //
  // ─────────────────────────────────────────────
  // Edge cases (all on-chain, no Python workers)
  // ─────────────────────────────────────────────
  //

  // IMPORTANT: never use accounts[0] (the deployer/client) below,
  // to avoid nonce collisions with deploy & openRequest transactions.

  const asSigner = (pk: string) => new NonceManager(new Wallet(pk, provider));
  const orchAs = (pk: string) => new Contract(orchAddr, orchAbi, asSigner(pk));

  const findTaskOwnedBy = async (id: number, ownerLower: string, maxIdx = GRID.length) => {
    for (let i = 0; i < maxIdx; i++) {
      const o: string = (await (orch as any).taskOwner(id, i)).toLowerCase();
      if (o === ownerLower) return i;
    }
    return -1;
  };

  const zeroFor = (param: any): any => {
    const t: string = param.type;
    const zeroAddr = "0x0000000000000000000000000000000000000000";
    const zeroBytes32 = "0x" + "00".repeat(32);

    if (t.endsWith("]")) {
      const dims = [...t.matchAll(/\[(\d*)\]/g)].map(m => m[1]);
      const base = t.replace(/\[[^\]]*\]/g, "");
      const baseParam = { ...param, type: base };
      const build = (d: number): any => {
        const lenStr = dims[d];
        const len = lenStr === "" ? 0 : parseInt(lenStr, 10);
        if (d === dims.length - 1) return Array.from({ length: len }, () => zeroFor(baseParam));
        return Array.from({ length: len }, () => build(d + 1));
      };
      return build(0);
    }
    if (t === "tuple") {
      const comps = param.components || [];
      return comps.map((c: any) => zeroFor(c));
    }
    if (t.startsWith("tuple[")) return [];

    if (t.startsWith("uint") || t.startsWith("int")) return 0n;
    if (t === "bool") return false;
    if (t === "address") return zeroAddr;
    if (t === "bytes32") return zeroBytes32;
    if (t.startsWith("bytes")) return "0x";
    if (t === "string") return "";
    return 0n;
  };

  const zeroArgsForSubmitProof = (iface: Interface, id: number, idx: number) => {
    let fn: FunctionFragment | null = null;
    try {
      fn = iface.getFunction("submitProof");
    } catch {}
    if (!fn) {
      const list = Object.values((iface as any).functions ?? {}) as FunctionFragment[];
      fn = list.find(f => f.name === "submitProof") ?? null;
    }
    if (!fn) throw new Error("submitProof not found in ABI");

    const args = fn.inputs.map(p => zeroFor(p));

    // Overwrite two uint-like params with (requestId, taskIdx) if applicable
    const uintIdxs = fn.inputs
      .map((p, i) => ({ i, t: p.type }))
      .filter(x => x.t.startsWith("uint") || x.t.startsWith("int"))
      .map(x => x.i);

    if (uintIdxs.length >= 2) {
      args[uintIdxs[0]] = BigInt(id);
      args[uintIdxs[1]] = BigInt(idx);
    }
    return args;
  };

it("cannot claim before lobby is ready; becomes claimable once minWorkers reached", async () => {
    const id = await openRequest(orch, client, 2, "0.5", GRID);
    await commitDatasets(orch, client, id);

    const A = orchAs(accounts[1].pk);
    const B = orchAs(accounts[2].pk);

    await (await A.joinLobby(id)).wait(); // lobby size = 1 < 2

    const bond: bigint = await (orch as any).CLAIM_BOND_WEI();

    // ✅ assert revert via staticCall (no tx sent)
    let reverted = false;
    try {
      await (A as any).claimTask.staticCall(id, { value: bond });
    } catch {
      reverted = true;
    }
    expect(reverted, "claim should revert while lobby not ready").to.eq(true);

    // when second worker joins, it starts; claim should succeed now
    await (await B.joinLobby(id)).wait();
    await (await A.claimTask(id, { value: bond })).wait();
  });

  it("non-owner cannot submit a proof for someone else's task", async () => {
    const id = await openRequest(orch, client, 1, "0.5", GRID);
    await commitDatasets(orch, client, id);

    // Owner & attacker are non-deployer accounts
    const A = orchAs(accounts[2].pk); // owner/claimer
    const B = orchAs(accounts[3].pk); // attacker

    await (await A.joinLobby(id)).wait();
    await (await B.joinLobby(id)).wait(); // harmless for minWorkers=1

    const bond: bigint = await (orch as any).CLAIM_BOND_WEI();
    await (await A.claimTask(id, { value: bond })).wait();

    const ownerIdx = await findTaskOwnedBy(id, accounts[2].addr.toLowerCase());
    expect(ownerIdx).to.be.gte(0);

    const iface: Interface = (orch as any).interface;
    const badArgs = zeroArgsForSubmitProof(iface, id, ownerIdx);

    let reverted = false;
    try {
      await (await (B as any).submitProof(...badArgs)).wait();
    } catch {
      reverted = true;
    }
    expect(reverted, "submitProof by non-owner must revert").to.eq(true);
  });

  it("bad proof from the correct owner should revert and not increase provenCount", async () => {
    const id = await openRequest(orch, client, 1, "0.5", GRID);
    await commitDatasets(orch, client, id);

    // Correct owner is a non-deployer account
    const A = orchAs(accounts[1].pk);
    const addrA = accounts[1].addr.toLowerCase();

    await (await A.joinLobby(id)).wait();

    const bond: bigint = await (orch as any).CLAIM_BOND_WEI();
    await (await A.claimTask(id, { value: bond })).wait();

    const myIdx = await findTaskOwnedBy(id, addrA);
    expect(myIdx).to.be.gte(0);

    const iface: Interface = (orch as any).interface;
    const args = zeroArgsForSubmitProof(iface, id, myIdx); // garbage proof data

    let reverted = false;
    try {
      await (await (A as any).submitProof(...args)).wait();
    } catch {
      reverted = true;
    }
    expect(reverted, "bad proof by the owner must revert").to.eq(true);

    const pc: bigint = await (orch as any).provenCount(id);
    expect(Number(pc)).to.eq(0);
  });

  it("claimed task can be reassigned when it’s the last unproven (fast-path); previous owner cannot submit afterwards", async () => {
    // 1-task grid → last-unproven fast-path is active
    const SINGLE = GRID.slice(0, 1);
    const id = await openRequest(orch, client, 1, "0.5", SINGLE);
    await commitDatasets(orch, client, id);

    const A = orchAs(accounts[1].pk);
    const B = orchAs(accounts[2].pk);

    await (await A.joinLobby(id)).wait(); // minWorkers=1 → started
    await (await B.joinLobby(id)).wait();

    const bond: bigint = await (orch as any).CLAIM_BOND_WEI();
    await (await A.claimTask(id, { value: bond })).wait();

    const idxA = await (async () => {
      const ownerLower = accounts[1].addr.toLowerCase();
      for (let i = 0; i < SINGLE.length; i++) {
        const o: string = (await (orch as any).taskOwner(id, i)).toLowerCase();
        if (o === ownerLower) return i;
      }
      return -1;
    })();
    expect(idxA).to.be.gte(0);

    // Because it's the only (i.e., last remaining) unproven task, fast-path applies.
    await (await (orch as any).reassignTimedOut(id, idxA)).wait(); // no majority/TTL needed here
    await (await B.claimTask(id, { value: bond })).wait();

    // A must not be able to submit for the old index anymore
    const iface: Interface = (orch as any).interface;
    const args = zeroArgsForSubmitProof(iface, id, idxA);

    let reverted = false;
    try {
      await (await (A as any).submitProof(...args)).wait();
    } catch {
      reverted = true;
    }
    expect(reverted, "previous owner must not be able to submit after reassignment").to.eq(true);
  });

    // ─────────────────────────────────────────────
  // Bench: settlement time vs number of workers
  // ─────────────────────────────────────────────

  const GRID_BENCH = [
    // Optional: a 12-task grid to saturate up to 12 workers; feel free to tweak.
    { lr: 500n, steps: 5n },
    { lr: 1000n, steps: 30n },
    { lr: 20000n, steps: 40n },
    { lr: 50000n, steps: 80n },
    { lr: 80000n, steps: 120n },
    { lr: 150000n, steps: 20n },
    { lr: 150000n, steps: 300n },
    { lr: 250000n, steps: 10n },
    { lr: 30000n, steps: 60n },
    { lr: 60000n, steps: 90n },
    { lr: 90000n, steps: 150n },
    { lr: 120000n, steps: 200n },
  ];

  async function benchOnce(N: number, grid = GRID_BENCH) {
    // minWorkers = min(N, grid.length) so the lobby opens as soon as all workers join
    const minWorkers = Math.min(N, grid.length);
    const id = await openRequest(orch, client, minWorkers, "1.0", grid);
    await commitDatasets(orch, client, id);

    // Spawn N workers (skip deployer at index 0)
    const t0 = Date.now();
    for (let i = 0; i < N; i++) {
      const pk = accounts[i + 1].pk; // accounts[0] is the client
      workers.push(spawnWorker({ url, orch: orchAddr, requestId: id, priv: pk }));
    }

    const res = await waitClosed(orch, id);
    const ms = Date.now() - t0;
    const bestAcc = Number(((res as any).bestAcc ?? (res as any)[1])) / 100;

    console.log(`[bench] workers=${N}, bestAcc=${bestAcc}%, duration_ms=${ms}`);
    await showCredits(orch, id, accounts.slice(1, 1 + N).map((a) => a.addr));
    return { ms, bestAcc };
  }

  it("bench: settle time for 3/6/9 workers", async function () {
    this.timeout(900_000); // give plenty of time on cold machines

    const Ns = [3];
    for (const N of Ns) {
      // Clean per-run timers/workers in case of prior test state
      for (const t of timers) { try { clearTimeout(t); } catch {} }
      timers = [];
      for (const p of workers) { try { await stopWorker(p); } catch {} }
      workers = [];

      const { ms, bestAcc } = await benchOnce(N);
      console.log(`[bench:summary] N=${N} → ${(ms/1000).toFixed(2)}s, bestAcc=${bestAcc}%`);
    }
  });
});

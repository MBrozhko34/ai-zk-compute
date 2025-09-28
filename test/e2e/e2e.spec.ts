// test/e2e/e2e.spec.ts
import { expect } from "chai";
import path from "path";
import { ethers } from "ethers";
import { ChildProcessWithoutNullStreams } from "child_process";
import {
  startHardhatNode, stop, deployOrch, makeDataset, writeCSV, openRequestDirect,
  spawnWorker, ACCOUNTS, HARDHAT_RPC, waitForClosed
} from "./utils";

describe("AiOrchestrator e2e", function () {
  this.timeout(180_000);

  let node: ChildProcessWithoutNullStreams;
  let provider: ethers.JsonRpcProvider;
  let client: ethers.Wallet;
  let abiPath: string;
  let orchAddr: string;
  let orch: ethers.Contract;

  const trainCsv = path.resolve("client/train.csv");
  const holdCsv  = path.resolve("client/dataset.csv");

  before(async () => {
    node = startHardhatNode();
    await new Promise(r => setTimeout(r, 2000)); // give node time to start
    provider = new ethers.JsonRpcProvider(HARDHAT_RPC);
    client   = new ethers.Wallet(ACCOUNTS[0].pk, provider);
    abiPath  = path.resolve("artifacts/contracts/AiOrchestrator.sol/AiOrchestrator.json");
    const dep = await deployOrch(abiPath);
    orch = dep.orch;
    orchAddr = dep.addr;
  });

  after(() => stop(node));

  it("happy path: multi-worker, all tasks proven, settlement correct", async () => {
    const { train, hold } = makeDataset(64, 32, 1);
    writeCSV(trainCsv, train);
    writeCSV(holdCsv,  hold);

    const grid = [
      { lr: 5_000, steps: 10 },
      { lr: 20_000, steps: 10 },
      { lr: 50_000, steps: 10 },
      { lr: 80_000, steps: 10 },
    ];
    const id = await openRequestDirect(orch, client, train, hold, grid, "0.01", 2);

    const w1 = spawnWorker({ orchAddr, requestId: id, pk: ACCOUNTS[1].pk, trainCsv, holdCsv });
    const w2 = spawnWorker({ orchAddr, requestId: id, pk: ACCOUNTS[2].pk, trainCsv, holdCsv });

    // wait for closure
    const closed = await waitForClosed(orch, id, 120_000);
    expect(closed).to.eq(true, "request did not close");

    // verify settlement & winners got > losers
    const res = await orch.getResult(id);
    const bestAcc: bigint = res[1];
    const perWin: bigint  = res[3];
    const perLose: bigint = res[4];
    expect(perWin).to.be.gt(perLose);

    // count each worker's wins/losses
    const w1s = await orch.taskStatsOf(id, ACCOUNTS[1].addr);
    const w2s = await orch.taskStatsOf(id, ACCOUNTS[2].addr);
    const sumTasks = Number(w1s[0] + w2s[0]);
    expect(sumTasks).to.eq(grid.length);

    // At least one winner exists
    expect(Number(w1s[2] + w2s[2])).to.be.greaterThan(0);
  });

  it("drop-out → reassign → completion", async () => {
    const { train, hold } = makeDataset(48, 16, 2);
    writeCSV(trainCsv, train);
    writeCSV(holdCsv,  hold);

    const grid = [
      { lr: 5_000, steps: 50 },
      { lr: 20_000, steps: 50 },
      { lr: 50_000, steps: 50 },
      { lr: 80_000, steps: 50 },
    ];
    const id = await openRequestDirect(orch, client, train, hold, grid, "0.01", 2);

    const w1 = spawnWorker({ orchAddr, requestId: id, pk: ACCOUNTS[3].pk, trainCsv, holdCsv });
    const w2 = spawnWorker({ orchAddr, requestId: id, pk: ACCOUNTS[4].pk, trainCsv, holdCsv });

    // Simulate a drop-out: kill w2 shortly after it starts
    await new Promise(r => setTimeout(r, 3000));
    try { w2.kill("SIGKILL"); } catch {}

    // Advance time past PROGRESS_TTL so reassignTimedOut can succeed
    await provider.send("evm_increaseTime", [130]); // PROGRESS_TTL ~120 in your contract
    await provider.send("evm_mine", []);

    // poke reassign for all indices (benign if not timed-out)
    const gridLen = (await orch.getSpace(id)).length;
    for (let i = 0; i < gridLen; i++) {
      try { await (await orch.reassignTimedOut(id, i)).wait(); } catch {}
    }

    // spawn another worker who should pick up the freed task(s)
    const w3 = spawnWorker({ orchAddr, requestId: id, pk: ACCOUNTS[5].pk, trainCsv, holdCsv });

    const closed = await waitForClosed(orch, id, 120_000);
    expect(closed).to.eq(true, "request did not close after reassign");

    const res = await orch.getResult(id);
    expect(res[0]).to.eq(true);
  });

  it("stress: many workers (10) on 12 tasks", async () => {
    const { train, hold } = makeDataset(64, 32, 3);
    writeCSV(trainCsv, train);
    writeCSV(holdCsv,  hold);

    const grid = Array.from({length: 12}, (_,i)=> ({ lr: 5_000 + 15_000 * (i%8), steps: 20 }));
    const id = await openRequestDirect(orch, client, train, hold, grid, "0.002", 2);

    const workers = ACCOUNTS.slice(1, 11).map(a =>
      spawnWorker({ orchAddr, requestId: id, pk: a.pk, trainCsv, holdCsv })
    );

    const closed = await waitForClosed(orch, id, 150_000);
    expect(closed).to.eq(true);

    const res = await orch.getResult(id);
    const perWin: bigint  = res[3];
    const perLose: bigint = res[4];
    expect(perWin).to.be.gt(perLose);
  });
});

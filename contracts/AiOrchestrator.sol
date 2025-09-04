// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

import "./Groth16Verifier.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract AiOrchestrator is ReentrancyGuard {
    /*────────────────── Types ──────────────────*/
    struct HyperParam { uint256 lr; uint256 steps; }

    struct Request {
        address   client;
        string    datasetCID;
        uint256   bountyWei;         // bounty pool (can grow via slashed bonds)
        HyperParam[] space;

        // lobby & lifecycle
        uint256   minWorkers;        // start threshold
        address[] lobby;             // all joiners (pre/post start)
        bool      started;           // true when lobby.length >= minWorkers
        bool      closed;            // set on settlement

        // results (after settlement)
        uint256   bestAcc;           // basis points
        uint256   perWinnerTaskWei;  // reward per winning task
        uint256   perLoserTaskWei;   // reward per losing task
    }

    /*────────────────── Constants ──────────────────*/
    Groth16Verifier public immutable V;

    // Per-claim bond (refunded on success, slashed on timeout)
    uint256 public constant CLAIM_BOND_WEI = 0.005 ether;
    // Claim timeout window (enable reassignment after majority proven)
    uint256 public constant CLAIM_TTL = 10; // seconds
    uint256 public constant STALL_TTL = 60; // seconds (choose to taste)

    // Winner/loser weights for per-task split
    uint256 private constant WIN_W = 3;
    uint256 private constant LOS_W = 1;

    /*────────────────── Storage ──────────────────*/
    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // Per-task assignment & state
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;     // req ⇒ idx ⇒ worker
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;       // req ⇒ idx ⇒ acc_bps (0 = not yet proven)
    mapping(uint256 => mapping(uint256 => uint256)) public claimedAt;     // req ⇒ idx ⇒ timestamp
    mapping(uint256 => mapping(uint256 => uint256)) public taskBondWei;   // req ⇒ idx ⇒ bond amount (if any)

    // Lobby registry (joiners may claim; join allowed before or after start)
    mapping(uint256 => mapping(address => bool)) public joined;           // req ⇒ addr ⇒ joined?

    // Pull-payments (rewards + bond refunds)
    mapping(uint256 => mapping(address => uint256)) public credit;        // req ⇒ addr ⇒ wei

    /*────────────────── Events ──────────────────*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 minWorkers, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joinedCount, uint256 minWorkers, bool started);
    event TaskClaimed(uint256 id, uint256 idx, address node, uint256 bondWei);
    event TaskReassigned(uint256 id, uint256 idx, address prevClaimer, uint256 slashedWei, uint256 newBountyWei);
    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnerTaskCount, uint256 perWinnerTaskWei, uint256 perLoserTaskWei);
    event BadProof(uint256 id, uint256 idx, address node, uint256 slashedWei);

    constructor(address verifier) { V = Groth16Verifier(verifier); }

    /*────────────────── Client API ──────────────────*/
    function openRequest(
        string calldata cid,
        HyperParam[] calldata grid,
        uint256 minWorkers
    ) external payable returns (uint256 id) {
        require(grid.length > 0, "grid empty");
        require(msg.value > 0, "bounty=0");
        require(minWorkers > 0 && minWorkers <= grid.length, "bad minWorkers");

        // Enforce uniqueness of (lr,steps) pairs
        for (uint256 i; i < grid.length; ++i) {
            for (uint256 j = i + 1; j < grid.length; ++j) {
                require(
                    !(grid[i].lr == grid[j].lr && grid[i].steps == grid[j].steps),
                    "duplicate hp"
                );
            }
        }

        id = nextId++;
        Request storage q = R[id];
        q.client     = msg.sender;
        q.datasetCID = cid;
        q.bountyWei  = msg.value;
        q.minWorkers = minWorkers;
        for (uint256 k; k < grid.length; ++k) q.space.push(grid[k]);

        emit RequestOpened(id, grid.length, minWorkers, msg.value);
    }

    /*────────────────── Lobby ──────────────────*/
    // Join allowed before or AFTER start. No stake at join; bond is per-claim.
    function joinLobby(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(!joined[id][msg.sender], "already joined");

        joined[id][msg.sender] = true;
        q.lobby.push(msg.sender);

        // start when threshold met (only once)
        if (!q.started && q.lobby.length >= q.minWorkers) {
            q.started = true;
        }
        emit LobbyJoined(id, msg.sender, q.lobby.length, q.minWorkers, q.started);
    }

    // For clients/workers UI: needed = minWorkers, joinedCount = lobby size, ready = started
    function lobbyCounts(uint256 id)
        external view
        returns (uint256 needed, uint256 joinedCount, bool ready)
    {
        Request storage q = R[id];
        return (q.minWorkers, q.lobby.length, q.started);
    }

    /*────────────────── Worker API ──────────────────*/
    // Pay a per-task bond and claim the next unassigned, unproven task.
    function claimTask(uint256 id) external payable nonReentrant returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "not started");
        require(joined[id][msg.sender], "not in lobby");
        require(msg.value == CLAIM_BOND_WEI, "bond");

        // find an unassigned & unproven task
        for (uint256 i; i < q.space.length; ++i) {
            if (taskOwner[id][i] == address(0) && taskAcc[id][i] == 0) {
                taskOwner[id][i]  = msg.sender;
                claimedAt[id][i]  = block.timestamp;
                taskBondWei[id][i] = msg.value;
                emit TaskClaimed(id, i, msg.sender, msg.value);
                return i;
            }
        }
        revert("no tasks left");
    }

    /*────────────── Worker API (patched) ─────────────*/
    // Allow anyone to free a timed-out claim (majority proven & TTL),
    // OR immediately if this is the *last* remaining unproven task.
    function reassignTimedOut(uint256 id, uint256 idx) external nonReentrant {
        Request storage q = R[id];
        require(q.started && !q.closed, "bad state");
        require(idx < q.space.length, "bad idx");
        require(taskAcc[id][idx] == 0, "already proven");
        address prev = taskOwner[id][idx];
        require(prev != address(0), "not claimed");

        bool lastUnproven = (_provenCount(id) == q.space.length - 1);
        if (!lastUnproven) {
            uint256 t = claimedAt[id][idx];
            // Normal path: majority proven + TTL
            if (_majorityProven(id)) {
                require(block.timestamp >= t + CLAIM_TTL, "not timed out");
            } else {
                // Stall path: allow after global stall timeout even without majority
                require(block.timestamp >= t + STALL_TTL, "stall TTL");
            }
        }

        uint256 b = taskBondWei[id][idx];
        if (b > 0) {
            q.bountyWei += b;
            taskBondWei[id][idx] = 0;
        }
        taskOwner[id][idx] = address(0);
        claimedAt[id][idx] = 0;

        emit TaskReassigned(id, idx, prev, b, q.bountyWei);
    }

    // Submit ZK proof for your claimed task. Refunds your bond and (if last) settles rewards.
    function submitProof(
        uint256 id,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[3] calldata pubSig   // [lr, steps, acc_bps]
    ) external nonReentrant {
        Request storage q = R[id];
        require(q.started, "not started");
        require(!q.closed, "closed");
        require(V.verifyProof(a, b, c, pubSig), "bad proof");

        // locate task idx by (lr,steps)
        uint256 idx; bool found;
        for (uint256 i; i < q.space.length; ++i) {
            if (pubSig[0] == q.space[i].lr && pubSig[1] == q.space[i].steps) { idx = i; found = true; break; }
        }
        require(found, "hp unknown");
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(taskAcc[id][idx] == 0, "already proven");

        taskAcc[id][idx] = pubSig[2];

        // refund bond into your credit
        uint256 bnd = taskBondWei[id][idx];
        if (bnd > 0) {
            credit[id][msg.sender] += bnd;
            taskBondWei[id][idx] = 0;
        }

        emit ProofAccepted(id, idx, msg.sender, pubSig[2]);

        // if all tasks proven → compute settlement once
        if (_allProven(id)) _computeSettlement(id);
    }

    function submitProofOrSlash(
        uint256 id,
        uint256 idx,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[3] calldata pubSig   // [lr, steps, acc_bps]
    ) external nonReentrant {
        Request storage q = R[id];
        require(q.started && !q.closed, "bad state");
        require(idx < q.space.length, "bad idx");
        require(taskOwner[id][idx] == msg.sender, "task not yours");

        // Require that pubSig matches the claimed index to prevent cross-index shenanigans
        require(
            pubSig[0] == q.space[idx].lr && pubSig[1] == q.space[idx].steps,
            "pubSig mismatch"
        );

        // If the proof is invalid: slash bond, free the claim, emit, and return
        if (!V.verifyProof(a, b, c, pubSig)) {
            uint256 bnd = taskBondWei[id][idx];
            if (bnd > 0) {
                q.bountyWei += bnd;
                taskBondWei[id][idx] = 0;
            }
            address prev = taskOwner[id][idx];
            taskOwner[id][idx] = address(0);
            claimedAt[id][idx] = 0;
            emit BadProof(id, idx, prev, bnd);
            return;
        }

        // Valid proof ⇒ accept (same as submitProof)
        require(taskAcc[id][idx] == 0, "already proven");
        taskAcc[id][idx] = pubSig[2];

        // refund bond
        uint256 bndOk = taskBondWei[id][idx];
        if (bndOk > 0) {
            credit[id][msg.sender] += bndOk;
            taskBondWei[id][idx] = 0;
        }

        emit ProofAccepted(id, idx, msg.sender, pubSig[2]);
        if (_allProven(id)) _computeSettlement(id);
    }

    /*────────────────── Internal helpers ──────────────────*/
    function _allProven(uint256 id) internal view returns (bool) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) if (taskAcc[id][i] == 0) return false;
        return true;
    }

    function _provenCount(uint256 id) internal view returns (uint256 n) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) if (taskAcc[id][i] != 0) n++;
        return n;
    }

    function _majorityProven(uint256 id) internal view returns (bool) {
        Request storage q = R[id];
        uint256 n = q.space.length;
        uint256 p = _provenCount(id);
        return (p * 2) > n;
    }

    // Per-task split: winners (acc == best) get WIN_W × unit, losers get LOS_W × unit
    // Credits are assigned per task, so multi-task workers earn the sum.
    function _computeSettlement(uint256 id) internal {
        Request storage q = R[id];
        require(!q.closed, "already closed");

        uint256 n = q.space.length;

        // best accuracy
        uint256 best;
        for (uint256 i; i < n; ++i) {
            uint256 a = taskAcc[id][i];
            if (a > best) best = a;
        }
        q.bestAcc = best;

        // count winner tasks vs loser tasks
        uint256 winTasks;
        uint256 loseTasks;
        for (uint256 i; i < n; ++i) {
            uint256 a = taskAcc[id][i];
            if (a == best) winTasks++;
            else loseTasks++;
        }

        // compute per-task amounts
        uint256 totalWeight = winTasks * WIN_W + loseTasks * LOS_W;
        // guard (shouldn't happen because all tasks proven → winTasks>0)
        if (totalWeight == 0) { q.closed = true; return; }

        uint256 unit = q.bountyWei / totalWeight;
        q.perWinnerTaskWei = unit * WIN_W;
        q.perLoserTaskWei  = unit * LOS_W;

        // assign credits per task
        uint256 distributed;
        for (uint256 i; i < n; ++i) {
            address w = taskOwner[id][i];
            if (w == address(0)) continue; // defensive
            bool isWin = (taskAcc[id][i] == best);
            uint256 amt = isWin ? q.perWinnerTaskWei : q.perLoserTaskWei;
            credit[id][w] += amt;
            distributed += amt;
        }

        // remainder wei → sprinkle over winning tasks to favor winners
        uint256 remainder = q.bountyWei - distributed;
        for (uint256 i; i < n && remainder > 0; ++i) {
            if (taskAcc[id][i] == best) {
                credit[id][taskOwner[id][i]] += 1;
                remainder -= 1;
            }
        }

        q.closed = true;
        emit RequestClosed(id, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }

    /*────────────────── Withdraw (pull payments) ──────────────────*/
    function withdraw(uint256 id) external nonReentrant {
        uint256 amt = credit[id][msg.sender];
        require(amt > 0, "no credit");
        credit[id][msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amt}("");
        require(ok, "transfer failed");
    }

    /*────────────────── Views ──────────────────*/
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }

    // NEW: public wrapper for internal _provenCount (for clients/agents)
    function provenCount(uint256 id) external view returns (uint256) {
        return _provenCount(id);
    }

    // Useful stats for UIs/clients
    function taskStatsOf(uint256 id, address who)
        external view
        returns (uint256 totalTasks, uint256 provenTasks, uint256 winTasks, uint256 loseTasks)
    {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) {
            if (taskOwner[id][i] == who) {
                totalTasks++;
                if (taskAcc[id][i] != 0) {
                    provenTasks++;
                    if (taskAcc[id][i] == q.bestAcc) winTasks++; else loseTasks++;
                }
            }
        }
    }

    // UPDATED: always count winner tasks, even when bestAcc == 0
    function getResult(uint256 id) external view returns (
        bool closed,
        uint256 bestAcc,
        uint256 winnerTaskCount,
        uint256 perWinnerTaskWei,
        uint256 perLoserTaskWei
    ) {
        Request storage q = R[id];
        uint256 winTasks;
        for (uint256 i; i < q.space.length; ++i) {
            if (taskAcc[id][i] == q.bestAcc) winTasks++;
        }
        return (q.closed, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }
}

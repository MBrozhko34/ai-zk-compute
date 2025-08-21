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
        uint256   bountyWei;     // total bounty supplied by client [+ any slashed stakes]
        HyperParam[] space;

        // lobby & lifecycle
        address[] lobby;         // join order (first N are "initial" workers)
        bool      started;       // set when lobby.size == grid.size
        bool      closed;        // set when settlement computed

        // results (after settlement)
        address[] winners;
        uint256   bestAcc;       // basis points
        uint256   perWinnerWei;  // credit per winner
        uint256   perLoserWei;   // credit per loser (participants only)
    }

    /*────────────────── Constants ──────────────────*/
    // Fixed stake required to join for a given request.
    uint256 public constant STAKE_WEI   = 0.01 ether;
    // Penalty on dropouts (in basis points, e.g., 5000 = 50%).
    uint256 public constant PENALTY_BPS = 5000; // 50%
    // Grace window (seconds) after majority of proofs is reached before reclaims allowed.
    uint256 public constant GRACE_SECS  = 10;

    // Weight split for reward pools (guarantees perWinner > perLoser when w,l>0)
    uint256 private constant WIN_W = 3;
    uint256 private constant LOS_W = 1;

    /*────────────────── Storage ──────────────────*/
    Groth16Verifier public immutable V;

    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // book-keeping
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;   // req ⇒ idx ⇒ worker
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;     // req ⇒ idx ⇒ acc_bps
    mapping(uint256 => mapping(address => bool))    public joined;      // req ⇒ addr (staked & registered)
    mapping(uint256 => mapping(address => uint256)) public joinIndex;   // req ⇒ addr ⇒ lobby pos (0..N-1 for initial)
    mapping(uint256 => mapping(address => uint256)) public stakeOf;     // req ⇒ addr ⇒ stake held (wei)

    // settlement helpers
    mapping(uint256 => uint256) public provenCount;    // req ⇒ # of proven tasks
    mapping(uint256 => uint256) public graceStartAt;   // req ⇒ timestamp when majority was first reached

    // pull-payments (rewards + stake refunds)
    mapping(uint256 => mapping(address => uint256)) public credit;      // req ⇒ addr ⇒ wei

    /*────────────────── Events ──────────────────*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joined, uint256 needed, bool started);
    event LobbyReady(uint256 id);
    event TaskClaimed(uint256 id, uint256 idx, address node);
    event TaskReassigned(uint256 id, uint256 idx, address oldWorker, address newWorker);
    event StakeDeposited(uint256 id, address node, uint256 amount);
    event StakeRefunded(uint256 id, address node, uint256 amount);
    event StakeSlashed(uint256 id, address node, uint256 amount);
    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnersCount, uint256 perWinnerWei, uint256 perLoserWei);

    constructor(address verifier) { V = Groth16Verifier(verifier); }

    /*────────────────── Client API ──────────────────*/
    function openRequest(string calldata cid, HyperParam[] calldata grid)
        external payable returns (uint256 id)
    {
        require(grid.length > 0, "grid empty");
        require(msg.value > 0, "bounty=0");
        id = nextId++;
        Request storage q = R[id];
        q.client     = msg.sender;
        q.datasetCID = cid;
        q.bountyWei  = msg.value;
        for (uint256 i; i < grid.length; ++i) q.space.push(grid[i]);
        emit RequestOpened(id, grid.length, msg.value);
    }

    /*────────────────── Lobby (stake + register) ──────────────────*/
    // NOTE: joining is allowed even AFTER start, to permit replacements to register.
    function joinLobby(uint256 id) external payable nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(msg.value == STAKE_WEI, "bad stake");

        // deposit stake
        require(stakeOf[id][msg.sender] == 0, "already staked");
        stakeOf[id][msg.sender] = msg.value;
        joined[id][msg.sender]  = true;
        emit StakeDeposited(id, msg.sender, msg.value);

        // append to lobby (even after start, for replacements)
        uint256 pos = q.lobby.length;
        q.lobby.push(msg.sender);

        // If not started and we just reached N = grid size, mark started
        uint256 needed = q.space.length;
        if (!q.started && q.lobby.length == needed) {
            q.started = true;
            emit LobbyReady(id);
        }

        emit LobbyJoined(id, msg.sender, q.lobby.length, needed, q.started);
    }

    function lobbyCounts(uint256 id) external view returns (uint256 needed, uint256 joinedCount, bool ready) {
        Request storage q = R[id];
        return (q.space.length, q.lobby.length, q.started);
    }

    /*────────────────── Worker API ──────────────────*/
    // Deterministic index derived from the FIRST N (initial) lobby order:
    // the first N addresses are mapped to indices [0..N-1] by their join order.
    // Replacements (joined after start) MUST use reclaimTask().
    function claimTask(uint256 id) external returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "not ready");
        require(joined[id][msg.sender], "not joined");

        uint256 n = q.space.length;

        // Determine if this address is one of the first N joiners.
        // If yes, its index is its lobby position [0..N-1].
        uint256 pos;
        bool found;
        for (uint256 i; i < q.lobby.length; ++i) {
            if (q.lobby[i] == msg.sender) { pos = i; found = true; break; }
        }
        require(found, "not in lobby"); // should always be true due to joined[]
        require(pos < n, "replacements must reclaim");

        idx = pos;

        // idempotent: set owner once, or check it matches
        if (taskOwner[id][idx] == address(0)) {
            taskOwner[id][idx] = msg.sender;
        } else {
            require(taskOwner[id][idx] == msg.sender, "index taken");
        }
        emit TaskClaimed(id, idx, msg.sender);
    }

    // Replacement: after grace, allow any joined/staked account to take over an unproven task.
    function reclaimTask(uint256 id, uint256 idx) external nonReentrant {
        Request storage q = R[id];
        require(q.started, "not started");
        require(!q.closed, "closed");
        require(joined[id][msg.sender], "not joined");
        require(stakeOf[id][msg.sender] >= STAKE_WEI, "no stake");
        require(idx < q.space.length, "bad idx");
        require(taskAcc[id][idx] == 0, "already proven");

        // Must wait until majority has submitted + grace has elapsed
        uint256 n = q.space.length;
        uint256 needMaj = (n + 1) / 2; // ceil(n/2)
        require(provenCount[id] >= needMaj, "no majority yet");
        uint256 startTs = graceStartAt[id];
        require(startTs != 0 && block.timestamp >= startTs + GRACE_SECS, "grace");

        address old = taskOwner[id][idx];
        if (old != address(0) && old != msg.sender) {
            // slash previous assignee (if they had stake)
            uint256 st = stakeOf[id][old];
            if (st > 0) {
                uint256 pen = (st * PENALTY_BPS) / 10_000;
                uint256 back = st - pen;
                stakeOf[id][old] = 0;
                // add penalty into bounty pot so winners benefit
                q.bountyWei += pen;
                // let the dropped worker withdraw remaining stake (pull model)
                credit[id][old] += back;
                emit StakeSlashed(id, old, pen);
            }
        }

        taskOwner[id][idx] = msg.sender;
        emit TaskReassigned(id, idx, old, msg.sender);
    }

    // On the last proof, settlement is computed automatically (no transfers here).
    function submitProof(
        uint256 id,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[3] calldata pubSignals   // [lr, steps, acc_bps]
    ) external nonReentrant {
        Request storage q = R[id];
        require(q.started, "not started");
        require(!q.closed, "closed");
        require(V.verifyProof(a,b,c,pubSignals), "bad proof");

        // locate idx by (lr, steps)
        uint256 idx; bool found;
        for (uint256 i; i < q.space.length; ++i) {
            if (pubSignals[0]==q.space[i].lr && pubSignals[1]==q.space[i].steps) { idx=i; found=true; break; }
        }
        require(found, "hp unknown");
        require(taskOwner[id][idx]==msg.sender, "task not yours");
        require(taskAcc[id][idx]==0, "already proven");

        taskAcc[id][idx] = pubSignals[2];
        emit ProofAccepted(id, idx, msg.sender, pubSignals[2]);

        // stake refund (pull) on successful proof
        uint256 st = stakeOf[id][msg.sender];
        if (st > 0) {
            stakeOf[id][msg.sender] = 0;
            credit[id][msg.sender] += st;
            emit StakeRefunded(id, msg.sender, st);
        }

        // majority tracking & grace start
        uint256 pc = ++provenCount[id];
        uint256 n = q.space.length;
        if (pc >= (n + 1)/2 && graceStartAt[id] == 0) {
            graceStartAt[id] = block.timestamp;
        }

        // If this was the last missing proof, compute the settlement now.
        if (_allProven(id)) {
            _computeSettlement(id);
        }
    }

    /*────────────────── Internal: settlement (no external calls) ──────────────────*/
    function _allProven(uint256 id) internal view returns (bool) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) {
            if (taskAcc[id][i] == 0) return false;
        }
        return true;
    }

    // Weight-based split: winners get WIN_W per head; losers get LOS_W per head.
    function _computeSettlement(uint256 id) internal {
        Request storage q = R[id];
        require(!q.closed, "already closed");

        uint256 n = q.space.length;

        // 1) best accuracy
        uint256 best;
        for (uint256 i; i < n; ++i) {
            uint256 a = taskAcc[id][i];
            if (a > best) best = a;
        }
        q.bestAcc = best;

        // 2) winners (ties allowed)
        delete q.winners;
        for (uint256 i; i < n; ++i) {
            if (taskAcc[id][i] == best) q.winners.push(taskOwner[id][i]);
        }
        uint256 w = q.winners.length;
        require(w > 0, "no winners");
        uint256 l = n - w; // all proven ⇒ losers = rest

        // 3) proportional split by weights
        uint256 pot = q.bountyWei;
        uint256 totalWeight = w * WIN_W + l * LOS_W; // >0 since w>0
        uint256 unit = pot / totalWeight;

        uint256 perWinner = unit * WIN_W;
        uint256 perLoser  = (l > 0) ? (unit * LOS_W) : 0;

        q.perWinnerWei = perWinner;
        q.perLoserWei  = perLoser;

        // 4) assign credits (rewards only; stake refunds already credited on success)
        uint256 distributed = 0;
        for (uint256 i; i < n; ++i) {
            address worker = taskOwner[id][i];
            if (worker == address(0)) continue;
            bool isWinner = (taskAcc[id][i] == best);
            uint256 amt = isWinner ? perWinner : perLoser;
            credit[id][worker] += amt;
            distributed += amt;
        }

        // 5) distribute leftover wei (if any) to winners to favor winners
        uint256 remainder = pot - distributed;
        uint256 winnersCount = q.winners.length;
        for (uint256 i; i < winnersCount && remainder > 0; ++i) {
            credit[id][q.winners[i]] += 1;
            remainder -= 1;
        }

        q.closed = true;
        emit RequestClosed(id, q.bestAcc, w, perWinner, perLoser);
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

    function taskOf(uint256 id, address node) external view returns (uint256 idx) {
        for (uint256 i; i < R[id].space.length; ++i) if (taskOwner[id][i]==node) return i;
        revert("none");
    }

    function getResult(uint256 id) external view returns (
        bool closed,
        uint256 bestAcc,
        uint256 winnersCount,
        uint256 perWinnerWei,
        uint256 perLoserWei
    ) {
        Request storage q = R[id];
        return (q.closed, q.bestAcc, q.winners.length, q.perWinnerWei, q.perLoserWei);
    }

    function winnerAt(uint256 id, uint256 i) external view returns (address) {
        return R[id].winners[i];
    }

    function getCredit(uint256 id, address who) external view returns (uint256) {
        return credit[id][who];
    }
}

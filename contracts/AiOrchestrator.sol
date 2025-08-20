// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

import "./Groth16Verifier.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract AiOrchestrator is ReentrancyGuard {
    /*────────────────── Data structures ──────────────────*/
    struct HyperParam { uint256 lr; uint256 steps; }

    struct Request {
        address   client;
        string    datasetCID;
        uint256   bountyWei;         // total bounty supplied by client
        HyperParam[] space;          // full HP grid

        // lobby & lifecycle
        address[] lobby;             // addresses in join order
        bool      started;           // becomes true when lobby size == grid size
        bool      closed;            // set on settlement

        // results
        address[] winners;           // winners after settlement (ties allowed)
        uint256   bestAcc;           // best accuracy in basis points
        uint256   perWinnerWei;      // payout per winner
    }

    Groth16Verifier public immutable V;

    // req ⇒ Request
    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // per-task book-keeping
    // req ⇒ idx ⇒ node assigned to that hyper-param
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;
    // req ⇒ idx ⇒ accuracy (0-10000) once proven
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;
    // req ⇒ addr ⇒ joined
    mapping(uint256 => mapping(address => bool)) public joined;
    // NEW: req ⇒ addr ⇒ lobby position (0-based)
    mapping(uint256 => mapping(address => uint256)) public joinIndex;

    /*── Events ──*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joined, uint256 needed);
    event LobbyReady(uint256 id);
    event TaskClaimed(uint256 id, uint256 idx, address node);
    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnersCount, uint256 perWinnerWei);

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

    /*────────────────── Lobby API ──────────────────*/
    function joinLobby(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(!q.started, "already started");
        require(!joined[id][msg.sender], "already joined");
        require(q.lobby.length < q.space.length, "lobby full");

        joined[id][msg.sender] = true;

        // record this account's 0-based position once, O(1)
        uint256 pos = q.lobby.length;
        joinIndex[id][msg.sender] = pos;

        q.lobby.push(msg.sender);
        emit LobbyJoined(id, msg.sender, q.lobby.length, q.space.length);

        // when the lobby fills, just flip the flag (keep this cheap!)
        if (q.lobby.length == q.space.length) {
            q.started = true;
            emit LobbyReady(id);
        }
    }

    function lobbyCounts(uint256 id)
        external view
        returns (uint256 needed, uint256 joinedCount, bool ready)
    {
        Request storage q = R[id];
        needed = q.space.length;
        joinedCount = q.lobby.length;
        ready = q.started;
    }

    /*────────────────── Worker API ──────────────────*/
    // Returns the unique index deterministically derived from lobby order.
    function claimTask(uint256 id) external returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "lobby not ready");
        require(joined[id][msg.sender], "not in lobby");

        // O(1): derive your index from stored lobby position
        idx = joinIndex[id][msg.sender];

        // one-time assignment (idempotent)
        if (taskOwner[id][idx] == address(0)) {
            taskOwner[id][idx] = msg.sender;
        } else {
            require(taskOwner[id][idx] == msg.sender, "index taken");
        }

        emit TaskClaimed(id, idx, msg.sender);
    }

    // Prove the claimed task. When the last task is proven, automatically settle.
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

        // locate task index by (lr,steps)
        uint256 idx; bool found;
        for (uint256 i; i < q.space.length; ++i) {
            if (pubSignals[0]==q.space[i].lr && pubSignals[1]==q.space[i].steps) {
                idx=i; found=true; break;
            }
        }
        require(found, "hp unknown");
        require(taskOwner[id][idx]==msg.sender, "task not yours");

        // also ensure the claimed idx matches lobby position deterministically
        require(joinIndex[id][msg.sender] == idx, "wrong index");

        require(taskAcc[id][idx]==0, "already proven");

        taskAcc[id][idx] = pubSignals[2];
        emit ProofAccepted(id, idx, msg.sender, pubSignals[2]);

        // if that was the last missing proof, auto-finalize and pay
        if (_allProven(id)) {
            _finalizeAndPay(id);
        }
    }

    /*────────────────── Settlement (internal) ──────────────────*/
    function _allProven(uint256 id) internal view returns (bool ok) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) {
            if (taskAcc[id][i] == 0) return false;
        }
        return true;
    }

    function _finalizeAndPay(uint256 id) internal {
        Request storage q = R[id];
        require(!q.closed, "already closed");

        // find best accuracy
        uint256 bestAcc;
        for (uint256 i; i < q.space.length; ++i) {
            uint256 a = taskAcc[id][i];
            if (a > bestAcc) bestAcc = a;
        }
        q.bestAcc = bestAcc;

        // collect all winners (ties)
        delete q.winners;
        for (uint256 i; i < q.space.length; ++i) {
            if (taskAcc[id][i] == bestAcc) {
                q.winners.push(taskOwner[id][i]);
            }
        }
        require(q.winners.length > 0, "no winners");

        // split bounty evenly among winners (draws share equally)
        uint256 per = q.bountyWei / q.winners.length;
        q.perWinnerWei = per;

        for (uint256 i; i < q.winners.length; ++i) {
            payable(q.winners[i]).transfer(per);
        }

        q.closed = true;
        emit RequestClosed(id, q.bestAcc, q.winners.length, per);
    }

    /*────────────────── Views ──────────────────*/
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }

    function taskOf(uint256 id, address node) external view returns (uint256 idx) {
        for (uint256 i; i<R[id].space.length; ++i) if (taskOwner[id][i]==node) return i;
        revert("none");
    }

    function getResult(uint256 id) external view returns (
        bool closed,
        uint256 bestAcc,
        uint256 winnersCount,
        uint256 perWinnerWei
    ) {
        Request storage q = R[id];
        return (q.closed, q.bestAcc, q.winners.length, q.perWinnerWei);
    }

    function winnerAt(uint256 id, uint256 i) external view returns (address) {
        return R[id].winners[i];
    }
}

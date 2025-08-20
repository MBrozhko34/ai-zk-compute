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
        uint256   bountyWei;       // total bounty supplied by client
        HyperParam[] space;        // full hp grid
        bool      closed;
    }

    Groth16Verifier public immutable V;

    // req ⇒ Request
    mapping(uint256 => Request) public R;
    uint256 public nextId;

    /* per‑task book‑keeping */
    // req ⇒ idx ⇒ node that claimed it
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;
    // req ⇒ idx ⇒ accuracy (0‑10000) once proven
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;

    /*── Events ──*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 bountyWei);
    event TaskClaimed(uint256 id, uint256 idx, address node);
    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event RequestClosed(uint256 id, address winner, uint256 winAmt, uint256 shareAmt);

    constructor(address verifier) { V = Groth16Verifier(verifier); }

    /*────────────────── Client API ──────────────────*/
    function openRequest(string calldata cid, HyperParam[] calldata grid) external payable returns (uint256 id) {
        require(grid.length > 0, "grid empty");
        id = nextId++;
        Request storage q = R[id];
        q.client     = msg.sender;
        q.datasetCID = cid;
        q.bountyWei  = msg.value;
        for (uint256 i; i < grid.length; ++i) q.space.push(grid[i]);
        emit RequestOpened(id, grid.length, msg.value);
    }

    /*────────────────── Worker API ──────────────────*/
    // 1️⃣ grab a unique task index
    function claimTask(uint256 id) external returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        for (uint256 i; i < q.space.length; ++i) {
            if (taskOwner[id][i] == address(0)) {
                taskOwner[id][i] = msg.sender;
                emit TaskClaimed(id,i,msg.sender);
                return i;
            }
        }
        revert("no tasks left");
    }

    // 2️⃣ prove the claimed task
    function submitProof(
        uint256 id,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[3] calldata pubSignals   // [lr, steps, acc_bps]
    ) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(V.verifyProof(a,b,c,pubSignals), "bad proof");

        // locate task index
        uint256 idx; bool found;
        for (uint256 i; i < q.space.length; ++i) {
            if (pubSignals[0]==q.space[i].lr && pubSignals[1]==q.space[i].steps) { idx=i; found=true; break; }
        }
        require(found, "hp unknown");
        require(taskOwner[id][idx]==msg.sender, "task not yours");
        require(taskAcc[id][idx]==0, "task already proven");

        taskAcc[id][idx] = pubSignals[2];
        emit ProofAccepted(id, idx, msg.sender, pubSignals[2]);
    }

    /*────────────────── Settlement ──────────────────*/
    function closeRequest(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "already closed");

        uint256 taskCount = q.space.length;
        uint256 participants;
        uint256 bestAcc; uint256 bestIdx;
        for(uint256 i; i<taskCount; ++i){
            if(taskAcc[id][i]>0){ // proven
                participants++;
                if(taskAcc[id][i]>bestAcc){ bestAcc=taskAcc[id][i]; bestIdx=i; }
            }
        }
        require(participants>0, "no proofs");

        address winner = taskOwner[id][bestIdx];
        uint256 pot = q.bountyWei;
        uint256 winAmt = pot / 2;                 // 50 %
        uint256 share = participants>1 ? (pot - winAmt)/(participants-1) : 0;

        // pay winner
        payable(winner).transfer(winAmt);
        // pay consolation
        for(uint256 i; i<taskCount; ++i) {
            address n = taskOwner[id][i];
            if(n!=address(0) && n!=winner) payable(n).transfer(share);
        }
        q.closed = true;
        emit RequestClosed(id, winner, winAmt, share);
    }

    /* view helpers */
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }
    function taskOf(uint256 id, address node) external view returns (uint256 idx) {
        for(uint256 i; i<R[id].space.length; ++i) if(taskOwner[id][i]==node) return i;
        revert("none");
    }
}
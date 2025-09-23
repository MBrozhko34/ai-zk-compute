// contracts/AiOrchestrator.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./Groth16Verifier.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * Level-2 “Track B+” hybrid:
 *   • Proof-of-inference (accuracy) via SNARK public acc_bps, and the contract recomputes
 *     hold-out accuracy on-chain from published final weights; equality is required (R3).
 *   • Proof-of-training (probabilistic): random step checks on a Merkle-committed transcript (R2).
 *
 * Circuit public signals (10):
 *   [0] lr, [1] steps, [2] acc_bps,
 *   [3] w0, [4] w1, [5] w2, [6] w3, [7] b0, [8] b1, [9] bO
 *
 * Requirements:
 *   R0  Same model/dataset: model & init rule are fixed in circuit; client sets the hold-out once.
 *   R1  Assigned hyper-parameters: enforced by (lr,steps) public signals and training checks.
 *   R2  Actually executed training: K random step checks over the transcript (Merkle + rule).
 *   R3  Don’t lie about accuracy: on-chain hold-out recompute must equal SNARK acc_bps.
 */
contract AiOrchestrator is ReentrancyGuard {
    struct HyperParam { uint256 lr; uint256 steps; }
    struct Request {
        address   client;
        string    datasetCID;
        uint256   bountyWei;
        HyperParam[] space;

        uint256   minWorkers;
        address[] lobby;
        bool      started;
        bool      closed;

        // Hold-out (R3)
        uint256[] hx0;
        uint256[] hx1;
        uint256[] hy;
        bool      holdoutSet;

        // Settlement
        uint256   bestAcc;           // bps
        uint256   perWinnerTaskWei;  // wei
        uint256   perLoserTaskWei;   // wei
    }

    Groth16Verifier public immutable V;

    uint256 public constant CLAIM_BOND_WEI = 0.005 ether;
    uint256 public constant CLAIM_TTL      = 10;
    uint256 public constant STALL_TTL      = 60;
    uint256 private constant WIN_W = 3;
    uint256 private constant LOS_W = 1;

    // Fixed XOR training set order per epoch
    uint256 private constant X00 = 0; uint256 private constant X01 = 0; uint256 private constant Y0 = 0;
    uint256 private constant X10 = 1; uint256 private constant X11 = 0; uint256 private constant Y1 = 1;
    uint256 private constant X20 = 0; uint256 private constant X21 = 1; uint256 private constant Y2 = 1;
    uint256 private constant X30 = 1; uint256 private constant X31 = 1; uint256 private constant Y3 = 0;

    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // per-task
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;
    mapping(uint256 => mapping(uint256 => uint256)) public claimedAt;
    mapping(uint256 => mapping(uint256 => uint256)) public taskBondWei;
    mapping(uint256 => mapping(address => bool))   public joined;
    mapping(uint256 => mapping(address => uint256)) public credit;

    // transcript + challenges
    mapping(uint256 => mapping(uint256 => bytes32)) public trRoot;
    mapping(uint256 => mapping(uint256 => uint256)) public trSteps;
    mapping(uint256 => mapping(uint256 => bytes32)) public trSeed;
    mapping(uint256 => mapping(uint256 => uint256)) public challengeK;
    mapping(uint256 => mapping(uint256 => uint256)) public answeredMask;

    event RequestOpened(uint256 id, uint256 taskCount, uint256 minWorkers, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joinedCount, uint256 minWorkers, bool started);
    event TaskClaimed(uint256 id, uint256 idx, address node, uint256 bondWei);
    event TaskReassigned(uint256 id, uint256 idx, address prevClaimer, uint256 slashedWei, uint256 newBountyWei);

    event TranscriptCommitted(uint256 id, uint256 idx, address worker, bytes32 root, uint256 totalSteps);
    event ChallengesFinalized(uint256 id, uint256 idx, bytes32 seed, uint256 k);
    event ChallengeAnswered(uint256 id, uint256 idx, uint256 i, uint256 stepIndex);

    event BadProof(uint256 id, uint256 idx, address node, uint256 slashedWei);
    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event OnChainAccChecked(uint256 id, uint256 idx, uint256 claimedAccBps, uint256 accOnChain, uint256 rows);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnerTaskCount, uint256 perWinnerTaskWei, uint256 perLoserTaskWei);

    constructor(address verifier) { V = Groth16Verifier(verifier); }

    /*──────────── Client API ────────────*/

    function openRequest(
        string calldata cid,
        HyperParam[] calldata grid,
        uint256 minWorkers
    ) external payable returns (uint256 id) {
        require(grid.length > 0, "grid empty");
        require(msg.value > 0, "bounty=0");
        require(minWorkers > 0 && minWorkers <= grid.length, "bad minWorkers");

        for (uint256 i; i < grid.length; ++i) {
            for (uint256 j = i + 1; j < grid.length; ++j) {
                require(!(grid[i].lr == grid[j].lr && grid[i].steps == grid[j].steps), "duplicate hp");
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

    function setHoldoutDataset(
        uint256 id,
        uint256[] calldata x0,
        uint256[] calldata x1,
        uint256[] calldata y
    ) external {
        Request storage q = R[id];
        require(msg.sender == q.client, "only client");
        require(!q.started, "already started");
        require(x0.length > 0 && x0.length == x1.length && x0.length == y.length, "bad dataset");
        for (uint256 i; i < x0.length; ++i) {
            require(x0[i] <= 1 && x1[i] <= 1 && y[i] <= 1, "dataset not binary");
        }
        q.hx0 = x0; q.hx1 = x1; q.hy = y; q.holdoutSet = true;
    }

    /*──────────── Lobby ────────────*/

    function joinLobby(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(!joined[id][msg.sender], "already joined");
        joined[id][msg.sender] = true;
        q.lobby.push(msg.sender);
        if (!q.started && q.lobby.length >= q.minWorkers) q.started = true;
        emit LobbyJoined(id, msg.sender, q.lobby.length, q.minWorkers, q.started);
    }

    function lobbyCounts(uint256 id)
        external view
        returns (uint256 needed, uint256 joinedCount, bool ready)
    {
        Request storage q = R[id];
        return (q.minWorkers, q.lobby.length, q.started);
    }

    /*──────────── Workers: claim & timeout ────────────*/

    function claimTask(uint256 id) external payable nonReentrant returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "not started");
        require(q.holdoutSet, "holdout not set");
        require(joined[id][msg.sender], "not in lobby");
        require(msg.value == CLAIM_BOND_WEI, "bond");

        for (uint256 i; i < q.space.length; ++i) {
            if (taskOwner[id][i] == address(0) && taskAcc[id][i] == 0) {
                taskOwner[id][i]   = msg.sender;
                claimedAt[id][i]   = block.timestamp;
                taskBondWei[id][i] = msg.value;
                emit TaskClaimed(id, i, msg.sender, msg.value);
                return i;
            }
        }
        revert("no tasks left");
    }

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
            if (_majorityProven(id)) {
                require(block.timestamp >= t + CLAIM_TTL, "not timed out");
            } else {
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

    /*──────────── Track B+: transcript + challenges ────────────*/

    function commitTranscript(uint256 id, uint256 idx, bytes32 root, uint256 totalSteps) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(root != bytes32(0) && totalSteps > 0, "bad transcript");
        require(trRoot[id][idx] == bytes32(0), "already committed");

        trRoot[id][idx]  = root;
        trSteps[id][idx] = totalSteps;
        claimedAt[id][idx] = block.timestamp;              // <── NEW
        emit TranscriptCommitted(id, idx, msg.sender, root, totalSteps);
    }

    function finalizeChallenges(uint256 id, uint256 idx, uint256 k) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(trRoot[id][idx] != bytes32(0), "no transcript");
        require(challengeK[id][idx] == 0, "already finalized");
        require(k > 0 && k <= 32, "k out of range");

        bytes32 seed = keccak256(
            abi.encodePacked(block.prevrandao, block.timestamp, msg.sender, id, idx, trRoot[id][idx], trSteps[id][idx])
        );
        trSeed[id][idx] = seed;
        challengeK[id][idx] = k;
        claimedAt[id][idx] = block.timestamp;              // <── NEW
        emit ChallengesFinalized(id, idx, seed, k);
    }

    function getChallenge(uint256 id, uint256 idx, uint256 i) public view returns (uint256 stepIndex) {
        require(challengeK[id][idx] != 0, "not finalized");
        require(i < challengeK[id][idx], "i>=K");
        uint256 total = trSteps[id][idx];
        bytes32 seed = trSeed[id][idx];
        stepIndex = uint256(keccak256(abi.encodePacked(seed, i))) % total;
    }

    function trainingChecksPassed(uint256 id, uint256 idx) public view returns (bool) {
        uint256 K = challengeK[id][idx];
        if (K == 0) return false;
        uint256 mask = answeredMask[id][idx];
        return mask == ((uint256(1) << K) - 1);
    }

    function _hashPair(bytes32 a, bytes32 b) internal pure returns (bytes32) {
        return a < b ? keccak256(abi.encodePacked(a, b)) : keccak256(abi.encodePacked(b, a));
    }
    function _verifySorted(bytes32[] calldata proof, bytes32 root, bytes32 leaf) internal pure returns (bool ok) {
        bytes32 h = leaf;
        for (uint256 i; i < proof.length; ++i) {
            h = _hashPair(h, proof[i]);
        }
        return h == root;
    }
    function _hashW(uint256[7] memory w) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(w[0],w[1],w[2],w[3],w[4],w[5],w[6]));
    }
    function _leafForStep(uint256 step, uint256[7] memory ws, uint256[7] memory wn) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(step, _hashW(ws), _hashW(wn)));
    }

    // Canonical update rule — Hidden0 Step>=1, Hidden1 Step>=2, bO floor=0.
    function _lrBucket(uint256 lr) internal pure returns (uint256 L) {
        L = 1; if (lr >= 50000) L += 1; if (lr >= 100000) L += 1;
    }
    function _applyOneStep(
        uint256[7] memory w, // [w0,w1,w2,w3,b0,b1,bO]
        uint256 x0, uint256 x1, uint256 y, uint256 L
    ) internal pure returns (uint256[7] memory r) {
        uint256 w0 = w[0]; uint256 w1 = w[1]; uint256 w2 = w[2]; uint256 w3 = w[3];
        uint256 b0 = w[4]; uint256 b1 = w[5]; uint256 bO = w[6];

        uint256 s0 = (w0 * x0 + w1 * x1 + b0 >= 1) ? 1 : 0;
        uint256 s1 = (w2 * x0 + w3 * x1 + b1 >= 2) ? 1 : 0;
        uint256 o  = (int256(int256(s0) - int256(s1) + int256(bO)) >= 1) ? 1 : 0;

        uint256 pos = (1 - o) * y;
        uint256 neg = (1 - y) * o;

        uint256 d0 = L * x0;
        uint256 d1 = L * x1;

        w0 = _satUpdate(w0, pos, neg, d0, 0);
        w1 = _satUpdate(w1, pos, neg, d1, 0);
        b0 = _satUpdate(b0, pos, neg, L,  0);

        w2 = _satUpdate(w2, neg, pos, d0, 0);
        w3 = _satUpdate(w3, neg, pos, d1, 0);
        b1 = _satUpdate(b1, neg, pos, L,  0);

        bO = _satUpdate(bO, pos, neg, L,  0);

        r[0]=w0; r[1]=w1; r[2]=w2; r[3]=w3; r[4]=b0; r[5]=b1; r[6]=bO;
    }
    function _satUpdate(uint256 x, uint256 inc, uint256 dec, uint256 delta, uint256 floor) internal pure returns (uint256 y) {
        uint256 xcap = x + inc * delta;
        if (xcap > 3) xcap = 3;
        uint256 sub = dec * delta;
        if (xcap < sub + floor) return floor;
        return xcap - sub;
    }

    function bindFinalWeights(
        uint256 id,
        uint256 idx,
        uint256 i,
        uint256[7] calldata wStart,
        uint256[7] calldata wEnd,
        bytes32[] calldata proof
    ) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        uint256 K = challengeK[id][idx];
        require(K != 0, "not finalized");
        require(i < K, "i>=K");
        uint256 mask = answeredMask[id][idx];
        uint256 bit  = (uint256(1) << i);
        require((mask & bit) == 0, "already answered");

        uint256 stepIndex = getChallenge(id, idx, i);
        bytes32 leaf = _leafForStep(stepIndex, wStart, wEnd);
        require(_verifySorted(proof, trRoot[id][idx], leaf), "bad merkle");

        (uint256 x0, uint256 x1, uint256 y) =
            (stepIndex % 4 == 0) ? (X00,X01,Y0) :
            (stepIndex % 4 == 1) ? (X10,X11,Y1) :
            (stepIndex % 4 == 2) ? (X20,X21,Y2) : (X30,X31,Y3);

        uint256 L = _lrBucket(R[id].space[idx].lr);
        uint256[7] memory next = _applyOneStep(wStart, x0, x1, y, L);
        for (uint256 t; t < 7; ++t) { require(next[t] == wEnd[t], "bad update"); }

        answeredMask[id][idx] = mask | bit;
        claimedAt[id][idx]    = block.timestamp;  
        emit ChallengeAnswered(id, idx, i, stepIndex);
    }

    /*──────────── SNARK + hold-out recheck ────────────*/

    function _toFixed10(uint256[10] calldata src) internal pure returns (uint256[10] memory out) {
        for (uint256 i; i < 10; ++i) out[i] = src[i];
    }

    function submitProof(
        uint256 id,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[10] calldata pubSig
    ) external nonReentrant {
        _submitProofCore(id, a, b, c, pubSig, false, 0);
    }

    function submitProofOrSlash(
        uint256 id,
        uint256 idx,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[10] calldata pubSig
    ) external nonReentrant {
        _submitProofCore(id, a, b, c, pubSig, true, idx);
    }

    function _submitProofCore(
        uint256 id,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[10] calldata pubSig,
        bool withIdx,
        uint256 idxParam
    ) internal {
        Request storage q = R[id];
        require(q.started && !q.closed, "bad state");
        require(q.holdoutSet, "holdout not set");

        uint256 idx; bool found;
        for (uint256 i; i < q.space.length; ++i) {
            if (pubSig[0] == q.space[i].lr && pubSig[1] == q.space[i].steps) { idx = i; found = true; break; }
        }
        require(found, "hp unknown");
        if (withIdx) require(idx == idxParam, "pubSig mismatch");
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(taskAcc[id][idx] == 0, "already proven");
        require(trainingChecksPassed(id, idx), "training checks incomplete");

        require(V.verifyProof(a, b, c, _toFixed10(pubSig)), "bad proof");

        uint256 accOnChain = _accFromWeightsOnHoldout(
            pubSig[3], pubSig[4], pubSig[5], pubSig[6], pubSig[7], pubSig[8], pubSig[9],
            q.hx0, q.hx1, q.hy
        );
        emit OnChainAccChecked(id, idx, pubSig[2], accOnChain, q.hx0.length);
        require(accOnChain == pubSig[2], "acc mismatch");

        taskAcc[id][idx] = pubSig[2];
        uint256 bnd = taskBondWei[id][idx];
        if (bnd > 0) { credit[id][msg.sender] += bnd; taskBondWei[id][idx] = 0; }

        emit ProofAccepted(id, idx, msg.sender, pubSig[2]);
        if (_allProven(id)) _computeSettlement(id);
    }

    function _accFromWeightsOnHoldout(
        uint256 w0, uint256 w1, uint256 w2, uint256 w3,
        uint256 b0, uint256 b1, uint256 bO,
        uint256[] storage x0,
        uint256[] storage x1,
        uint256[] storage y
    ) internal view returns (uint256 acc_bps) {
        uint256 n = x0.length;
        require(n > 0 && n == x1.length && n == y.length, "bad dataset len");
        uint256 correct;
        unchecked {
            for (uint256 i; i < n; ++i) {
                uint256 s0 = (w0 * x0[i] + w1 * x1[i] + b0 >= 1) ? 1 : 0;
                uint256 s1 = (w2 * x0[i] + w3 * x1[i] + b1 >= 2) ? 1 : 0;
                uint256 out = (int256(int256(s0) - int256(s1) + int256(bO)) >= 1) ? 1 : 0;
                if (out == y[i]) correct++;
            }
        }
        acc_bps = (correct * 10000) / n;
    }

    /*──────────── Helpers & settlement ────────────*/

    function _allProven(uint256 id) internal view returns (bool) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) if (taskAcc[id][i] == 0) return false;
        return true;
    }
    function _provenCount(uint256 id) internal view returns (uint256 n) {
        Request storage q = R[id];
        for (uint256 i; i < q.space.length; ++i) if (taskAcc[id][i] != 0) n++;
    }
    function _majorityProven(uint256 id) internal view returns (bool) {
        Request storage q = R[id];
        uint256 n = q.space.length;
        uint256 p = _provenCount(id);
        return (p * 2) > n;
    }

    function _computeSettlement(uint256 id) internal {
        Request storage q = R[id];
        require(!q.closed, "already closed");
        uint256 n = q.space.length;

        uint256 best;
        for (uint256 i; i < n; ++i) {
            uint256 a = taskAcc[id][i];
            if (a > best) best = a;
        }
        q.bestAcc = best;

        uint256 winTasks;
        uint256 loseTasks;
        for (uint256 i; i < n; ++i) {
            if (taskAcc[id][i] == best) winTasks++; else loseTasks++;
        }

        uint256 totalWeight = winTasks * WIN_W + loseTasks * LOS_W;
        if (totalWeight == 0) { q.closed = true; return; }

        uint256 unit = q.bountyWei / totalWeight;
        q.perWinnerTaskWei = unit * WIN_W;
        q.perLoserTaskWei  = unit * LOS_W;

        uint256 distributed;
        for (uint256 i; i < n; ++i) {
            address w = taskOwner[id][i];
            if (w == address(0)) continue;
            bool isWin = (taskAcc[id][i] == best);
            uint256 amt = isWin ? q.perWinnerTaskWei : q.perLoserTaskWei;
            credit[id][w] += amt;
            distributed += amt;
        }

        uint256 remainder = q.bountyWei - distributed;
        for (uint256 i; i < n && remainder > 0; ++i) {
            if (taskAcc[id][i] == best) { credit[id][taskOwner[id][i]] += 1; remainder -= 1; }
        }

        q.closed = true;
        emit RequestClosed(id, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }

    function withdraw(uint256 id) external nonReentrant {
        uint256 amt = credit[id][msg.sender];
        require(amt > 0, "no credit");
        credit[id][msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amt}("");
        require(ok, "transfer failed");
    }

    /*──────────── Views ────────────*/
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }
    function datasetLength(uint256 id) external view returns (uint256) { return R[id].hx0.length; }
    function provenCount(uint256 id) external view returns (uint256) { return _provenCount(id); }

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

    function getResult(uint256 id) external view returns (
        bool closed, uint256 bestAcc, uint256 winnerTaskCount, uint256 perWinnerTaskWei, uint256 perLoserTaskWei
    ) {
        Request storage q = R[id];
        uint256 winTasks;
        for (uint256 i; i < q.space.length; ++i) {
            if (taskAcc[id][i] == q.bestAcc) winTasks++;
        }
        return (q.closed, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }
}

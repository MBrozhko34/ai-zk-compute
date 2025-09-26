// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * Track B+ integrity with a real 2-layer neural model (integer MLP):
 *   - Architecture: 2 -> 4 -> 1 with ReLU gate at hidden and sign at output
 *   - Params: int16-like in [-127, +127], stored as int256 on-chain
 *   - Inputs: x0,x1 ∈ {0..15}, y ∈ {0,1}
 *   - Deterministic integer SGD-like update (documented in comments).
 *
 * R0: client commits Merkle root of training set (indexed leaves keccak(i,x0,x1,y)) and uploads hold-out set.
 * R1: hyper-params fixed by request.grid.
 * R2: worker commits Merkle root of training transcript and answers K random step checks.
 *     Each check re-runs exactly one canonical update with the verified sample.
 * R3: the hold-out accuracy is recomputed on-chain from the submitted final weights.
 */

contract AiOrchestrator is ReentrancyGuard {
    /*──────── Model & data constants ────────*/
    uint256 private constant NFEAT = 2;
    uint256 private constant H     = 4;
    int256  private constant CAP   = 127;  // weights/biases saturate in [-127, +127]
    uint256 private constant XMAX  = 15;   // features 0..15

    // Indexing of the flattened weight vector:
    //  W1: [0..7]  (H * NFEAT)  => j*2 + {0,1}
    //  B1: [8..11] (H)
    //  V : [12..15](H)
    //  b2: [16]
    uint256 private constant W_SIZE = 17;

    /*──────── Orchestrator economics & bookkeeping ────────*/
    uint256 public constant CLAIM_BOND_WEI = 0.005 ether;
    uint256 public constant CLAIM_TTL      = 10;
    uint256 public constant STALL_TTL      = 60;
    uint256 private constant WIN_W = 3;
    uint256 private constant LOS_W = 1;

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

        // R0: training set commitment (indexed leaves)
        bytes32   trainingRoot;
        uint256   trainingLen;

        // R3: holdout set
        uint256[] hx0;
        uint256[] hx1;
        uint256[] hy;
        bool      holdoutSet;

        // settlement
        uint256   bestAcc;
        uint256   perWinnerTaskWei;
        uint256   perLoserTaskWei;
    }

    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // per-task
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;     // acc_bps or 0
    mapping(uint256 => mapping(uint256 => uint256)) public claimedAt;
    mapping(uint256 => mapping(uint256 => uint256)) public taskBondWei;
    mapping(uint256 => mapping(address => bool))    public joined;
    mapping(uint256 => mapping(address => uint256)) public credit;

    // transcript & challenges
    mapping(uint256 => mapping(uint256 => bytes32)) public trRoot;
    mapping(uint256 => mapping(uint256 => uint256)) public trSteps;
    mapping(uint256 => mapping(uint256 => bytes32)) public trSeed;
    mapping(uint256 => mapping(uint256 => uint256)) public challengeK;
    mapping(uint256 => mapping(uint256 => uint256)) public answeredMask;

    /*──────── Events ────────*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 minWorkers, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joinedCount, uint256 minWorkers, bool started);
    event TaskClaimed(uint256 id, uint256 idx, address node, uint256 bondWei);
    event TaskReassigned(uint256 id, uint256 idx, address prevClaimer, uint256 slashedWei, uint256 newBountyWei);

    event TrainingRootSet(uint256 id, bytes32 root, uint256 len);
    event TranscriptCommitted(uint256 id, uint256 idx, address worker, bytes32 root, uint256 totalSteps);
    event ChallengesFinalized(uint256 id, uint256 idx, bytes32 seed, uint256 k);
    event ChallengeAnswered(uint256 id, uint256 idx, uint256 i, uint256 stepIndex);

    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event OnChainAccChecked(uint256 id, uint256 idx, uint256 claimedAccBps, uint256 accOnChain, uint256 rows);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnerTaskCount, uint256 perWinnerTaskWei, uint256 perLoserTaskWei);

    /*──────── Client API ────────*/
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

    function setTrainingDatasetRoot(uint256 id, bytes32 root, uint256 len) external {
        Request storage q = R[id];
        require(msg.sender == q.client, "only client");
        require(!q.started, "already started");
        require(root != bytes32(0) && len > 0, "bad training root");
        q.trainingRoot = root;
        q.trainingLen  = len;
        emit TrainingRootSet(id, root, len);
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
            require(x0[i] <= XMAX && x1[i] <= XMAX && (y[i] == 0 || y[i] == 1), "holdout out of range");
        }
        q.hx0 = x0; q.hx1 = x1; q.hy = y; q.holdoutSet = true;
    }

    /*──────── Lobby ────────*/
    function joinLobby(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(!joined[id][msg.sender], "already joined");
        joined[id][msg.sender] = true;
        q.lobby.push(msg.sender);
        if (!q.started && q.lobby.length >= q.minWorkers) q.started = true;
        emit LobbyJoined(id, msg.sender, q.lobby.length, q.minWorkers, q.started);
    }

    function lobbyCounts(uint256 id) external view returns (uint256 needed, uint256 joinedCount, bool ready) {
        Request storage q = R[id];
        return (q.minWorkers, q.lobby.length, q.started);
    }

    /*──────── Claim / timeout ────────*/
    function claimTask(uint256 id) external payable nonReentrant returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "not started");
        require(q.holdoutSet, "holdout not set");
        require(q.trainingLen > 0 && q.trainingRoot != bytes32(0), "train root not set");
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

    /*──────── Transcript + challenges (Track B+) ────────*/
    function commitTranscript(uint256 id, uint256 idx, bytes32 root, uint256 totalSteps) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(root != bytes32(0) && totalSteps > 0, "bad transcript");
        require(trRoot[id][idx] == bytes32(0), "already committed");
        trRoot[id][idx]  = root;
        trSteps[id][idx] = totalSteps;
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

    /*──────── Merkle helpers (sorted-pair) ────────*/
    function _hashPair(bytes32 a, bytes32 b) internal pure returns (bytes32) {
        return a < b ? keccak256(abi.encodePacked(a, b)) : keccak256(abi.encodePacked(b, a));
    }
    function _verifySorted(bytes32[] calldata proof, bytes32 root, bytes32 leaf) internal pure returns (bool ok) {
        bytes32 h = leaf;
        for (uint256 i; i < proof.length; ++i) h = _hashPair(h, proof[i]);
        return h == root;
    }

    function _hashW(int256[W_SIZE] memory w) internal pure returns (bytes32) {
        // encode all 17 ints into a packed hash
        return keccak256(abi.encodePacked(
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],
            w[8],w[9],w[10],w[11],
            w[12],w[13],w[14],w[15],w[16]
        ));
    }
    function _leafForStep(uint256 step, int256[W_SIZE] memory ws, int256[W_SIZE] memory we) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(step, _hashW(ws), _hashW(we)));
    }
    function _leafForSample(uint256 idx, uint256 x0, uint256 x1, uint256 y) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(idx, x0, x1, y));
    }

    /*──────── Integer math helpers ────────*/
    function _sgn(int256 v) internal pure returns (int256) { return v > 0 ? int256(1) : (v < 0 ? int256(-1) : int256(0)); }
    function _sat(int256 x) internal pure returns (int256) {
        if (x >  CAP) return CAP;
        if (x < -CAP) return -CAP;
        return x;
    }
    function _lrBucket(uint256 lr) internal pure returns (uint256 L) {
        L = 1; if (lr >= 50_000) L += 1; if (lr >= 100_000) L += 1;
    }

    /*──────── Canonical one-step update for the 2→4→1 MLP ────────*/
    function _applyOneStepMLP(
        int256[W_SIZE] memory w,
        uint256 x0,
        uint256 x1,
        uint256 y,
        uint256 L
    ) internal pure returns (int256[W_SIZE] memory r) {
        // unpack (views)
        int256[W_SIZE] memory a = w; // local copy/mutable
        // forward
        int256[4] memory pre;
        int256[4] memory s;

        for (uint256 j = 0; j < H; ++j) {
            uint256 base = j * NFEAT;
            pre[j] = a[base + 0] * int256(uint256(x0)) + a[base + 1] * int256(uint256(x1)) + a[8 + j];
            s[j]   = pre[j] >= 0 ? int256(1) : int256(0);
        }
        int256 z = a[16]; // b2
        for (uint256 j = 0; j < H; ++j) z += a[12 + j] * s[j];
        int256 p = (z >= 0) ? int256(1) : int256(0);
        int256 e = int256(uint256(y)) - p;  // -1, 0, +1

        if (e != 0) {
            // update V using s, but keep V_old for backprop sign into W1
            int256[4] memory Vold;
            for (uint256 j = 0; j < H; ++j) Vold[j] = a[12 + j];

            // second-layer
            for (uint256 j = 0; j < H; ++j) {
                int256 dv = e * int256(L) * s[j];
                a[12 + j] = _sat(a[12 + j] + dv);
            }
            a[16] = _sat(a[16] + e * int256(L));

            // first-layer (only if gate active)
            for (uint256 j = 0; j < H; ++j) {
                if (s[j] != 0) {
                    int256 sign_v = _sgn(Vold[j]);
                    int256 d = e * int256(L) * sign_v;
                    uint256 base = j * NFEAT;
                    a[base + 0] = _sat(a[base + 0] + d * int256(uint256(x0)));
                    a[base + 1] = _sat(a[base + 1] + d * int256(uint256(x1)));
                    a[8 + j]    = _sat(a[8 + j]    + d);
                }
            }
        }
        for (uint256 i = 0; i < W_SIZE; ++i) r[i] = a[i];
    }

    /*──────── Challenge answer ────────*/
    function bindFinalWeights(
        uint256 id,
        uint256 idx,
        uint256 i,
        int256[W_SIZE] calldata wStart,
        int256[W_SIZE] calldata wEnd,
        uint256 sampleIndex,
        uint256 x0, uint256 x1, uint256 y,
        bytes32[] calldata trProof,
        bytes32[] calldata sampleProof
    ) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        uint256 K = challengeK[id][idx];
        require(K != 0, "not finalized");
        require(i < K, "i>=K");
        uint256 mask = answeredMask[id][idx];
        uint256 bit  = (uint256(1) << i);
        require((mask & bit) == 0, "already answered");

        Request storage q = R[id];
        require(q.trainingLen > 0, "no training set");
        require(x0 <= XMAX && x1 <= XMAX && (y == 0 || y == 1), "sample out of range");

        // verify transcript leaf
        uint256 stepIndex = getChallenge(id, idx, i);
        bytes32 leafTr = _leafForStep(stepIndex, wStart, wEnd);
        require(_verifySorted(trProof, trRoot[id][idx], leafTr), "bad merkle transcript");

        // verify sample leaf and modulo index
        require(sampleIndex == (stepIndex % q.trainingLen), "wrong sample idx");
        bytes32 leafSamp = _leafForSample(sampleIndex, x0, x1, y);
        require(_verifySorted(sampleProof, q.trainingRoot, leafSamp), "bad merkle sample");

        // re-run update with LR bucket
        uint256 L = _lrBucket(q.space[idx].lr);
        int256[W_SIZE] memory next = _applyOneStepMLP(wStart, x0, x1, y, L);
        for (uint256 t; t < W_SIZE; ++t) require(next[t] == wEnd[t], "bad update");

        answeredMask[id][idx] = mask | bit;
        emit ChallengeAnswered(id, idx, i, stepIndex);
    }

    /*──────── Hold-out accuracy (R3) ────────*/
    function _predict(int256[W_SIZE] memory w, uint256 x0, uint256 x1) internal pure returns (uint256) {
        int256 z = w[16];
        for (uint256 j; j < H; ++j) {
            uint256 base = j * NFEAT;
            int256 pre = w[base + 0] * int256(uint256(x0)) + w[base + 1] * int256(uint256(x1)) + w[8 + j];
            int256 s   = pre >= 0 ? int256(1) : int256(0);
            z += w[12 + j] * s;
        }
        return (z >= 0) ? 1 : 0;
    }

    function _accOnHoldout(int256[W_SIZE] calldata w, uint256[] storage x0, uint256[] storage x1, uint256[] storage y)
        internal view returns (uint256 acc_bps)
    {
        uint256 n = x0.length;
        require(n > 0 && n == x1.length && n == y.length, "bad dataset len");
        uint256 correct;
        unchecked { for (uint256 i; i < n; ++i) if (_predict(w, x0[i], x1[i]) == y[i]) correct++; }
        return (correct * 10000) / n;
    }

    function submitResult(
        uint256 id,
        uint256 idx,
        int256[W_SIZE] calldata finalW,
        uint256 claimedAccBps
    ) external nonReentrant {
        Request storage q = R[id];
        require(q.started && !q.closed, "bad state");
        require(q.holdoutSet, "holdout not set");
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(taskAcc[id][idx] == 0, "already proven");
        require(trainingChecksPassed(id, idx), "training checks incomplete");

        uint256 accOnChain = _accOnHoldout(finalW, q.hx0, q.hx1, q.hy);
        emit OnChainAccChecked(id, idx, claimedAccBps, accOnChain, q.hx0.length);
        require(accOnChain == claimedAccBps, "acc mismatch");

        taskAcc[id][idx] = claimedAccBps;

        uint256 b = taskBondWei[id][idx];
        if (b > 0) { credit[id][msg.sender] += b; taskBondWei[id][idx] = 0; }

        emit ProofAccepted(id, idx, msg.sender, claimedAccBps);
        if (_allProven(id)) _computeSettlement(id);
    }

    /*──────── Settlement & helpers ────────*/
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
        for (uint256 i; i < n; ++i) { uint256 a = taskAcc[id][i]; if (a > best) best = a; }
        q.bestAcc = best;

        uint256 winTasks; uint256 loseTasks;
        for (uint256 i; i < n; ++i) { if (taskAcc[id][i] == best) winTasks++; else loseTasks++; }

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
            if (taskAcc[id][i] == q.bestAcc) { credit[id][taskOwner[id][i]] += 1; remainder -= 1; }
        }

        q.closed = true;
        emit RequestClosed(id, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }

    /*──────── Views ────────*/
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }
    function datasetLength(uint256 id) external view returns (uint256) { return R[id].hx0.length; }
    function provenCount(uint256 id) external view returns (uint256) { return _provenCount(id); }

    function taskStatsOf(uint256 id, address who)
        external view returns (uint256 totalTasks, uint256 provenTasks, uint256 winTasks, uint256 loseTasks)
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
        for (uint256 i; i < q.space.length; ++i) if (taskAcc[id][i] == q.bestAcc) winTasks++;
        return (q.closed, q.bestAcc, winTasks, q.perWinnerTaskWei, q.perLoserTaskWei);
    }
}

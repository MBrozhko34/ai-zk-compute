// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

// number of 32-entry limbs for 256 rows
uint256 constant LIMBS    = 256 / 32;                   // 8
uint256 constant N_PUBLIC = 1 + (2 * 17) + (4 * LIMBS);   // 67

interface IGroth16Verifier {
    function verifyProof(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[N_PUBLIC] memory input
    ) external view returns (bool);
}

/**
 * ZK‑ONLY orchestrator (Groth16):
 *   • Model: integer MLP 2 → 4 → 1, weights in [-127,127], inputs x ∈ [0..15], y ∈ {0,1}
 *   • R0: client commits Merkle root of training set (indexed leaves keccak(i,x0,x1,y))
 *   • R1: hyper‑params fixed by request.grid
 *   • R2: worker commits training transcript root and answers K random step checks
 *   • R3: hold‑out accuracy is proved **only** via Groth16 over K sampled rows
 *
 */
contract AiOrchestrator is ReentrancyGuard {
    /*──────── Model & data constants ────────*/
    uint256 private constant NFEAT = 2;
    uint256 private constant H     = 4;
    int256  private constant CAP   = 127;  // weights/biases saturate in [-127, +127]
    uint256 private constant XMAX  = 15;   // features 0..15

    // Flattened weight layout:
    //  W1: [0..7]  (H*NFEAT) => j*2 + {0,1}
    //  B1: [8..11] (H)
    //   V: [12..15](H)
    //  b2: [16]
    uint256 private constant W_SIZE = 17;

    // Number of hold‑out samples verified/used by zk circuit (masked to K if len<K)
    uint256 public constant HOLDOUT_K = 256;

    /*──────── ZK verifier (Groth16) for holdout accuracy ────────*/
    IGroth16Verifier public accVerifier;

    constructor(IGroth16Verifier _accVerifier) {
        require(address(_accVerifier) != address(0), "verifier=0");
        accVerifier = _accVerifier;
    }

    /*──────── Economics & bookkeeping ────────*/
    uint256 public constant CLAIM_BOND_WEI = 0.005 ether;
    uint256 public constant CLAIM_TTL      = 300;
    uint256 public constant STALL_TTL      = 900;
    uint256 public constant PROGRESS_TTL   = 120;
    uint256 private constant WIN_W         = 3;
    uint256 private constant LOS_W         = 1;

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

        // R3: zk root‑mode only
        bytes32   holdoutRoot;
        uint256   holdoutLen;
        bool      holdoutRootSet;

        // settlement
        uint256   bestAcc;
        uint256   perWinnerTaskWei;
        uint256   perLoserTaskWei;
    }

    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // per task
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
    event HoldoutRootSet(uint256 id, bytes32 root, uint256 len);

    event TranscriptCommitted(uint256 id, uint256 idx, address worker, bytes32 root, uint256 totalSteps);
    event ChallengesFinalized(uint256 id, uint256 idx, bytes32 seed, uint256 k);
    event ChallengeAnswered(uint256 id, uint256 idx, uint256 i, uint256 stepIndex);

    event ProofAccepted(uint256 id, uint256 idx, address node, uint256 acc);
    event RequestClosed(uint256 id, uint256 bestAcc, uint256 winnerTaskCount, uint256 perWinnerTaskWei, uint256 perLoserTaskWei);
    event CreditWithdrawn(uint256 indexed id, address indexed to, uint256 amount);

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

    // Training set root (indexed leaves keccak(i,x0,x1,y))
    function setTrainingDatasetRoot(uint256 id, bytes32 root, uint256 len) external {
        Request storage q = R[id];
        require(msg.sender == q.client, "only client");
        require(!q.started, "already started");
        require(root != bytes32(0) && len > 0, "bad training root");
        q.trainingRoot = root;
        q.trainingLen  = len;
        emit TrainingRootSet(id, root, len);
    }

    // Hold‑out root (zk‑only build accepts only root‑mode)
    function setHoldoutDatasetRoot(uint256 id, bytes32 root, uint256 len) external {
        Request storage q = R[id];
        require(msg.sender == q.client, "only client");
        require(!q.started, "already started");
        require(root != bytes32(0) && len > 0, "bad holdout root");
        q.holdoutRoot    = root;
        q.holdoutLen     = len;
        q.holdoutRootSet = true;
        emit HoldoutRootSet(id, root, len);
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
        require(q.holdoutRootSet && q.holdoutLen > 0, "holdout not set");
        require(q.trainingLen > 0 && q.trainingRoot != bytes32(0), "train root not set");
        require(joined[id][msg.sender], "not in lobby");
        require(msg.value == CLAIM_BOND_WEI, "bond");

        for (uint256 i; i < q.space.length; ++i) {
            if (taskOwner[id][i] == address(0) && taskAcc[id][i] == 0) {
                taskOwner[id][i]   = msg.sender;
                claimedAt[id][i]   = block.timestamp;
                taskBondWei[id][i] = msg.value;
                _touchProgress(id, i);
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

        uint256 base = _majorityProven(id) ? CLAIM_TTL : STALL_TTL;
        uint256 t0   = claimedAt[id][idx];
        uint256 lp   = lastProgressAt[id][idx];
        uint256 nowT = block.timestamp;

        bool hasProgress = (lp != 0 && lp > t0);
        bool timedOut = (nowT >= t0 + base) || (hasProgress && nowT >= lp + PROGRESS_TTL);
        require(timedOut, "not timed out");

        uint256 b = taskBondWei[id][idx];
        if (b > 0) {
            q.bountyWei += b;
            taskBondWei[id][idx] = 0;
        }
        taskOwner[id][idx]       = address(0);
        claimedAt[id][idx]       = 0;
        lastProgressAt[id][idx]  = 0;

        emit TaskReassigned(id, idx, prev, b, q.bountyWei);
    }

    mapping(uint256 => mapping(uint256 => uint256)) public lastProgressAt;
    function _touchProgress(uint256 id, uint256 idx) internal { lastProgressAt[id][idx] = block.timestamp; }

    /*──────── Merkle helpers (sorted‑pair) ────────*/
    function _hashPair(bytes32 a, bytes32 b) internal pure returns (bytes32) {
        return a < b ? keccak256(abi.encodePacked(a, b)) : keccak256(abi.encodePacked(b, a));
    }
    function _verifySorted(bytes32[] calldata proof, bytes32 root, bytes32 leaf) internal pure returns (bool ok) {
        bytes32 h = leaf;
        for (uint256 i; i < proof.length; ++i) h = _hashPair(h, proof[i]);
        return h == root;
    }
    function _verifySortedPacked(
        bytes32[] calldata proofs,
        uint256 off,
        uint256 count,
        bytes32 root,
        bytes32 leaf
    ) internal pure returns (bool ok) {
        bytes32 h = leaf;
        for (uint256 i; i < count; ++i) h = _hashPair(h, proofs[off + i]);
        return h == root;
    }

    function _hashW(int256[W_SIZE] memory w) internal pure returns (bytes32) {
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

    /*──────── Integer helpers ────────*/
    function _sgn(int256 v) internal pure returns (int256) { return v > 0 ? int256(1) : (v < 0 ? int256(-1) : int256(0)); }
    function _sat(int256 x) internal pure returns (int256) {
        if (x >  CAP) return CAP;
        if (x < -CAP) return -CAP;
        return x;
    }
    function _lrBucket(uint256 lr) internal pure returns (uint256 L) {
        if (lr < 20_000) return 1;
        if (lr < 40_000) return 2;
        if (lr < 60_000) return 3;
        if (lr < 80_000) return 4;
        if (lr < 100_000) return 5;
        if (lr < 120_000) return 6;
        if (lr < 140_000) return 7;
        return 8;
    }
    function _mapByteSigned(uint256 b, uint256 limit) internal pure returns (int256) {
        unchecked { return int256(b % (2*limit + 1)) - int256(limit); }
    }

    function _detInitW(uint256 id, uint256 idx) internal view returns (int256[W_SIZE] memory w) {
        bytes32 seed = keccak256(abi.encodePacked("init/", id, idx, R[id].trainingRoot));
        uint256 counter = 0;
        bytes32 buf = keccak256(abi.encodePacked(seed, counter));
        uint256 pos = 0;
        for (uint256 i = 0; i < W_SIZE; ++i) {
            if (pos == 32) { unchecked { counter++; } buf = keccak256(abi.encodePacked(seed, counter)); pos = 0; }
            uint8 b = uint8(buf[pos]); unchecked { pos++; }
            if (i < 8)       w[i] = _mapByteSigned(b, 3);
            else if (i < 12) w[i] = _mapByteSigned(b, 6);
            else if (i < 16) w[i] = _mapByteSigned(b, 2);
            else             w[i] = _mapByteSigned(b, 2);
        }
    }

    /*──────── Canonical one‑step update ────────*/
    function _applyOneStepMLP(
        int256[W_SIZE] memory a,
        uint256 x0,
        uint256 x1,
        uint256 y,
        uint256 L
    ) internal pure returns (int256[W_SIZE] memory r) {
        int256[4] memory s;
        for (uint256 j; j < H; ++j) {
            uint256 base = j * NFEAT;
            int256 pre = a[base + 0] * int256(uint256(x0)) + a[base + 1] * int256(uint256(x1)) + a[8 + j];
            s[j]       = pre >= 0 ? int256(1) : int256(0);
        }
        int256 z = a[16];
        for (uint256 j; j < H; ++j) z += a[12 + j] * s[j];
        int256 p = (z >= 0) ? int256(1) : int256(0);
        int256 e = int256(uint256(y)) - p;

        if (e != 0) {
            int256[4] memory Vold;
            for (uint256 j; j < H; ++j) Vold[j] = a[12 + j];
            for (uint256 j; j < H; ++j) a[12 + j] = _sat(a[12 + j] + e * int256(L) * s[j]);
            a[16] = _sat(a[16] + e * int256(L));
            for (uint256 j; j < H; ++j) if (s[j] != 0) {
                int256 d = e * int256(L) * _sgn(Vold[j]);
                uint256 base = j * NFEAT;
                a[base + 0] = _sat(a[base + 0] + d * int256(uint256(x0)));
                a[base + 1] = _sat(a[base + 1] + d * int256(uint256(x1)));
                a[8 + j]    = _sat(a[8 + j]    + d);
            }
        }
        for (uint256 i; i < W_SIZE; ++i) r[i] = a[i];
    }

    /*──────── Transcript + challenges ────────*/
    function commitTranscript(uint256 id, uint256 idx, bytes32 root, uint256 totalSteps) external {
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(root != bytes32(0) && totalSteps > 0, "bad transcript");
        require(trRoot[id][idx] == bytes32(0), "already committed");
        trRoot[id][idx]  = root;
        trSteps[id][idx] = totalSteps;
        _touchProgress(id, idx);
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
        _touchProgress(id, idx);
        emit ChallengesFinalized(id, idx, seed, k);
    }

    function getChallenge(uint256 id, uint256 idx, uint256 i) public view returns (uint256 stepIndex) {
        require(challengeK[id][idx] != 0, "not finalized");
        require(i < challengeK[id][idx], "i>=K");
        uint256 total = trSteps[id][idx];
        require(total > 0, "no transcript");

        if (i == 0) return 0;
        if (total == 1) return 0;

        bytes32 seed = trSeed[id][idx];
        return 1 + (uint256(keccak256(abi.encodePacked(seed, i))) % (total - 1));
    }

    function getHoldoutChallenge(uint256 id, uint256 idx, uint256 i) public view returns (uint256 sampleIndex) {
        require(challengeK[id][idx] != 0, "not finalized");
        Request storage q = R[id];
        require(q.holdoutLen > 0, "no holdout");
        bytes32 seed = trSeed[id][idx];
        sampleIndex = uint256(keccak256(abi.encodePacked(seed, "holdout", i))) % q.holdoutLen;
    }

    function trainingChecksPassed(uint256 id, uint256 idx) public view returns (bool) {
        uint256 K = challengeK[id][idx];
        if (K == 0) return false;
        uint256 mask = answeredMask[id][idx];
        return mask == ((uint256(1) << K) - 1);
    }

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

        uint256 stepIndex = getChallenge(id, idx, i);
        bytes32 leafTr = _leafForStep(stepIndex, wStart, wEnd);
        require(_verifySorted(trProof, trRoot[id][idx], leafTr), "bad merkle transcript");

        if (stepIndex == 0) {
            int256[W_SIZE] memory want = _detInitW(id, idx);
            for (uint256 t; t < W_SIZE; ++t) require(wStart[t] == want[t], "bad init");
        }

        require(sampleIndex == (stepIndex % q.trainingLen), "wrong sample idx");
        bytes32 leafSamp = _leafForSample(sampleIndex, x0, x1, y);
        require(_verifySorted(sampleProof, q.trainingRoot, leafSamp), "bad merkle sample");

        uint256 L = _lrBucket(R[id].space[idx].lr);
        int256[W_SIZE] memory next = _applyOneStepMLP(wStart, x0, x1, y, L);
        for (uint256 t; t < W_SIZE; ++t) require(next[t] == wEnd[t], "bad update");

        answeredMask[id][idx] = mask | bit;
        _touchProgress(id, idx);
        emit ChallengeAnswered(id, idx, i, stepIndex);
    }

    /*──────── zk submit ────────*/
    function submitResultZK(
        uint256 id,
        uint256 idx,
        int256[W_SIZE] calldata finalW,
        uint256 claimedAccBps,

        // Fixed-size vectors of length HOLDOUT_K
        uint256[] calldata sampleIdx,
        uint256[] calldata mask,          // 0/1; must be 1 for i<K and 0 for i>=K
        uint256[] calldata x0,
        uint256[] calldata x1,
        uint256[] calldata y,

        // concatenated Merkle proofs, sizes per i
        bytes32[] calldata proofsConcat,
        uint256[] calldata proofSizes,

        // Groth16 proof
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c
    ) external nonReentrant {
        Request storage q = R[id];
        require(q.started && !q.closed, "bad state");
        require(q.holdoutRootSet && q.holdoutLen > 0, "no holdout root");
        require(taskOwner[id][idx] == msg.sender, "task not yours");
        require(taskAcc[id][idx] == 0, "already proven");
        require(trainingChecksPassed(id, idx), "training checks incomplete");

        uint256 K = HOLDOUT_K;
        if (K > q.holdoutLen) K = q.holdoutLen;

        require(sampleIdx.length == HOLDOUT_K, "bad sampleIdx len");
        require(mask.length      == HOLDOUT_K, "bad mask len");
        require(x0.length        == HOLDOUT_K && x1.length == HOLDOUT_K && y.length == HOLDOUT_K, "bad xyz len");
        require(proofSizes.length == HOLDOUT_K, "bad sizes len");

        // Verify mask + Merkle + ranges for active rows
        uint256 off;
        for (uint256 i; i < HOLDOUT_K; ++i) {
            uint256 m = mask[i];
            require(m == 0 || m == 1, "mask 0/1");
            bool shouldActive = (i < K);
            if (shouldActive) {
                require(m == 1, "mask missing");
                uint256 want = getHoldoutChallenge(id, idx, i);
                require(sampleIdx[i] == want, "wrong sample idx");
                require(x0[i] <= XMAX && x1[i] <= XMAX && (y[i] == 0 || y[i] == 1), "sample out of range");

                bytes32 leaf = _leafForSample(want, x0[i], x1[i], y[i]);
                uint256 sz = proofSizes[i];
                require(off + sz <= proofsConcat.length, "bad proofs span");
                require(_verifySortedPacked(proofsConcat, off, sz, q.holdoutRoot, leaf), "bad merkle sample");
                off += sz;
            } else {
                require(m == 0, "mask extra");
                require(x0[i] <= XMAX && x1[i] <= XMAX && (y[i] == 0 || y[i] == 1), "masked out of range");
            }
        }

        // Build public input for Groth16 verifier (packed form):
        // [ acc_bps,
        //   w_abs[17], w_sign[17],
        //   mask_p[8], x0_p[8], x1_p[8], y_p[8] ]

        // Circuit order: [ acc_bps, w_abs[17], w_sign[17],
        //                  mask_p[8], x0_p[8], x1_p[8], y_p[8] ]
        uint256[N_PUBLIC] memory input;
        uint256 p = 0;

        // 0) accuracy
        input[p++] = claimedAccBps;

        // 1) |w| and 2) sign bits
        for (uint256 t; t < W_SIZE; ++t) {
            int256 v = finalW[t];
            bool neg = v < 0;
            uint256 a0 = uint256(neg ? -v : v);
            require(a0 <= uint256(int256(CAP)), "w out of range");
            input[p++] = a0;
        }
        for (uint256 t2; t2 < W_SIZE; ++t2) {
            input[p++] = finalW[t2] < 0 ? 1 : 0;
        }

        // 3) mask_p[8], 4) x0_p[8], 5) x1_p[8], 6) y_p[8]
        uint256[LIMBS] memory M;
        uint256[LIMBS] memory X0;
        uint256[LIMBS] memory X1;
        uint256[LIMBS] memory Y;

        for (uint256 g; g < LIMBS; ++g) {
            uint256 base = g * 32;
            uint256 mLimb; uint256 x0Limb; uint256 x1Limb; uint256 yLimb;
            for (uint256 k; k < 32; ++k) {
                uint256 i = base + k;
                mLimb  |= (mask[i] & 1) << k;       // bits (LSB-first)
                yLimb  |= (y[i]    & 1) << k;       // bits (LSB-first)
                x0Limb |= (x0[i] & 15)  << (4*k);   // 4-bit nibbles (LSB-first)
                x1Limb |= (x1[i] & 15)  << (4*k);
            }
            M[g]  = mLimb;
            X0[g] = x0Limb;
            X1[g] = x1Limb;
            Y[g]  = yLimb;
        }

        // append in circuit order (do NOT interleave by group)
        for (uint256 g; g < LIMBS; ++g) input[p++] = M[g];
        for (uint256 g; g < LIMBS; ++g) input[p++] = X0[g];
        for (uint256 g; g < LIMBS; ++g) input[p++] = X1[g];
        for (uint256 g; g < LIMBS; ++g) input[p++] = Y[g];

        // Call the verifier with fixed-size public input
        bool ok = accVerifier.verifyProof(a, b, c, input);
        require(ok, "bad zk proof");

        taskAcc[id][idx] = claimedAccBps;
        _touchProgress(id, idx);

        uint256 bnd = taskBondWei[id][idx];
        if (bnd > 0) { credit[id][msg.sender] += bnd; taskBondWei[id][idx] = 0; }

        emit ProofAccepted(id, idx, msg.sender, claimedAccBps);
        if (_allProven(id)) _computeSettlement(id);
    }

    /*──────── Settlement & views ────────*/
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

    function withdraw(uint256 id) external nonReentrant {
        uint256 amt = credit[id][msg.sender];
        require(amt > 0, "no credit");
        credit[id][msg.sender] = 0;
        (bool ok, ) = payable(msg.sender).call{value: amt}("");
        require(ok, "xfer");
        emit CreditWithdrawn(id, msg.sender, amt);
    }

    /* Views */
    function getSpace(uint256 id) external view returns (HyperParam[] memory) { return R[id].space; }
    function datasetLength(uint256 id) external view returns (uint256) { return R[id].holdoutLen; }
    function provenCount(uint256 id) external view returns (uint256 n) { return _provenCount(id); }

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

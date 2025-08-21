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
        uint256   bountyWei;
        HyperParam[] space;

        // lobby & lifecycle
        address[] lobby;    // join order
        bool      started;  // set when lobby.size == grid.size
        bool      closed;   // set when settlement computed

        // results (after settlement)
        address[] winners;
        uint256   bestAcc;        // basis points
        uint256   perWinnerWei;   // credit per winner
        uint256   perLoserWei;    // credit per loser (participants only)
    }

    /*────────────────── Storage ──────────────────*/
    Groth16Verifier public immutable V;

    mapping(uint256 => Request) public R;
    uint256 public nextId;

    // book-keeping
    mapping(uint256 => mapping(uint256 => address)) public taskOwner;   // req ⇒ idx ⇒ worker
    mapping(uint256 => mapping(uint256 => uint256)) public taskAcc;     // req ⇒ idx ⇒ acc_bps
    mapping(uint256 => mapping(address => bool))    public joined;      // req ⇒ addr
    mapping(uint256 => mapping(address => uint256)) public joinIndex;   // req ⇒ addr ⇒ lobby pos

    // pull-payments
    mapping(uint256 => mapping(address => uint256)) public credit;      // req ⇒ addr ⇒ wei

    // ─── Payout weights (winner gets WIN_W × loser) ───
    uint256 private constant WIN_W = 3;
    uint256 private constant LOS_W = 1;

    /*────────────────── Events ──────────────────*/
    event RequestOpened(uint256 id, uint256 taskCount, uint256 bountyWei);
    event LobbyJoined(uint256 id, address node, uint256 joined, uint256 needed);
    event LobbyReady(uint256 id);
    event TaskClaimed(uint256 id, uint256 idx, address node);
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

    /*────────────────── Lobby ──────────────────*/
    function joinLobby(uint256 id) external nonReentrant {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(!q.started, "started");
        require(!joined[id][msg.sender], "already joined");
        require(q.lobby.length < q.space.length, "lobby full");

        joined[id][msg.sender] = true;
        uint256 pos = q.lobby.length;
        joinIndex[id][msg.sender] = pos;
        q.lobby.push(msg.sender);
        emit LobbyJoined(id, msg.sender, q.lobby.length, q.space.length);

        if (q.lobby.length == q.space.length) {
            q.started = true;
            emit LobbyReady(id);
        }
    }

    function lobbyCounts(uint256 id) external view returns (uint256 needed, uint256 joinedCount, bool ready) {
        Request storage q = R[id];
        return (q.space.length, q.lobby.length, q.started);
    }

    /*────────────────── Worker API ──────────────────*/
    // deterministic index derived from lobby order
    function claimTask(uint256 id) external returns (uint256 idx) {
        Request storage q = R[id];
        require(!q.closed, "closed");
        require(q.started, "not ready");
        require(joined[id][msg.sender], "not in lobby");
        idx = joinIndex[id][msg.sender];
        if (taskOwner[id][idx] == address(0)) {
            taskOwner[id][idx] = msg.sender;
        } else {
            require(taskOwner[id][idx] == msg.sender, "index taken");
        }
        emit TaskClaimed(id, idx, msg.sender);
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
        require(joinIndex[id][msg.sender] == idx, "wrong index");
        require(taskAcc[id][idx]==0, "already proven");

        taskAcc[id][idx] = pubSignals[2];
        emit ProofAccepted(id, idx, msg.sender, pubSignals[2]);

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

    // Weight-based split: each winner has WIN_W weight, loser has LOS_W weight.
    // Guarantees perWinner > perLoser for any w,l > 0.
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

        // 4) assign credits
        uint256 distributed = 0;
        for (uint256 i; i < n; ++i) {
            address worker = taskOwner[id][i];
            if (worker == address(0)) continue;
            bool isWinner = (taskAcc[id][i] == best);
            uint256 amt = isWinner ? perWinner : perLoser;
            credit[id][worker] += amt;
            distributed += amt;
        }

        // 5) distribute remainder wei (if any) to winners to favor winners
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

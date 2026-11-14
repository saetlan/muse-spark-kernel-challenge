"""Simulation of UD4's push to row owners and per-owner finalize (cuda_ud4.py steps 8-9, make_ud4.py), with the source's
formulas asserted present: row hr's partial from CTA `rank` goes to owner hr % CS at slot (rank * 16 + hr) * LBP + k
of the owner's buffer (CS x 16 x LBP floats in every CTA); owner `rank` finalizes rows rank + CS * w for warps
w < 16 / CS. Checked on random warp partials: every remote store in bounds of its owner's buffer, each (rank, hr, k)
slot of each owned row written exactly once and no slot of a row the CTA does not own, every row finalized exactly
once across the cluster, the owner reads only written slots, and the merged rows equal a direct merge of all warps of
all ranks. Mutants (owner hr / ROWS, finalize row rank * ROWS + w, slot stride 64) must fail.
Run: .venv/bin/python attention/local/sim_ud4_merge.py"""
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "cuda_ud4.py")).read(), ns)
SRC = ns["CUDA_UD4_SRC"]
LBP = int(re.search(r"constexpr int LBP = (\d+);", SRC).group(1))
for frag in ("st_cluster(map_rank(slot0 + (hr * LBP + d) * 4, static_cast<uint32_t>(hr % CS)), o);",
             "const uint32_t slot0 = smem_u32(lbuf) + rank * (16 * LBP * 4);",
             "const uint32_t owner = static_cast<uint32_t>(threadIdx.x % CS);",
             "const int hr = static_cast<int>(rank) + CS * warp;", "if (warp >= ROWS) return;",
             "constexpr int ROWS = 16 / CS;", "(kv0 * 8 + static_cast<int>(rank) + CS * warp) * 64"):
    assert SRC.count(frag) == 1, f"formula not in cuda_ud4.py: {frag}"


def run(CS, WARPS, mut=None):
    rnd = random.Random(CS * 100 + WARPS)
    ROWS = 16 // CS
    size = CS * 16 * LBP
    buf = [[None] * size for _ in range(CS)]
    writes = [[0] * size for _ in range(CS)]
    problems = []
    P = [[dict(m=[[rnd.uniform(-5, 5) for _ in range(8)] for _ in range(2)],
               l=[[rnd.uniform(0.1, 3) for _ in range(8)] for _ in range(2)],
               o=[[[rnd.uniform(-1, 1) for _ in range(64)] for _ in range(8)] for _ in range(2)])
          for _ in range(WARPS)] for _ in range(CS)]
    owner_of = (lambda hr: hr // ROWS) if mut == "owner" else (lambda hr: hr % CS)
    stride = 64 if mut == "stride" else LBP
    for rank in range(CS):
        W = P[rank]
        sm_w, sm_mm, sm_lt = [0.0] * (16 * WARPS), [0.0] * 16, [0.0] * 16
        for tid in range(16):
            h, row = tid >> 3, tid & 7
            mm = max([-1.0e30] + [W[w]["m"][h][row] for w in range(WARPS)])
            lt = 0.0
            for w in range(WARPS):
                wt = 2.0 ** (W[w]["m"][h][row] - mm)
                sm_w[tid * WARPS + w] = wt
                lt += wt * W[w]["l"][h][row]
            sm_mm[tid], sm_lt[tid] = mm, lt
        slot0 = rank * (16 * stride)

        def store(own, a, v):
            if not 0 <= a < size:
                problems.append(f"store out of bounds: rank {rank} -> owner {own} at {a}")
                return
            buf[own][a] = v
            writes[own][a] += 1
        for t in range(WARPS * 32):
            for idx in range(t, 16 * 64, WARPS * 32):
                hr, d = idx >> 6, idx & 63
                h, row = hr >> 3, hr & 7
                o = sum(sm_w[hr * WARPS + w] * W[w]["o"][h][row][d] for w in range(WARPS))
                store(owner_of(hr), slot0 + hr * stride + d, o)
            if t < 16:
                store(owner_of(t), slot0 + t * stride + 64, sm_mm[t])
                store(owner_of(t), slot0 + t * stride + 65, sm_lt[t])
    finalized = {}
    for rank in range(CS):                           # step 9: owner rank, warps w < ROWS
        assert ROWS <= WARPS
        for w in range(ROWS):
            hr = rank * ROWS + w if mut == "fin_row" else rank + CS * w
            rows = [(r * 16 + hr) * LBP for r in range(CS)]
            for p in rows:
                for k in range(66):
                    if writes[rank][p + k] != 1:
                        problems.append(f"owner {rank} reads slot {p + k} (row {hr}) written {writes[rank][p + k]} times")
                        break
            vals = [[0.0 if x is None else x for x in buf[rank][p:p + 66]] for p in rows]
            mm = max([-1.0e30] + [v[64] for v in vals])
            l_, o = 0.0, [0.0] * 64
            for v in vals:
                wt = 2.0 ** (v[64] - mm)
                l_ += wt * v[65]
                for d in range(64):
                    o[d] += wt * v[d]
            if hr in finalized:
                problems.append(f"row {hr} finalized twice")
            finalized[hr] = [x / l_ if l_ else float("nan") for x in o]
    if sorted(finalized) != list(range(16)):
        problems.append(f"rows finalized: {sorted(finalized)}")
    for own in range(CS):                            # no slot of a row the CTA does not own
        for r in range(CS):
            for hr in range(16):
                if hr % CS != own and any(writes[own][(r * 16 + hr) * LBP + k] for k in range(66)):
                    problems.append(f"owner {own} got row {hr} it does not own")
                    break
    for hr, got in finalized.items():
        h, row = hr >> 3, hr & 7
        M = max(P[r][w]["m"][h][row] for r in range(CS) for w in range(WARPS))
        L = sum(2.0 ** (P[r][w]["m"][h][row] - M) * P[r][w]["l"][h][row] for r in range(CS) for w in range(WARPS))
        for d in range(64):
            want = sum(2.0 ** (P[r][w]["m"][h][row] - M) * P[r][w]["o"][h][row][d] for r in range(CS) for w in range(WARPS)) / L
            if not math.isclose(got[d], want, rel_tol=1e-9, abs_tol=1e-12):
                problems.append(f"merge mismatch row {hr} d {d}")
                break
    return problems


bad = 0
for CS in (8, 16):
    for WARPS in (4, 8):
        probs = run(CS, WARPS)
        print(("ok   " if not probs else "FAIL ") + f"CS {CS} WARPS {WARPS}: {probs[:2]}")
        bad += bool(probs)
for mut in ("owner", "fin_row", "stride"):
    probs = run(8, 4, mut)
    print(("caught   " if probs else "SURVIVED ") + f"mutant {mut}")
    bad += not probs
seq = [m.group(0) for m in re.finditer(r"barrier\.cluster\.(arrive|wait)[a-z.]*|cluster_sync\(\);", SRC.split("template <int WARPS")[1])]
ok = seq[:3] == ["barrier.cluster.arrive.relaxed.aligned", "barrier.cluster.wait.aligned", "cluster_sync();"]
print(("ok   " if ok else "FAIL ") + f"cluster barrier order: {seq}")
bad += not ok
print("sim_ud4_merge:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

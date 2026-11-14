"""Simulation of UD3's CTA partial, push and leader merge (cuda_ud3.py steps 7-9), parsed index
formulas checked by execution on random warp partials: every remote store lands inside the
leader buffer (CS x 16 x LBP floats), each slot (rank, row, 0..65) is written exactly once, the
leader reads only written slots, and the merged result equals a direct merge of all warps of all
ranks (exact softmax-merge algebra). Also checks the source's buffer size against the host's
dynamic shared memory and setup limits, and that the cluster barrier sequence is arrive / wait /
arrive+wait in that order. Mutants (wrong stride, wrong row) must fail.
Run: .venv/bin/python attention/local/sim_ud3_merge.py"""
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "cuda_ud3.py")).read(), ns)
SRC = ns["CUDA_UD3_SRC"]
LBP = int(re.search(r"constexpr int LBP = (\d+);", SRC).group(1))


def ex2(x):
    return 2.0 ** x


def run(CS, WARPS, mut=None):
    """Execute the kernel's steps 7-9 for one cluster on random partials; return (merged, problems)."""
    rnd = random.Random(CS * 100 + WARPS)
    size = CS * 16 * LBP
    lbuf = [None] * size
    writes = [0] * size
    problems = []
    # per rank, per warp: m[h][row], l[h][row], o[h][row][d]
    P = [[dict(m=[[rnd.uniform(-5, 5) for _ in range(8)] for _ in range(2)],
               l=[[rnd.uniform(0.1, 3) for _ in range(8)] for _ in range(2)],
               o=[[[rnd.uniform(-1, 1) for _ in range(64)] for _ in range(8)] for _ in range(2)])
          for _ in range(WARPS)] for _ in range(CS)]
    for rank in range(CS):
        W = P[rank]
        sm_w, sm_mm, sm_lt = [0.0] * (16 * WARPS), [0.0] * 16, [0.0] * 16
        for tid in range(16):                        # step 7: (h, row) = (tid >> 3, tid & 7)
            h, row = tid >> 3, tid & 7
            mm = max([-1.0e30] + [W[w]["m"][h][row] for w in range(WARPS)])
            lt = 0.0
            for w in range(WARPS):
                wt = ex2(W[w]["m"][h][row] - mm)
                sm_w[tid * WARPS + w] = wt
                lt += wt * W[w]["l"][h][row]
            sm_mm[tid], sm_lt[tid] = mm, lt
        lead = rank * (16 * LBP) if mut != "stride" else rank * (16 * 64)
        for t in range(WARPS * 32):                  # step 8: the push loop, every thread
            for idx in range(t, 16 * 64, WARPS * 32):
                hr, d = idx >> 6, idx & 63
                h, row = hr >> 3, hr & 7
                o = sum(sm_w[hr * WARPS + w] * W[w]["o"][h][row][d] for w in range(WARPS))
                a = lead + hr * LBP + d
                if not 0 <= a < size:
                    problems.append(f"store out of bounds: rank {rank} idx {idx} -> {a}")
                    continue
                lbuf[a] = o
                writes[a] += 1
            if t < 16:
                for k, v in ((64, sm_mm[t]), (65, sm_lt[t])):
                    a = lead + t * LBP + k
                    if not 0 <= a < size:
                        problems.append(f"store out of bounds: rank {rank} m/l {t} -> {a}")
                        continue
                    lbuf[a] = v
                    writes[a] += 1
    for r in range(CS):
        for hr in range(16):
            for k in range(66):
                c = writes[(r * 16 + hr) * LBP + k]
                if c != 1:
                    problems.append(f"slot ({r}, {hr}, {k}) written {c} times")
    if any(lbuf[(r * 16 + hr) * LBP + k] is None for r in range(CS) for hr in range(16) for k in range(66)):
        problems.append("the leader buffer has unwritten slots")
        lbuf = [float("nan") if x is None else x for x in lbuf]
    merged = {}
    for warp in range(WARPS):                       # step 9: leader
        for k2 in range(16 // WARPS):
            hr = warp + k2 * WARPS
            rows = [(r * 16 + hr) * LBP for r in range(CS)]
            if mut == "row":
                rows = [(r * 16 + (hr ^ 1)) * LBP for r in range(CS)]
            mm = max([-1.0e30] + [lbuf[p + 64] for p in rows])
            l_, o = 0.0, [0.0] * 64
            for p in rows:
                wt = ex2(lbuf[p + 64] - mm)
                l_ += wt * lbuf[p + 65]
                for lane in range(32):
                    o[lane] += wt * lbuf[p + lane]
                    o[32 + lane] += wt * lbuf[p + 32 + lane]
            merged[hr] = [x / l_ for x in o]
    # direct merge over every warp of every rank
    for hr in range(16):
        h, row = hr >> 3, hr & 7
        allm = [P[r][w]["m"][h][row] for r in range(CS) for w in range(WARPS)]
        M = max(allm)
        L = sum(ex2(P[r][w]["m"][h][row] - M) * P[r][w]["l"][h][row] for r in range(CS) for w in range(WARPS))
        for d in range(64):
            want = sum(ex2(P[r][w]["m"][h][row] - M) * P[r][w]["o"][h][row][d] for r in range(CS) for w in range(WARPS)) / L
            if not math.isclose(merged[hr][d], want, rel_tol=1e-9, abs_tol=1e-12):
                problems.append(f"merge mismatch row {hr} d {d}: {merged[hr][d]} vs {want}")
                break
    return problems


bad = 0
for CS in (8, 16):
    for WARPS in (4, 8):
        probs = run(CS, WARPS)
        print(("ok   " if not probs else "FAIL ") + f"CS {CS} WARPS {WARPS}: {probs[:2]}")
        bad += bool(probs)
for mut in ("stride", "row"):
    probs = run(8, 4, mut)
    print(("caught   " if probs else "SURVIVED ") + f"mutant {mut}")
    bad += not probs
# host smem == kernel layout: ring + CS * 16 * LBP floats, setup limits likewise
host = re.search(r"const size_t smem = (.*?);\n", SRC).group(1)
ok = "stages * okc_ud3::STAGE_BYTES" in host and "cs) * 16 * okc_ud3::LBP * 4" in host and \
    "W * S * STAGE_BYTES + C * 16 * LBP * 4" in SRC and "dsm + WARPS * STAGES * STAGE_BYTES" in SRC
print(("ok   " if ok else "FAIL ") + "host dynamic smem / setup limit = ring + leader buffer")
bad += not ok
for w, s, c in ((4, 3, 8), (8, 2, 8), (4, 3, 16), (8, 2, 16)):
    total = w * s * 2 * 16 * 144 + c * 16 * LBP * 4
    ok = total <= 227 * 1024
    print(("ok   " if ok else "FAIL ") + f"W{w} S{s} CS{c}: {total} B dynamic smem")
    bad += not ok
seq = [m.group(0) for m in re.finditer(r"barrier\.cluster\.(arrive|wait)[a-z.]*|cluster_sync\(\);", SRC.split("template <int WARPS")[1])]
ok = seq[:3] == ["barrier.cluster.arrive.relaxed.aligned", "barrier.cluster.wait.aligned", "cluster_sync();"]
print(("ok   " if ok else "FAIL ") + f"cluster barrier order in the kernel: {seq}")
bad += not ok
print("sim_ud3_merge:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

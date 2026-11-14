"""Checks W2's wgmma issue / commit / wait order (make_w2.py consumer loop: FA3's intra-warpgroup order) with a model of wgmma commit groups:
wait_group N completes every group except the newest N. The control flow below is a transcription of the kernel's
(stage 0's Q.K, per stage Q.K(i) + P.V(i-1) / wait 1 / softmax / wait 0, the last P.V; PP turn barriers); planted mistakes must be caught.
Asserts, for nst = 1 .. 9: no wgmma pending across iterations (FA3's order; L27: windows across the loop back-edge
left ptxas serializing); the P tile holds P(i-1) when P.V(i-1) is issued; PP turn barriers: syncs = arrives on each id; Q.K(i) complete before softmax(i) reads its buffer; the other buffer is not in flight
when Q.K(i+1) is issued into it; P.V(i-1) complete before O reads opv, before P(i) overwrites the P tile and before
stage i-1 is released; every stage released exactly once, after its P.V; nothing in flight at the end.
Run: .venv/bin/python attention/local/sim_w2_pipe.py"""
import sys


def run(nst, mut=None):
    groups = []                      # FIFO of (kind, stage, buffer), oldest first
    done = set()                     # completed (kind, stage)
    released = []
    buf_inflight = {}                # buffer -> stage of the in-flight Q.K writing it
    errs = []

    def issue(kind, j, buf=None):
        if kind == "qk":
            if buf in buf_inflight:
                errs.append(f"Q.K({j}) issued into {buf} while Q.K({buf_inflight[buf]}) is in flight")
            buf_inflight[buf] = j
        if kind == "pv" and any(g[0] == "pv" for g in groups):
            errs.append(f"P.V({j}) issued while an earlier P.V is in flight (P tile / opv reuse)")
        groups.append((kind, j, buf))

    def wait(n):
        while len(groups) > n:
            kind, j, buf = groups.pop(0)
            done.add((kind, j))
            if kind == "qk":
                buf_inflight.pop(buf, None)

    def need(kind, j, what):
        if (kind, j) not in done:
            errs.append(f"{what}: {kind}({j}) not complete")

    turns = {3: [0, 0], 4: [0, 0]}          # barrier id -> [syncs, arrives] (PP model, per warpgroup pair)

    def pp_counts(h, k, phases):
        if mut == "pp_extra_pass" or h == 0 or k + 1 < phases:
            turns[4 - h][1] += 1
        turns[3 + h][0] += 1

    if nst > 0:
        phases = nst + 1
        for h in (0, 1):                      # turn barrier bookkeeping for both warpgroups
            if h == 1:
                turns[3][1] += 1               # warpgroup 1's start-up hand-over
            for k in range(phases):
                pp_counts(h, k, phases)
        # stage 0
        issue("qk", 0, "s0")
        wait(0 if mut != "wait1_at_0" else 1)
        need("qk", 0, "softmax(0) reads s0")
        p_tile = 0                             # P tile holds P(0)
        for i in range(1, nst):
            if groups:
                errs.append(f"iteration {i} entered with pending {[g[:2] for g in groups]}")
            issue("qk", i, "sc")
            if p_tile != i - 1:
                errs.append(f"P.V({i - 1}) issued with P({p_tile}) in the tile")
            issue("pv", i - 1)
            wait(1 if mut != "wait2" else 2)
            need("qk", i, f"softmax({i}) reads sc")
            wait(0 if mut != "no_wait0" else 1)
            need("pv", i - 1, f"O update / release({i - 1}) / P({i}) store")
            released.append(i - 1)
            p_tile = i
        if p_tile != nst - 1:
            errs.append(f"last P.V with P({p_tile}) in the tile")
        issue("pv", nst - 1)
        wait(0 if mut != "no_final_wait" else 1)
        need("pv", nst - 1, "final O update")
        released.append(nst - 1)
        for bid, (sy, ar) in turns.items():
            if sy != ar:
                errs.append(f"turn barrier {bid}: {sy} syncs vs {ar} arrives")
    if groups:
        errs.append(f"in flight at exit: {groups}")
    if sorted(released) != list(range(nst)):
        errs.append(f"released {released}")
    return errs


bad = 0
for nst in range(1, 10):
    e = run(nst)
    if e:
        print(f"FAIL nst {nst}: {e[:3]}")
        bad += 1
print(f"{'ok  ' if not bad else 'FAIL'} W2 order for nst 1..9")
for mut in ("wait1_at_0", "wait2", "no_wait0", "no_final_wait", "pp_extra_pass"):
    caught = any(run(nst, mut) for nst in range(1, 10))
    print(("caught  " if caught else "MISSED  ") + f"mutant {mut}")
    bad += not caught
print("sim_w2_pipe:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

"""Checks W3's consumer order (make_w3.py) with a model of wgmma commit groups (wait_group N completes every group except
the newest N), transcribed from the generated kernel's control flow: pair `par` of NP takes global stages gst(i) =
NP i + par, i < nloc = (nst - par + NP - 1) / NP; stage 0: Q.K, wait 0, softmax, P -> tile; stage i >= 1: Q.K(i),
P.V(i-1) INTO O, wait 1, softmax(i), wait 0, O *= alpha(i), release(i-1), P(i) -> tile; last: P.V into O, wait 0,
release. Ping-pong turn barriers 5 + 2 par + h (bar.sync by h on its own id, bar.arrive on the partner's).
Asserts for nst = 0 .. 12 and NP = 1, 2: every global stage is consumed by exactly one pair and released once, after
its P.V; O is never read or written by the threads while a P.V into O is in flight; O is at stage i's running max when
P.V(i) is issued (rescaled by alpha(i) after P.V(i-1) completed, never twice); no wgmma pending across iterations;
the P tile holds P(i-1) when P.V(i-1) is issued; per barrier id, syncs = arrives; barrier ids distinct per pair and
not 1 .. NWG (P-tile barriers). Planted mistakes must be caught.
Run: .venv/bin/python attention/local/sim_w3_pipe.py"""
import sys


def run(nst, NP, mut=None):
    errs = []
    released = []
    turns = {}

    def tb(bid, kind):
        turns.setdefault(bid, [0, 0])[kind] += 1

    for par in range(NP):
        nloc = (nst - par + NP - 1) // NP
        gst = (lambda i, par=par: NP * i + par) if mut != "parity" else (lambda i: i)
        phases = nloc + 1
        for h in (0, 1):
            if nloc <= 0:
                continue
            if h == 1:
                tb(5 + 2 * par, 1)
            for k in range(phases):
                tb(5 + 2 * par + h, 0)
                if h == 0 or k + 1 < phases or mut == "pp_extra":
                    tb((6 + 2 * par - h) if mut != "pp_id" else (5 + 2 * par - h), 1)
        if nloc <= 0:
            continue
        groups = []
        o_scale = None          # the stage whose running max O is at (None: O = 0)
        o_has = []              # stages whose P.V is in O
        p_tile = None

        def issue(kind, j):
            if kind == "pv":
                if any(g[0] == "pv" for g in groups):
                    errs.append("two P.V in flight on O")
                if o_has and o_scale != j:      # O holds earlier P.V: it must sit at stage j's max
                    errs.append(f"P.V({j}) issued with O at stage {o_scale}'s max")
                if p_tile != j:
                    errs.append(f"P.V({j}) issued with P({p_tile}) in the tile")
            groups.append((kind, j))

        def wait(n):
            while len(groups) > n:
                kind, j = groups.pop(0)
                if kind == "pv":
                    o_has.append(j)

        def touch_o(what):
            if any(g[0] == "pv" for g in groups):
                errs.append(f"{what} while a P.V into O is in flight")

        issue("qk", gst(0))
        wait(0)
        p_tile = gst(0)
        o_scale = None
        for i in range(1, nloc):
            if groups:
                errs.append(f"iteration {i} entered with {groups}")
            issue("qk", gst(i))
            issue("pv", gst(i - 1))
            wait(1 if mut != "wait2" else 2)
            if ("qk", gst(i)) in groups:      # softmax(i) reads its scores
                errs.append(f"softmax({gst(i)}) before its Q.K completed")
            if mut == "rescale_early":
                touch_o("rescale")
                o_scale = gst(i)
            wait(0 if mut != "no_wait0" else 1)
            if mut != "rescale_early":
                touch_o("rescale")
                if mut != "no_rescale":
                    o_scale = gst(i)
            if gst(i - 1) not in o_has:
                errs.append(f"release({gst(i - 1)}) before its P.V completed")
            released.append(gst(i - 1))
            p_tile = gst(i)
        issue("pv", gst(nloc - 1))
        wait(0 if mut != "no_final_wait" else 1)
        touch_o("final O read")
        if gst(nloc - 1) not in o_has:
            errs.append("last release before its P.V completed")
        released.append(gst(nloc - 1))
        if groups:
            errs.append(f"in flight at exit: {groups}")
        if sorted(o_has) != sorted(gst(i) for i in range(nloc)):
            errs.append(f"O holds {o_has}")
    if sorted(released) != list(range(nst)):
        errs.append(f"released {sorted(released)} for nst {nst}")
    for bid, (sy, ar) in turns.items():
        if sy != ar:
            errs.append(f"turn barrier {bid}: {sy} syncs vs {ar} arrives")
        if bid <= 2 * NP:
            errs.append(f"turn barrier {bid} collides with a P-tile barrier")
    return errs


bad = 0
for NP in (1, 2):
    for nst in range(0, 13):
        e = run(nst, NP)
        if e:
            print(f"FAIL NP {NP} nst {nst}: {e[:3]}")
            bad += 1
print(f"{'ok  ' if not bad else 'FAIL'} W3 order for NP 1, 2 and nst 0..12")
for mut in ("parity", "pp_extra", "pp_id", "wait2", "rescale_early", "no_wait0", "no_rescale", "no_final_wait"):
    caught = any(run(nst, NP, mut) for NP in (1, 2) for nst in range(0, 13))
    print(("caught  " if caught else "MISSED  ") + f"mutant {mut}")
    bad += not caught
print("sim_w3_pipe:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

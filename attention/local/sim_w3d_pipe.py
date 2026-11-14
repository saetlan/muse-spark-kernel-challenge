"""Deadlock simulation of W3d's producer / consumer barriers (make_w3d.py, AF61): 4 producer warps and NP consumer pairs
run as step programs; a step blocks only on an mbarrier the other side must arrive on (cp.async groups always
complete eventually, so waiting on them never deadlocks). Transcribed rules:
- producer warp (each of 4), iteration i: [order "new": if i >= S, relay every stage <= i - S + NP not yet relayed];
  if i >= S, wait empty(i - S) (released by the owning consumer pair); issue stage i; relay stage i - S + 1 if not yet;
  after the loop relay the rest. full(j) needs all 4 warps' relay(j).
- consumer pair par (FA3 order, make_w3.py): local stage k = global NP*k + par: k = 0: wait full; k >= 1: wait
  full(gst(k)), then release(gst(k-1)); after the loop release(gst(nloc - 1)).
Asserts: the "new" order finishes for every nst 0..16, S in 4..8, NP 1..2 with every stage relayed and released once;
the "old" order (no relay before the empty wait, L35's W3d) deadlocks for some nst (the hang L35 hit).
Run: .venv/bin/python attention/local/sim_w3d_pipe.py"""
import sys


def run(nst, S, NP, order, P=4):
    relayed = [set() for _ in range(P)]
    released = set()
    prod = [dict(i=0, phase="pre", done=False, nrel=0) for _ in range(P)]
    cons = []
    for par in range(NP):
        nloc = (nst - par + NP - 1) // NP
        cons.append(dict(par=par, nloc=nloc, k=0, done=nloc <= 0))

    def full(j):
        return all(j in r for r in relayed)
    progress = True
    while progress:
        progress = False
        for q, w in enumerate(prod):
            if w["done"]:
                continue
            i = w["i"]
            if i >= nst:
                for j in range(nst):
                    relayed[q].add(j)
                w["done"] = True
                progress = True
                continue
            if w["phase"] == "pre":
                if i >= S and order == "new":
                    for j in range(0, i - S + NP + 1):
                        relayed[q].add(j)
                w["phase"] = "wait"
                progress = True
            if w["phase"] == "wait":
                if i >= S and (i - S) not in released:
                    continue
                # issue stage i, then relay i - S + 1
                if i - S + 1 >= 0:
                    for j in range(0, i - S + 2):
                        relayed[q].add(j)
                w["i"] += 1
                w["phase"] = "pre"
                progress = True
        for c in cons:
            if c["done"]:
                continue
            k, par = c["k"], c["par"]
            if k < c["nloc"]:
                g = NP * k + par
                if not full(g):
                    continue
                if k >= 1:
                    released.add(NP * (k - 1) + par)
                c["k"] += 1
                progress = True
            else:
                if c["nloc"] > 0:
                    released.add(NP * (c["nloc"] - 1) + par)
                c["done"] = True
                progress = True
    ok = all(w["done"] for w in prod) and all(c["done"] for c in cons)
    return ok and released == set(range(nst)) and all(r == set(range(nst)) for r in relayed)


bad = 0
for NP in (1, 2):
    for S in range(4, 9):
        if S - 1 - NP < 0:
            continue
        for nst in range(0, 17):
            if not run(nst, S, NP, "new"):
                print(f"FAIL new order deadlocks: nst {nst} S {S} NP {NP}")
                bad += 1
old_hangs = any(not run(nst, S, 2, "old") for S in (4, 6, 7) for nst in range(0, 17))
print(("caught  " if old_hangs else "MISSED  ") + "the old order (L35's W3d) deadlocks")
bad += not old_hangs
print("sim_w3d_pipe:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

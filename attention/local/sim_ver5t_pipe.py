"""Protocol simulation of V5t's stage pipeline (cuda_ver5t.py): producer copies -> land_bar -> producer
re-lays V (transpose) -> full_bar -> consumers read K (ring) and V (vt) -> empty_bar -> producer reuses
the slot. The transpose lag D and the drain loop are parsed from the kernel source; the model runs the
producer and WC consumer warps as sequential programs under a random scheduler, with cp.async copies
completing at random later times (but before any wait that requires them). Barrier phases are
tracked per slot like mbarrier parity. Checks, for STAGES 2..4 and 0..24 stages, many seeds:
  - no deadlock (every warp finishes);
  - a consumer reads stage j's K only after stage j's copies landed and V only after transpose(j)
    wrote it (and before anything overwrites either);
  - copies(j) overwrite ring slot j % S only after every consumer released stage j - S;
  - transpose(j) reads ring V of stage j only after it landed and before copies(j + S) overwrite it,
    and overwrites vt slot j % S only after every consumer released stage j - S;
  - every stage is transposed exactly once.
Mutants (lag S instead of S - 1; drain starting one stage late; full released before the transpose
writes) must fail.
Run: .venv/bin/python attention/local/sim_ver5t_pipe.py"""
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "cuda_ver5t.py")).read(), ns)
SRC = ns["CUDA_VER5T_SRC"]
m_lag = re.search(r"if \(i >= STAGES - (\d)\) transpose\(i - \(STAGES - (\d)\)\);", SRC)
m_drain = re.search(r"for \(int j = max\(nst - \(STAGES - (\d)\), 0\); j < nst; \+\+j\) transpose\(j\);", SRC)
assert m_lag and m_drain and m_lag.group(1) == m_lag.group(2), "transpose schedule not found"
LAG_SRC, DRAIN_SRC = int(m_lag.group(1)), int(m_drain.group(1))
WC = 8


def simulate(S, nst, seed, lag_off=0, drain_off=0, early_full=False):
    rnd = random.Random(seed)
    D = S - LAG_SRC + lag_off            # transpose(i - D) after issuing stage i
    drain_from = max(nst - (S - DRAIN_SRC) + drain_off, 0)
    ring_owner = {}                      # slot -> stage whose copies were issued into it
    landed = set()                       # stages whose copies completed
    pending = []                         # stages with copies in flight
    vt_owner = {}                        # slot -> stage whose V was re-laid into it
    transposed = []
    full_done, released = set(), {}      # stage -> number of consumer warps that released it
    problems = []

    def prod():
        def transpose(j):
            while j not in landed:       # land_bar wait (copies of j complete at random)
                yield "wait"
            if ring_owner.get(j % S) != j:
                problems.append(f"transpose({j}) reads a ring slot holding stage {ring_owner.get(j % S)}")
            prev = j - S
            if prev >= 0 and released.get(prev, 0) < WC:
                problems.append(f"transpose({j}) overwrites vt slot before stage {prev} was released")
            if early_full:
                full_done.add(j)
                yield "step"
            vt_owner[j % S] = j
            transposed.append(j)
            yield "step"
            full_done.add(j)
        for i in range(nst):
            if i >= S:
                while released.get(i - S, 0) < WC:   # empty_bar wait
                    yield "wait"
            ring_owner[i % S] = i
            pending.append(i)
            yield "step"
            if D >= 0 and i >= D:
                yield from transpose(i - D)
        for j in range(drain_from, nst):
            yield from transpose(j)

    def cons(w):
        for i in range(nst):
            while i not in full_done:
                yield "wait"
            if i not in landed or ring_owner.get(i % S) != i:
                problems.append(f"consumer {w} reads K of stage {i} before it landed / after an overwrite")
            if vt_owner.get(i % S) != i:
                problems.append(f"consumer {w} reads V of stage {i} from a vt slot holding {vt_owner.get(i % S)}")
            yield "step"
            released[i] = released.get(i, 0) + 1
            yield "step"

    progs = [prod()] + [cons(w) for w in range(WC)]
    alive = list(range(len(progs)))
    stuck = 0
    while alive:
        if pending and rnd.random() < 0.3:              # some copies complete
            landed.add(pending.pop(rnd.randrange(len(pending))))
        k = rnd.choice(alive)
        try:
            r = next(progs[k])
            stuck = 0 if r == "step" else stuck + 1
        except StopIteration:
            alive.remove(k)
            stuck = 0
        if stuck > 2000 and not pending:
            problems.append(f"deadlock (S {S}, nst {nst})")
            break
        if stuck > 20000:
            problems.append(f"deadlock (S {S}, nst {nst})")
            break
    if sorted(transposed) != list(range(nst)) and not any("deadlock" in p for p in problems):
        problems.append(f"stages transposed {sorted(transposed)} (want 0..{nst - 1} once)")
    return problems


def run_all(**mut):
    bad = []
    for S in (2, 3, 4):
        for nst in range(0, 25):
            for seed in range(6):
                p = simulate(S, nst, seed * 1000 + nst * 10 + S, **mut)
                if p:
                    bad.append(p[0])
                    break
    return bad


probs = run_all()
print(("ok   " if not probs else "FAIL ") + f"V5t pipeline (lag STAGES - {LAG_SRC}, drain from nst - (STAGES - {DRAIN_SRC})): {probs[:3]}")
bad = bool(probs)
for name, mut in (("lag STAGES instead of STAGES - 1", dict(lag_off=1)), ("drain starts one stage late", dict(drain_off=1)),
                  ("full released before the V writes", dict(early_full=True))):
    p = run_all(**mut)
    print(("caught   " if p else "SURVIVED ") + f"mutant: {name} ({p[:1]})")
    bad |= not p
print("sim_ver5t_pipe:", "all ok" if not bad else "PROBLEMS")
sys.exit(1 if bad else 0)

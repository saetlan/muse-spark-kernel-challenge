"""Protocol simulation of W1's stage pipeline (cuda_w1.py): copier copies -> land_bar (32 noinc arrivals)
-> 2 transposer warps (each re-lays half of the stage's V) -> full_bar (2 arrivals) -> 8 consumer warps
read K (slot) and Vt (slot) -> empty_bar (8 arrivals) -> copier reuses the slot. Barrier counts are
parsed from the kernel source; random scheduler, copies complete at random later times. Checks, for
STAGES 3, 4, 6 and 0..20 stages, many seeds: no deadlock; a transposer reads a stage's staging only
after it landed and before it is overwritten, and writes Vt only after every consumer warp released the
slot's previous stage; a consumer reads K / Vt of stage i only when both transposers finished stage i
and nothing overwrote them. Mutants (full count 1, empty count 4) must fail.
Run: .venv/bin/python attention/local/sim_w1_pipe.py"""
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "cuda_w1.py")).read(), ns)
SRC = ns["CUDA_W1_SRC"]
cnt = {name: int(v) for v, name in re.findall(r"mbarrier\.init\.shared::cta\.b64 \[%0\], (\d+);\" ::\"r\"\(smem_u32\(&(\w+)\[s\]", SRC)}
assert set(cnt) == {"land_bar", "full_bar", "empty_bar"}, cnt
NT, NC = 2, 8


def simulate(S, nst, seed, full_n=None, empty_n=None):
    rnd = random.Random(seed)
    full_n = full_n or cnt["full_bar"]
    empty_n = empty_n or cnt["empty_bar"]
    owner, vt_parts, landed, pending = {}, {}, set(), []
    full_arr, empty_arr = {}, {}
    problems = []

    def copier():
        for i in range(nst):
            if i >= S:
                while empty_arr.get(i - S, 0) < empty_n:
                    yield "wait"
            owner[i % S] = i
            pending.append(i)
            yield "step"

    def transposer(t):
        for j in range(nst):
            while j not in landed:
                yield "wait"
            if owner.get(j % S) != j:
                problems.append(f"transposer {t} reads staging of stage {j} holding {owner.get(j % S)}")
            if j >= S and empty_arr.get(j - S, 0) < cnt["empty_bar"]:
                problems.append(f"transposer {t} writes Vt of stage {j} before stage {j - S} was released")
            vt_parts.setdefault(j % S, {})
            vt_parts[j % S] = {k: v for k, v in vt_parts[j % S].items() if v == j}
            vt_parts[j % S][t] = j
            yield "step"
            full_arr[j] = full_arr.get(j, 0) + 1

    def consumer(c):
        for i in range(nst):
            while full_arr.get(i, 0) < full_n:
                yield "wait"
            parts = vt_parts.get(i % S, {})
            if owner.get(i % S) != i or i not in landed:
                problems.append(f"consumer {c} reads K of stage {i} from a slot holding {owner.get(i % S)}")
            if sorted(k for k, v in parts.items() if v == i) != list(range(NT)):
                problems.append(f"consumer {c} reads Vt of stage {i} with parts {parts}")
            yield "step"
            empty_arr[i] = empty_arr.get(i, 0) + 1

    progs = [copier()] + [transposer(t) for t in range(NT)] + [consumer(c) for c in range(NC)]
    alive, stuck = list(range(len(progs))), 0
    while alive:
        if pending and rnd.random() < 0.3:
            landed.add(pending.pop(rnd.randrange(len(pending))))
        k = rnd.choice(alive)
        try:
            stuck = 0 if next(progs[k]) == "step" else stuck + 1
        except StopIteration:
            alive.remove(k)
            stuck = 0
        if stuck > 4000 and not pending:
            problems.append(f"deadlock (S {S}, nst {nst})")
            break
    return problems


def run_all(**mut):
    for S in (3, 4, 6):
        for nst in range(0, 21):
            for seed in range(5):
                p = simulate(S, nst, seed * 977 + nst * 13 + S, **mut)
                if p:
                    return p[:1]
    return []


probs = run_all()
print(("ok   " if not probs else "FAIL ") + f"W1 pipeline (land {cnt['land_bar']}, full {cnt['full_bar']}, empty {cnt['empty_bar']}): {probs}")
bad = bool(probs) or cnt != {"land_bar": 32, "full_bar": 2, "empty_bar": 8}
for name, mut in (("full released after one transposer", dict(full_n=1)), ("empty after one warpgroup", dict(empty_n=4))):
    p = run_all(**mut)
    print(("caught   " if p else "SURVIVED ") + f"mutant: {name} ({p})")
    bad |= not p
print("sim_w1_pipe:", "all ok" if not bad else "PROBLEMS")
sys.exit(1 if bad else 0)

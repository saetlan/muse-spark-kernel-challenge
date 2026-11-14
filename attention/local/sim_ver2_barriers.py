"""Discrete simulation of cuda_ver2.py's mbarrier protocol (no data): one producer and 4 consumer
warps, STAGES full/empty barriers with the kernel's counts (full: 32 producer lanes, empty: 4
consumer warps) and parity formulas (producer waits empty with parity (i/S - 1) & 1 for i >= S;
consumers wait full with parity (i/S) & 1). A barrier's phase completes when its pending count
reaches 0, then resets. try_wait.parity p succeeds once the phase with parity p has completed,
i.e. the barrier's completed-phase count c satisfies c > k where the waited phase is k, k & 1 = p.
Random interleavings must always finish (no deadlock) and consumers must never read a stage
before the producer filled it for that use, nor the producer refill before all consumers released.
Run: .venv/bin/python attention/local/sim_ver2_barriers.py"""
import random


def run(nst, S, seed):
    rnd = random.Random(seed)
    full_done = [0] * S
    empty_done = [0] * S
    full_pending = [32] * S
    empty_pending = [4] * S
    filled_for = [-1] * S          # which use i the stage currently holds
    prod_i = 0
    cons_i = [0] * 4
    def can_wait(done, k):          # waiting for phase k (parity k & 1): completed phases > k
        return done > k
    steps = 0
    while prod_i < nst or any(c < nst for c in cons_i):
        steps += 1
        if steps > 100000:
            return "deadlock"
        actors = []
        if prod_i < nst:
            st = prod_i % S
            if prod_i < S or can_wait(empty_done[st], prod_i // S - 1):
                actors.append("p")
        for w in range(4):
            i = cons_i[w]
            if i < nst and can_wait(full_done[i % S], i // S):
                actors.append(w)
        if not actors:
            return "deadlock"
        a = rnd.choice(actors)
        if a == "p":
            st = prod_i % S
            if prod_i >= S:
                # every consumer must have released use prod_i - S
                assert all(c > prod_i - S for c in cons_i), "producer refilled a stage still in use"
            filled_for[st] = prod_i
            full_pending[st] -= 32          # all 32 lanes' copies land (arrive.noinc each)
            if full_pending[st] == 0:
                full_done[st] += 1
                full_pending[st] = 32
            prod_i += 1
        else:
            i = cons_i[a]
            st = i % S
            assert filled_for[st] == i, f"consumer read stage {st} holding use {filled_for[st]}, wanted {i}"
            empty_pending[st] -= 1
            if empty_pending[st] == 0:
                empty_done[st] += 1
                empty_pending[st] = 4
            cons_i[a] += 1
    return "ok"


bad = 0
for S in (2, 3, 4):
    for nst in (0, 1, 2, 3, 5, 8, 13, 40):
        for seed in range(30):
            r = run(nst, S, seed)
            if r != "ok":
                print("FAIL", S, nst, seed, r)
                bad += 1
print("sim_ver2_barriers:", "ok (no deadlock, no early read, no early refill)" if not bad else f"{bad} failures")
raise SystemExit(1 if bad else 0)

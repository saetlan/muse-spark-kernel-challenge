"""Randomized simulation of R14 v6's cluster ring protocol (gemm_kernel<2, G, CL, 0>): CL CTAs, each with a
producer thread and two consumer warpgroups, S ring slots, KC chunks, DEPTH = 2 wgmma groups in flight.

Ported statement by statement from v6/r14c.py (the lines are pinned in check_v6_cpu.py):
  producer: for kc: if round > 0: wait empty[s] parity (round-1)&1 (own CTA);
            arrive.expect_tx(full[s], W + B) on its own CTA; own weight copy -> own full[s];
            multicast slice `rank` of the B tile -> slot s of every CTA, complete_tx on each CTA's full[s];
            s, round advance.
  consumer: load_frag(slot 0, parity 0); Ring{si 0, sl 1, sr 0, pl 0};
            step kc: issue wgmma group (reads B of slot si); wait_group<DEPTH>;
                     if kc >= DEPTH: arrive empty[sr] of every CTA (CL arrivals); sr = next;
                     if kc + 1 < KC: load_frag(sl, pl) (wait full, read W of slot sl); sl = next; if sl == 0: pl ^= 1;
                     si = next.
  barriers: full init 1 (the producer's arrive.expect_tx), empty init 2*CL; mbarrier phase completes when
  pending arrivals == 0 and tx == 0 (tx may go negative: a remote slice can land before the local expect_tx);
  try_wait.parity(p) succeeds iff the current phase parity != p (hardware sees parity only).
  Async copies and wgmma groups complete at random later times. A cluster barrier after init and before exit.
Checks: data of slot s in CTA j is never overwritten while a consumer of CTA j still needs it (from its
load_frag of that chunk until the wgmma group reading it completed); every read sees the right chunk in all
slices; no deadlock; no copy in flight when the CTAs exit. MUTATIONS plants protocol bugs; each must be caught.

Run: .venv/bin/python prefill_layer/local/sim_cluster_ring.py
"""

import random
import sys

DEPTH = 2


class Bar:
    def __init__(self, count):
        self.count, self.pending, self.tx, self.phase = count, count, 0, 0

    def _check(self):
        if self.pending == 0 and self.tx == 0:
            self.phase += 1
            self.pending = self.count

    def arrive(self, tx=0):
        self.tx += tx
        self.pending -= 1
        assert self.pending >= 0, "barrier over-arrived"
        self._check()

    def complete_tx(self, n):
        self.tx -= n
        if self.pending == 0 and self.tx == 0:
            self._check()

    def done(self, parity):
        return (self.phase & 1) != parity


class Violation(Exception):
    pass


def simulate(CL, S, KC, seed, mut=None, order="base", NC=2):
    """NC consumer warpgroups per CTA (R54 wide_kernel: NC = 4, CL = 1, empty count 4)."""
    rng = random.Random(seed)
    W, B = 1, CL  # abstract byte counts: weights 1 unit, B tile CL slices of 1 unit
    empty_count = NC * CL if mut != "empty_count_2" else 2
    if mut == "wide_empty_3":
        empty_count = 3
    full = [[Bar(1) for _ in range(S)] for _ in range(CL)]
    empty = [[Bar(empty_count) for _ in range(S)] for _ in range(CL)]
    # slot contents: tag of the chunk last written: weights and each B slice
    wtag = [[None] * S for _ in range(CL)]
    btag = [[[None] * CL for _ in range(S)] for _ in range(CL)]
    needs = [[set() for _ in range(S)] for _ in range(CL)]  # (consumer, chunk) still needing slot data
    inflight = []  # pending async events: ("copy", cta, slot, kind, slice, chunk) / ("group", cta, c, slot, chunk)

    def land(ev):
        _, j, s, kind, sl, chunk = ev
        if needs[j][s] and any(k != chunk for (_, k) in needs[j][s]):
            raise Violation(f"overwrite: CTA {j} slot {s} chunk {chunk} while needed by {sorted(needs[j][s])}")
        if kind == "w":
            wtag[j][s] = chunk
            full[j][s].complete_tx(W)
        else:
            btag[j][s][sl] = chunk
            full[j][s].complete_tx(1)

    # actor state machines as generators yielding a predicate (blocked until True) or None (just a step)
    def producer(r):
        s, rnd = 0, 0
        for kc in range(KC // 2 if order == "pairs" else KC):
            if rnd > 0:
                bar = empty[r][s]
                par = (rnd - 1) & 1 if mut != "parity" else rnd & 1
                yield lambda bar=bar, par=par: bar.done(par)
            full[r][s].arrive(tx=(W + B) if mut != "expect_own_slice_only" else (W + 1))
            inflight.append(("copy", r, s, "w", 0, kc))
            targets = range(CL) if mut != "no_multicast" else [r]
            for j in targets:
                inflight.append(("copy", j, s, "b", r, kc))
            yield None
            s += 1
            if s == S:
                s, rnd = 0, rnd + 1

    groups = {}  # (cta, c) -> list of outstanding group chunks

    def consumer(r, c):
        def load_frag(slot, par, chunk):
            yield lambda: full[r][slot].done(par)
            if wtag[r][slot] != chunk or any(t != chunk for t in btag[r][slot]):
                raise Violation(f"CTA {r} c{c} read slot {slot} for chunk {chunk}: w {wtag[r][slot]} b {btag[r][slot]}")
            needs[r][slot].add((c, chunk))

        yield from load_frag(0, 0, 0)
        si, sl, sr, pl = 0, 1, 0, 0
        groups[(r, c)] = []
        if order == "pairs":
            # v6 DIAG 16/17: ring slots hold chunk pairs (tag = pair index). Step kc: issue chunk kc's group (B of
            # pair kc/2), wait_group<DEPTH>, at odd kc >= 3 release pair (kc-3)/2, and when kc+2 starts a pair wait
            # for that pair (its raw words are read one step ahead).
            def wait_pair(stage, par, tag):
                yield lambda: full[r][stage].done(par)
                if wtag[r][stage] != tag or any(t != tag for t in btag[r][stage]):
                    raise Violation(f"CTA {r} c{c} pair {tag} slot {stage}: w {wtag[r][stage]} b {btag[r][stage]}")
                needs[r][stage].add((c, tag))

            yield from wait_pair(0, 0, 0)
            ps, pp, s_issue, sr = 1 % S, 0 if S > 1 else 1, 0, 0
            for kc in range(KC):
                if any(t != kc // 2 for t in btag[r][s_issue]):
                    raise Violation(f"CTA {r} c{c} wgmma chunk {kc} slot {s_issue} sees {btag[r][s_issue]}")
                groups[(r, c)].append(kc)
                inflight.append(("gpair", r, c, s_issue, kc // 2, kc & 1, kc))
                yield None
                yield lambda: len(groups[(r, c)]) <= DEPTH
                if kc & 1 and kc >= (3 if mut != "pairs_release_early" else 1):
                    for j in range(CL):
                        empty[j][sr].arrive()
                    yield None
                    sr = sr + 1 if sr + 1 < S else 0
                if kc + 2 < KC and (kc + 2) % 2 == 0:
                    yield from wait_pair(ps, pp, (kc + 2) // 2)
                    ps = ps + 1 if ps + 1 < S else 0
                    if ps == 0:
                        pp ^= 1
                if kc & 1:
                    s_issue = s_issue + 1 if s_issue + 1 < S else 0
            yield lambda: len(groups[(r, c)]) == 0
            return
        if order == "early":
            # v6 DIAG 8/9: wait + read chunk kc+1 first, then issue chunk kc's group, decode in registers,
            # wait_group<DEPTH - 1>, then release chunk kc - (DEPTH - 1).
            for kc in range(KC):
                if kc + 1 < KC:
                    yield from load_frag(sl, pl, kc + 1)
                    sl = sl + 1 if sl + 1 < S else 0
                    if sl == 0:
                        pl ^= 1
                if any(t != kc for t in btag[r][si]):
                    raise Violation(f"CTA {r} c{c} wgmma chunk {kc} slot {si} sees {btag[r][si]}")
                groups[(r, c)].append(kc)
                inflight.append(("group", r, c, si, kc))
                yield None
                if mut != "early_no_wait":
                    yield lambda: len(groups[(r, c)]) <= DEPTH - 1
                if kc >= DEPTH - 1:
                    for j in range(CL):
                        empty[j][sr].arrive()
                    yield None
                    sr = sr + 1 if sr + 1 < S else 0
                si = si + 1 if si + 1 < S else 0
            yield lambda: len(groups[(r, c)]) == 0
            return
        for kc in range(KC):
            if any(t != kc for t in btag[r][si]):
                raise Violation(f"CTA {r} c{c} wgmma chunk {kc} slot {si} sees {btag[r][si]}")
            groups[(r, c)].append(kc)
            inflight.append(("group", r, c, si, kc))
            yield None
            yield lambda: len(groups[(r, c)]) <= DEPTH
            if kc >= (DEPTH if mut != "release_early" else DEPTH - 1):
                if mut == "release_own_only":
                    empty[r][sr].arrive()
                else:
                    for j in range(CL):
                        empty[j][sr].arrive()
                yield None
                sr = sr + 1 if sr + 1 < S else 0
            if kc + 1 < KC:
                yield from load_frag(sl, pl, kc + 1)
                sl = sl + 1 if sl + 1 < S else 0
                if sl == 0:
                    pl ^= 1
            si = si + 1 if si + 1 < S else 0
        yield lambda: len(groups[(r, c)]) == 0

    actors = []
    for r in range(CL):
        actors.append(producer(r))
        for c_ in range(NC):
            actors.append(consumer(r, c_))
    waiting = [None] * len(actors)
    finished = [False] * len(actors)
    steps = 0
    while True:
        steps += 1
        if steps > 200000:
            raise Violation("livelock")
        ready = [i for i, a in enumerate(actors) if not finished[i] and (waiting[i] is None or waiting[i]())]
        options = [("actor", i) for i in ready] + [("event", k) for k in range(len(inflight))]
        if not options:
            if all(finished):
                break
            raise Violation(f"deadlock (CL {CL}, S {S})")
        kind, idx = rng.choice(options)
        if kind == "event":
            ev = inflight.pop(idx)
            if ev[0] == "copy":
                land(ev)
            elif ev[0] == "gpair":
                _, r, c, slot, tag, last, chunk = ev
                lst = groups[(r, c)]
                if lst[0] != chunk:
                    inflight.append(ev)
                    continue
                if any(t != tag for t in btag[r][slot]):
                    raise Violation(f"CTA {r} c{c} wgmma chunk {chunk} slot {slot} data changed: {btag[r][slot]}")
                lst.pop(0)
                if last:
                    needs[r][slot].discard((c, tag))
            else:
                _, r, c, slot, chunk = ev
                lst = groups[(r, c)]
                if lst[0] != chunk:  # groups complete in order
                    inflight.append(ev)
                    continue
                if any(t != chunk for t in btag[r][slot]):
                    raise Violation(f"CTA {r} c{c} wgmma chunk {chunk} slot {slot} data changed: {btag[r][slot]}")
                lst.pop(0)
                needs[r][slot].discard((c, chunk))
            continue
        try:
            waiting[idx] = next(actors[idx])
        except StopIteration:
            finished[idx] = True
            waiting[idx] = None
    # cluster barrier before exit: every actor finished; nothing may still be in flight
    if inflight:
        raise Violation(f"exit with {len(inflight)} async operations in flight")
    return steps


MUTATIONS = ("empty_count_2", "release_own_only", "no_multicast", "expect_own_slice_only", "parity",
             "release_early", "ring_too_small", "early_no_wait", "pairs_release_early", "pairs_ring_2")


def main():
    runs = 0
    for order in ("base", "early"):
        for CL in (1, 2, 4):
            for S in (DEPTH + 2, DEPTH + 3, 5, 8):
                for seed in range(60):
                    simulate(CL, S, 11, seed, order=order)
                    runs += 1
    for seed in range(60):  # early order with the smallest ring it needs (3 slots)
        simulate(1, 3, 11, seed, order="early")
    for S_ in (4, 5, 8):  # R54: 4 consumer warpgroups, empty count 4, S >= 4 (wide_kernel)
        for seed in range(80):
            simulate(1, S_, 11, seed, NC=4)
            runs += 1
    for S_ in (3, 4, 6):  # pair stages: at least 3 slots
        for seed in range(80):
            simulate(1, S_, 16, seed, order="pairs")
            runs += 1
    print(f"ok: cluster ring protocol, {runs} random schedules (CL 1/2/4, S {DEPTH + 2}..8, KC 11): "
          "no overwrite in use, every read sees its chunk, no deadlock, nothing in flight at exit")
    missed = []
    for mut in MUTATIONS:
        caught = False
        for CL in (2, 4):
            for S in ((DEPTH,) if mut == "ring_too_small" else (2,) if mut == "pairs_ring_2" else (DEPTH + 2, 5)):
                for seed in range(60):
                    try:
                        order = "early" if mut == "early_no_wait" else "pairs" if mut.startswith("pairs") else "base"
                        simulate(CL, S, 16 if order == "pairs" else 11, seed,
                                 None if mut in ("ring_too_small", "pairs_ring_2") else mut, order=order)
                    except (Violation, AssertionError):
                        caught = True
                        break
                if caught:
                    break
            if caught:
                break
        if not caught:
            missed.append(mut)
    if missed:
        raise SystemExit(f"FAIL: protocol mutations not caught: {missed}")
    print(f"ok: all {len(MUTATIONS)} planted protocol bugs caught ({', '.join(MUTATIONS)})")
    for mut in ("wide_empty_3", "empty_count_2"):  # R54: an empty count below 4 frees a slot a warpgroup still reads
        caught = False
        for seed in range(200):
            try:
                simulate(1, 4, 11, seed, mut, NC=4)
            except (Violation, AssertionError):
                caught = True
                break
        if not caught:
            raise SystemExit(f"FAIL: R54 protocol mutation not caught: {mut}")
    print("ok: R54 (4 consumer warpgroups) planted empty-count bugs caught (3, 2)")




# ------------------------------------------------------------------------------------------------ R53
def simulate_dq(S, SA, KC, seed, mut=None):
    """R53 dq_kernel protocol: producer (TMA ring of S slots: W + B per chunk), two decode warpgroups (chunk kc
    by warpgroup kc % 2, 4 warps each) writing the A ring (SA slots), two MMA warpgroups (wait A-full, issue,
    wait_group<1>, release chunk kc-1's A slot and TMA slot). Ported from v6/r14c.py's dq_kernel; checks no
    overwrite in use (TMA slot W/B, A slot), right data everywhere, no deadlock, nothing in flight at exit."""
    rng = random.Random(seed)
    full = [Bar(1) for _ in range(S)]
    empty = [Bar(6 if mut != "dq_empty_5" else 5) for _ in range(S)]
    afull = [Bar(4) for _ in range(SA)]
    aempty = [Bar(2) for _ in range(SA)]
    wtag, btag, atag = [None] * S, [None] * S, [None] * SA
    need_w = [set() for _ in range(S)]   # decode warps reading W of chunk
    need_b = [set() for _ in range(S)]   # MMA groups reading B of chunk
    need_a = [set() for _ in range(SA)]  # MMA groups reading A of chunk
    inflight = []
    groups = {0: [], 1: []}

    def producer():
        s, rnd = 0, 0
        for kc in range(KC):
            if rnd > 0:
                bar, par = empty[s], (rnd - 1) & 1
                yield lambda bar=bar, par=par: bar.done(par)
            full[s].arrive(tx=2)
            inflight.append(("copy", s, "w", kc))
            inflight.append(("copy", s, "b", kc))
            yield None
            s += 1
            if s == S:
                s, rnd = 0, rnd + 1

    def decode_warp(d, wp):
        kc, s, ps, a, ra = d, d % S, (d // S) & 1, d % SA, d // SA
        while kc < KC:
            yield lambda s=s, ps=ps: full[s].done(ps)
            if kc >= SA:
                par = (ra - 1) & 1
                yield lambda a=a, par=par: aempty[a].done(par)
            if wtag[s] != kc:
                raise Violation(f"decode {d}.{wp} chunk {kc} slot {s} sees W {wtag[s]}")
            if need_a[a]:
                raise Violation(f"decode {d}.{wp} writes A slot {a} chunk {kc} while needed by {need_a[a]}")
            yield None
            atag[a] = kc if wp == 0 else atag[a]  # the tile is complete once all 4 warps arrived
            yield None
            if mut != "dq_no_afull":
                afull[a].arrive()
            empty[s].arrive()
            kc += 2
            s += 2
            if s >= S:
                s -= S
                ps ^= 1
            a += 2
            if a >= SA:
                a -= SA
                ra += 1

    def mma(c):
        a, pa, s = 0, 0, 0
        a_prev = s_prev = None
        for kc in range(KC):
            yield lambda a=a, pa=pa: afull[a].done(pa)
            if atag[a] != kc:
                raise Violation(f"MMA {c} chunk {kc} A slot {a} sees {atag[a]}")
            if btag[s] != kc:
                raise Violation(f"MMA {c} chunk {kc} B slot {s} sees {btag[s]}")
            need_a[a].add((c, kc))
            need_b[s].add((c, kc))
            groups[c].append(kc)
            inflight.append(("group", c, a, s, kc))
            yield None
            yield lambda: len(groups[c]) <= (1 if mut != "dq_wait2" else 2)
            if kc >= 1:
                aempty[a_prev].arrive()
                empty[s_prev].arrive()
                yield None
            a_prev, s_prev = a, s
            a += 1
            if a == SA:
                a, pa = 0, pa ^ 1
            s = s + 1 if s + 1 < S else 0
        yield lambda: len(groups[c]) == 0

    actors = [producer()] + [decode_warp(d, wp) for d in (0, 1) for wp in range(4)] + [mma(0), mma(1)]
    waiting = [None] * len(actors)
    finished = [False] * len(actors)
    steps = 0
    while True:
        steps += 1
        if steps > 400000:
            raise Violation("livelock")
        ready = [i for i, a in enumerate(actors) if not finished[i] and (waiting[i] is None or waiting[i]())]
        options = [("actor", i) for i in ready] + [("event", k) for k in range(len(inflight))]
        if not options:
            if all(finished):
                break
            raise Violation(f"dq deadlock (S {S}, SA {SA})")
        kind, idx = rng.choice(options)
        if kind == "event":
            ev = inflight.pop(idx)
            if ev[0] == "copy":
                _, s, what, chunk = ev
                if what == "w":
                    wtag[s] = chunk
                else:
                    if need_b[s] and any(k != chunk for (_, k) in need_b[s]):
                        raise Violation(f"B slot {s} overwritten by {chunk} while needed by {need_b[s]}")
                    btag[s] = chunk
                full[s].complete_tx(1)
            else:
                _, c, a, s, chunk = ev
                if groups[c][0] != chunk:
                    inflight.append(ev)
                    continue
                if atag[a] != chunk or btag[s] != chunk:
                    raise Violation(f"MMA {c} group {chunk}: A {atag[a]} B {btag[s]} changed under it")
                groups[c].pop(0)
                need_a[a].discard((c, chunk))
                need_b[s].discard((c, chunk))
            continue
        try:
            waiting[idx] = next(actors[idx])
        except StopIteration:
            finished[idx] = True
            waiting[idx] = None
    if inflight:
        raise Violation("dq exit with operations in flight")


def main_dq():
    runs = 0
    # Both rings must have an even slot count: decode warpgroup d then only uses slots of parity d, in order, so a
    # parity wait never looks two phases ahead (with S = 3 warpgroup 1 waits for slot 0's second phase before its
    # first completed and the stale parity passes: the "dq_odd_S" mutation below).
    for S in (4, 6, 8):
        for SA in (2, 4):
            for seed in range(50):
                simulate_dq(S, SA, 12, seed)
                runs += 1
    print(f"ok: R53 dq protocol, {runs} random schedules (S 4/6/8, SA 2/4, KC 12): no overwrite in use, right data, "
          "no deadlock, nothing in flight at exit")
    missed = []
    for mut in ("dq_empty_5", "dq_no_afull", "dq_wait2", "dq_odd_S"):
        caught = False
        for seed in range(80):
            try:
                simulate_dq(3 if mut == "dq_odd_S" else 4, 4, 12, seed, None if mut == "dq_odd_S" else mut)
            except (Violation, AssertionError):
                caught = True
                break
        if not caught:
            missed.append(mut)
    if missed:
        raise SystemExit(f"FAIL: dq protocol mutations not caught: {missed}")
    print("ok: dq planted protocol bugs caught (empty count 5, no A-full arrive, wait_group<2> before release, odd ring)")


if __name__ == "__main__":
    main()
    main_dq()

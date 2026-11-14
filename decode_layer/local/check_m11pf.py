"""Emulate L8's prefetch addresses with parts/m11_pf.py's own index formulas: for every (gated, TT, KS) M11
runs, on random active-expert sets, the bytes each warp prefetches for task `task + k * stride` must be
exactly the bytes that warp loads when it runs that task (codes and scales), inside the weight tensors,
16-B aligned, sizes multiples of 16; the per-lane mode's 128-B lines must cover the same ranges.

Run: .venv/bin/python decode_layer/local/check_m11pf.py
"""

import random
from pathlib import Path

KG = 4096 // 64  # 64-wide K chunks (K = 4096 for both GEMMs)
LOCAL = 32


def loads(e, tile, kpart, tt, ks, gated, rt_total):
    """Byte offsets the kernel's load(s, kg) touches for one warp of a task (wp[i] + kg * 32 + lane)."""
    kgw = KG // ks
    w, s = set(), set()
    for i in range(tt):
        rt = tile + i * (rt_total // 2) if gated else tile * tt + i
        base = (e * rt_total + rt) * KG * 32
        for kg in range(kpart * kgw, (kpart + 1) * kgw):
            for lane in range(32):
                idx = base + kg * 32 + lane
                w.update(range(idx * 16, idx * 16 + 16))
                s.update(range(idx * 2, idx * 2 + 2))
    return w, s


def prefetches(e2, tile2, kpart, tt, ks, gated, rt_total):
    """The (address, bytes) pairs m11_prefetch is given (b2 formula), per array."""
    kgw = KG // ks
    out = []
    for i in range(tt):
        rt2 = tile2 + i * (rt_total // 2) if gated else tile2 * tt + i
        b2 = (e2 * rt_total + rt2) * KG * 32 + kpart * kgw * 32
        out.append(((b2 * 16, kgw * 512), (b2 * 2, kgw * 64)))
    return out


def lines(addr, nbytes):
    covered = set()
    for lane in range(32):
        for off in range(lane * 128, nbytes, 32 * 128):
            covered.update(range(addr + off, addr + off + 128))
    return covered


def main():
    src = (Path(__file__).resolve().parents[1] / "parts" / "m11_pf.py").read_text()
    for line in (  # the formulas emulated here, as the kernel has them
            "const size_t b2 = (static_cast<size_t>(e2) * rt_total + rt2) * KG * 32 + static_cast<size_t>(kpart) * KGW * 32;",
            "const int rt2 = GATED ? tile2 + i * (rt_total / 2) : tile2 * TT + i;",
            "const int e2 = s_active[nxt / per_expert], tile2 = nxt % per_expert;",
            "for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;",
            "const size_t base = (static_cast<size_t>(e) * rt_total + rts[i]) * KG * 32 + lane;",
            "rw[s][i] = __ldcs(wp[i] + kg * 32);", "rs[s][i] = __ldcs(sp[i] + kg * 32);",
            "const int kbeg = kpart * KGW;", "m11_prefetch(w + b2, sf + b2, KGW, pf >> 4, lane);",
            "const int wbytes = kgw * 512, sbytes = kgw * 64;"):
        assert src.count(line) >= 1, f"formula changed in parts/m11_pf.py: {line}"
    rng = random.Random(5)
    combos = [(1, 2, ks, 512) for ks in (1, 2, 4)] + [(0, tt, ks, 256) for tt in (1, 2) for ks in (1, 2, 4)]
    n = 0
    for gated, tt, ks, rt_total in combos:
        per_expert = rt_total // 2 if gated else rt_total // tt
        w_size, s_size = LOCAL * rt_total * KG * 32 * 16, LOCAL * rt_total * KG * 32 * 2
        for trial in range(40):
            active = sorted(rng.sample(range(LOCAL), rng.randint(1, 6)))
            n_tasks = len(active) * per_expert
            stride = rng.choice((1, 3, 7, 64, 132, 132 * 4, 132 * 8)) * (4 // ks)
            for task in rng.sample(range(n_tasks), min(n_tasks, 4)):
                for k in (1, 2):
                    nxt = task + k * stride
                    if nxt >= n_tasks:
                        continue
                    e2, tile2 = active[nxt // per_expert], nxt % per_expert
                    for kpart in range(ks):
                        want_w, want_s = loads(e2, tile2, kpart, tt, ks, gated, rt_total)
                        got_w, got_s = set(), set()
                        for (wa, wb), (sa, sb) in prefetches(e2, tile2, kpart, tt, ks, gated, rt_total):
                            assert wa % 16 == 0 and sa % 16 == 0 and wb % 16 == 0 and sb % 16 == 0
                            assert wa + wb <= w_size and sa + sb <= s_size
                            got_w.update(range(wa, wa + wb))
                            got_s.update(range(sa, sa + sb))
                            assert lines(wa, wb) == set(range(wa, wa + wb)) and lines(sa, sb) == set(range(sa, sa + sb))
                        assert got_w == want_w and got_s == want_s, (gated, tt, ks, kpart)
                        n += 1
    print(f"check_m11pf: {n} warp prefetches = that warp's next-task loads exactly (codes, scales), in bounds, "
          f"aligned; per-lane lines cover the same bytes")


if __name__ == "__main__":
    main()

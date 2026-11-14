"""Emulate L9's staging addresses with parts/l9.py's own formulas: for every M11 slice geometry (ks 1/2/4) and
fraction f16, the bytes staged for (expert, row tile, warp slice) are exactly the first f16/16 of the bytes
that warp loads for that task in M11 (parts/m11_pf.py's load formula), inside the bank, 16-B aligned and
multiples of 16; every active expert's every tile and slice is covered once; the active set from ids
matches Python's (id - local_start) % 256 < 32 for any ids and local_start, negatives included.

Run: .venv/bin/python decode_layer/local/check_l9.py
"""

import random
from pathlib import Path

PARTS = Path(__file__).resolve().parents[1] / "parts"


def warp_loads(e, r, q, ks, rt_total):
    kgw = 64 // ks
    base = (e * rt_total + r) * 64 * 32
    codes, scales = set(), set()
    for kg in range(q * kgw, (q + 1) * kgw):
        for lane in range(32):
            idx = base + kg * 32 + lane
            codes.update(range(idx * 16, idx * 16 + 16))
            scales.update(range(idx * 2, idx * 2 + 2))
    return codes, scales


def main():
    src = (PARTS / "l9.py").read_text()
    for line in ("const int e = ((ids[i] - ls) % 256 + 256) % 256;",
                 "const uint32_t wbytes = static_cast<uint32_t>(kgw * 32 * f16), sbytes = static_cast<uint32_t>(kgw * 4 * f16);",
                 "const size_t base = (static_cast<size_t>(e) * rt_total + r) * 64 * 32 + static_cast<size_t>(q) * kgw * 32;",
                 "const int q = static_cast<int>(it % ks);", "const int r = static_cast<int>(tr % rt_total);",
                 "const int e = s_list[tr / rt_total];", "pf(w + base, wbytes, mode);", "pf(sf + base, sbytes, mode);"):
        assert src.count(line) == 1, f"formula changed in parts/l9.py: {line}"
    m11 = (PARTS / "m11_pf.py").read_text()
    assert "rw[s][i] = __ldcs(wp[i] + kg * 32);" in m11 and "const int kbeg = kpart * KGW;" in m11
    rng = random.Random(9)
    n = 0
    for rt_total in (512, 256):  # w13, w2
        w_size, s_size = 32 * rt_total * 64 * 32 * 16, 32 * rt_total * 64 * 32 * 2
        for ks in (1, 2, 4):
            for f16 in (1, 4, 8, 16):
                kgw = 64 // ks
                wbytes, sbytes = kgw * 32 * f16, kgw * 4 * f16
                assert wbytes % 16 == 0 and sbytes % 16 == 0
                active = sorted(rng.sample(range(32), rng.randint(1, 5)))
                items = len(active) * rt_total * ks
                seen = set()
                for it in rng.sample(range(items), min(items, 40)) + [0, items - 1]:
                    q, tr = it % ks, it // ks
                    r, e = tr % rt_total, active[tr // rt_total]
                    seen.add((e, r, q))
                    base = (e * rt_total + r) * 64 * 32 + q * kgw * 32
                    wa, sa = base * 16, base * 2
                    assert wa % 16 == 0 and sa % 16 == 0 and wa + wbytes <= w_size and sa + sbytes <= s_size
                    codes, scales = warp_loads(e, r, q, ks, rt_total)
                    lo_c, lo_s = min(codes), min(scales)
                    # the slice is contiguous, so its first f16/16 is [start, start + bytes)
                    assert codes == set(range(lo_c, lo_c + kgw * 512)) and scales == set(range(lo_s, lo_s + kgw * 64))
                    assert (wa, sa) == (lo_c, lo_s), (ks, f16, e, r, q)
                    n += 1
                # the item -> (e, r, q) map is a bijection onto active x tiles x slices
                assert len({(active[(it // ks) // rt_total], (it // ks) % rt_total, it % ks) for it in range(items)}) == items
    for _ in range(2000):  # the active-set formula vs Python's modulo (C++ % truncates toward zero)
        ls, i = rng.randrange(0, 256, 32), rng.randrange(-300, 600)
        c_mod = abs(i - ls) % 256 * (1 if i - ls >= 0 else -1)
        e = (c_mod + 256) % 256
        assert e == (i - ls) % 256
    print(f"check_l9: {n} staged slices = the first f16/16 of that warp's M11 loads, in bounds, aligned; "
          f"items cover every (expert, tile, slice) once; active set = Python's modulo")


if __name__ == "__main__":
    main()

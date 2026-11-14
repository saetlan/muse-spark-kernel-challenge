"""PE10 section: the FP32 router at m = 32/128/256 with token tiles covering all tokens, so the 8.4 MB of
weights are read once instead of once per 64-token tile (v4's m128/256 configs use BM 64): Triton
_gemm_kernel MODE 1 (SGLang's hi + lo split in registers, allowed at any batch size), BN 16-64,
split-K 4-32 with FP32 atomics, 96-300 programs; through the gate vs SGLang and the base's choice.
Defines sec_router_wide(lab, sweep, m) and a prepare() that runs it on top of the base (append after lab/sweep.py)."""

import math


def router_wide_cands(m, max_n=36):
    out = []
    mp = 1 << max(5, (m - 1).bit_length())         # m rounded up to a power of two, at least 32
    for bm in (256, 128, 64, 32):
        if bm > mp or bm < min(mp, 64):
            continue
        for bn in (16, 32, 64):
            for split in (4, 8, 16, 32):
                for bk in (32, 64, 128):
                    tiles = math.ceil(m / bm) * (256 // bn) * split
                    if not 96 <= tiles <= 300 or 8192 % (split * bk):
                        continue
                    for warps in (4, 8):
                        if bm * bn // (warps * 32) > 64:
                            continue
                        for stages in (3, 4):
                            if stages * (bm * bk * 2 + bk * bn * 4) > 200 * 1024:   # Triton's smem ring
                                continue
                            reread = math.ceil(m / bm)
                            out.append(((reread, abs(tiles - 132 * round(tiles / 132)), -bk, warps, stages),
                                        dict(kind="gemm", BM=bm, BN=bn, BK=bk, split=split, mode=1, warps=warps, stages=stages)))
    out.sort(key=lambda e: e[0])
    return [c for _, c in out][:max_n]


def sec_router_wide(lab, sweep, m):
    cands = router_wide_cands(m)
    if cands:
        sweep.shape(lab, "router", m, cands)
        rec = sweep.results[("router", m)]
        best = sorted(rec.get("all", []), key=lambda e: e[0])[:6]
        log(f"router m{m} wide (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
            f"{rec['wrong']} wrong) best: " + " ".join(f"{cfg_name(c)}:{us:.2f}" for us, c in best))


PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def prepare(weights):
    import time
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for m in (128, 256, 32):
        lab.section(f"router wide m{m}", sec_router_wide, sweep, m, min_left=15.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if layer == "router":
            log(f"V7 CHOICE {layer} m{m}: {c!r}")
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

"""PE20: the router's hi/lo swap kernel (_hl_swap_kernel: weight rows as the MMA's M, all tokens as its N) at m = 128/256,
where v9 runs _hl_gemm_kernel with tokens as M and only 32 weight rows as N (wgmma n32; 0.58-0.72 TB/s in PE19's
lab). Swap gives wgmma m64 x n128/n256, its most efficient shapes, with x re-read once per 64-128-row weight tile
from L2. Also re-times PE19's near-winners (router m16 hl_swap +2.3%, router m128 hl_gemm +2.5%) with this
lab's fresh baselines. Winners >= 3% enter CHOICE, printed as "V16 CHOICE" lines."""

import math
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
RETIME = {
    16: [dict(kind="hl_swap", BM=16, BN=64, BK=64, split=32, warps=4, stages=5),
         dict(kind="hl_gemm", BM=16, BN=32, BK=128, split=32, warps=2, stages=3),
         dict(kind="hl_swap", BM=16, BN=64, BK=64, split=32, warps=4, stages=4),
         dict(kind="hl_swap", BM=16, BN=64, BK=128, split=32, warps=4, stages=5)],
    128: [dict(kind="hl_gemm", BM=128, BN=32, BK=128, split=16, warps=4, stages=3),
          dict(kind="hl_gemm", BM=128, BN=32, BK=64, split=16, warps=8, stages=5)],
}


def swap_cands(m, max_n=44):
    """hl_swap with BM = m (all tokens in one N tile): BN 64/128 weight rows, BK 32-128, split 4-64, warps 4/8,
    stages 2-4; accumulator <= 128 FP32 per thread, Triton's ring <= 200 KB, 64-256 programs."""
    out = []
    for bn in (64, 128):
        for bk in (32, 64, 128):
            for split in (4, 8, 16, 32, 64):
                programs = (256 // bn) * split
                if not 64 <= programs <= 256 or 8192 % (split * bk) or 8192 // split < bk:
                    continue
                for warps in (4, 8):
                    if bn * m // (warps * 32) > 128 or (warps == 8 and bn < 128 and m < 256):
                        continue
                    for stages in (2, 3, 4):
                        if stages * (2 * bn * bk * 2 + bk * m * 2) > 200 * 1024:
                            continue
                        waves = programs / 132
                        out.append(((-round(waves / math.ceil(waves), 1), -bk, stages != 3, warps),
                                    dict(kind="hl_swap", BM=m, BN=bn, BK=bk, split=split, warps=warps, stages=stages)))
    out.sort(key=lambda e: e[0])
    return [c for _, c in out][:max_n]


def hl_name(c):
    return f"{'g' if c['kind'] == 'hl_gemm' else 's'}{c['BM']}/{c['BN']}/{c['BK']}/{c['split']}/{c['warps']}/{c['stages']}"


def sec_router(lab, sweep, m):
    cands = (swap_cands(m) if m >= 128 else []) + RETIME.get(m, [])
    sweep.shape(lab, "router", m, cands)
    rec = sweep.results[("router", m)]
    times = " ".join(f"{hl_name(c)}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0])[:12])
    log(f"pe20 router m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) BM/BN/BK/split/warps/stages:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v9) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for m in (256, 128, 16):
        lab.section(f"pe20 router m{m}", sec_router, sweep, m, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") in ("hl_gemm", "hl_swap"):
            log(f"V16 CHOICE {layer} m{m}: " + repr(c))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

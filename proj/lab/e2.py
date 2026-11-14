"""PE2: v1 timed by the judge, plus in prepare() a sweep of _fma_kernel (CUDA-core GEMV, exact
FP32 for the router), first on the router at m 1, 8, 16, 32, then on the BF16 layers at m 1 and 8.
A config replaces v1's choice only if it is correct and >= MIN_GAIN faster than both SGLang and
v1's current kernel for that shape (Sweep.commit).

prepare() has 300 s in all; v1's own prepare (image-kernel JIT warm-up) took 20 s in the scored
run but its JIT took ~90 s in PE1, so the lab gets whatever is left of PREPARE_CAP_S."""

import time

PREPARE_CAP_S = 265.0
LAB_MAX_S = 200.0


def fma_candidates(layer, m, n, k):
    wbytes = n * k * (4 if layer == "router" else 2)
    cands = []
    for bn in (2, 4, 8, 16, 32, 64):
        tiles = -(-n // bn)
        for split in (1, 2, 4, 8, 16, 32, 64):
            bk = k // split
            if bk < 128 or k % split:
                continue
            ctas = tiles * split
            if not (96 <= ctas <= 2048):
                continue
            if tiles * m * k * 2 > 2 * wbytes:  # x re-read through L2 by every row tile
                continue
            for warps in (4, 8):
                elems = bn * bk // (32 * warps)  # weights held per thread
                if 8 <= elems <= 128:
                    cands.append({"kind": "fma", "BN": bn, "split": split, "warps": warps})
    return cands


def sec_fma(lab, sweep, layers, ms):
    for m in ms:
        for layer in layers:
            n, k = lab_n(layer, m), FULL[layer][1]
            sweep.shape(lab, layer, m, fma_candidates(layer, m, n, k))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    log(f"v1 prepare took {time.monotonic() - t0:.1f}s; lab budget {budget:.0f}s")
    lab = Lab(budget)
    sweep = Sweep(state)
    lab.section("fma router", sec_fma, sweep, ("router",), (1, 8, 16, 32), min_left=10.0)
    lab.section("fma bf16 m1", sec_fma, sweep, ("unpooler", "o", "qkv_gate", "pooler"), (1,), min_left=10.0)
    lab.section("fma bf16 m8", sec_fma, sweep, ("unpooler", "o", "qkv_gate", "pooler"), (8,), min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    log(f"CHOICE now has {len(CHOICE)} shapes; prepare took {time.monotonic() - t0:.0f}s")
    return state

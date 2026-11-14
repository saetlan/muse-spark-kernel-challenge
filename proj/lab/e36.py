"""PE36 (P41, DG12-RS v8): does the dry pass of the epilogue code (DIAG 256, before the main loop) remove PE35's cold-code
cost (finalize 1.55 us, push 1.5 us)? unpooler m32 and o m32, BM32 S2 MATH 4 red 2: timings of d0 / d256 / d64 / d320, and
the timestamps of the last cold-L2 graph replay of d64 and d320 (new: dry pass done, pushes issued, finalize loads summed);
then the m32 sweep with d256 next to d0 (unpooler S2, pooler S3, o S2, qkv S4; pooler S3 red 0 as reference). Winners >= 3%
enter CHOICE (fallback: v10's entry), printed as "V29 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
STAMP_NAMES = ("gt0", "clk1", "copies", "precorr", "loop0", "reduced", "bar2", "final", "exit", "gt9", "smid", "loop1",
               "dry", "pushed", "loaded")


def sec_stamps(lab, d):
    layer, m = d["layer"], d["m"]
    n = FULL[layer][0]
    ctas = n // 128 * -(-m // d["BM"]) * d["S"]
    parts = []
    for b in (64, 320):
        c = dict(BM=d["BM"], S=d["S"], CPS=d["CPS"], RED=2, MATH=4)
        bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m) + (b,))
        if bufs is None:
            parts.append(f"d{b}: no buffers")
            continue
        st = (bufs["part"][: 16 * ctas].view(torch.int32).view(ctas, 16).to(torch.int64) & 0xFFFFFFFF).cpu()
        u32 = lambda a: (a & 0xFFFFFFFF).double()   # noqa: E731  (differences mod 2^32)
        gspan = u32(st[:, 9] - st[:, 0])
        cyc = u32(st[:, 8] - st[:, 1])
        ghz = float((cyc / gspan.clamp(min=1)).median())          # cycles per ns from this run
        f = 1.0 / (1e3 * ghz) if ghz > 0.5 else 1.0 / 1755.0       # us per cycle
        med = lambda a: float(u32(a).median()) * f                 # noqa: E731
        mx = lambda a: float(u32(a).max()) * f                     # noqa: E731
        g0 = st[:, 0]
        skew = float(u32(g0 - g0.min()).max()) / 1e3
        span = float(u32(st[:, 9] - g0.min()).max()) / 1e3
        line = (f"d{b}: {ghz:.3f} GHz, {len(set(st[:, 10].tolist()))} SMs; skew {skew:.2f} us, span {span:.2f} us; "
                f"loop0 {med(st[:, 4] - st[:, 1]):.2f}/{mx(st[:, 4] - st[:, 1]):.2f}, loop1-loop0 {med(st[:, 11] - st[:, 4]):.2f}, "
                f"copies {med(st[:, 2] - st[:, 1]):.2f}, ")
        line += f"precorr {med(st[:, 3] - st[:, 1]):.2f}/{mx(st[:, 3] - st[:, 1]):.2f}, "
        if b & 256:
            line += f"dry {med(st[:, 12] - st[:, 1]):.2f}/{mx(st[:, 12] - st[:, 1]):.2f}, "
        line += (f"push-issue {med(st[:, 13] - st[:, 4]):.2f}/{mx(st[:, 13] - st[:, 4]):.2f}, "
                 f"barrier {med(st[:, 5] - st[:, 13]):.2f}/{mx(st[:, 5] - st[:, 13]):.2f}, "
                 f"loads {med(st[:, 14] - st[:, 6]):.2f}, stores {med(st[:, 7] - st[:, 14]):.2f}, ")
        line += (f"push+wait {med(st[:, 5] - st[:, 4]):.2f}/{mx(st[:, 5] - st[:, 4]):.2f}, "
                 f"bar2 {med(st[:, 6] - st[:, 5]):.2f}/{mx(st[:, 6] - st[:, 5]):.2f}, "
                 f"final {med(st[:, 7] - st[:, 6]):.2f}/{mx(st[:, 7] - st[:, 6]):.2f}, exit {med(st[:, 8] - st[:, 7]):.2f}, "
                 f"cta {med(st[:, 8] - st[:, 1]):.2f}/{mx(st[:, 8] - st[:, 1]):.2f}")
        parts.append(line)
    log(f"dgrs stamps {layer} m{m} bm{d['BM']}s{d['S']}r2m4 (med/max us): " + " | ".join(parts))


def prepare(weights):
    t0 = time.monotonic()
    dgrs_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgrs_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for d in DGRS_DIAGS:
        lab.section(f"dgrs diag {d['layer']}", sec_dgrs_diag, d, min_left=10.0)
        lab.section(f"dgrs stamps {d['layer']}", sec_stamps, d, min_left=5.0)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgrs":
            log(f"V29 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

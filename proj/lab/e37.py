"""PE37 (P41, DG12-RS v8, pinned copy proj/dg/pinned/dgrs_v8.py): where does the finalize's ~1.3 us of slow shared-memory loads
come from? unpooler m32 and o m32, BM32 MATH 4: S 1 (plain launch, no cluster, no DSMEM: the finalize reads the CTA's own tile),
S 2 red 0 (pull: DSMEM loads) and S 2 red 2 (push: local loads after a cluster barrier), each at d0 and d64 (timestamps of the
last cold-L2 graph replay); plus (prefill's additions) d96 = stamps without the precompute (S 1 and S 2 red 2: is the wait on
the producer warps' sCorr / sScale writes?) and MATH 2 at S 2 red 2 d64 on unpooler (8 slot loads instead of 16: fixed wait
or per-load?). If S 1's loads are fast, the wait is DSMEM traffic; if they're ~1.3 us too, it's local. Winners >= 3% would
enter CHOICE ("V30 CHOICE"); none is expected."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_stamps(lab, d):
    layer, m = d["layer"], d["m"]
    n = FULL[layer][0]
    ctas = n // 128 * -(-m // d["BM"]) * d["S"]
    parts = []
    for b, mm, r in d["runs"]:
        if not b & 64:
            continue
        c = dict(BM=d["BM"], S=d["S"], CPS=d["CPS"], RED=r, MATH=mm)
        bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m) + (b,))
        if bufs is None:
            parts.append(f"r{r} d{b} m{mm}: no buffers")
            continue
        st = (bufs["part"][: 16 * ctas].view(torch.int32).view(ctas, 16).to(torch.int64) & 0xFFFFFFFF).cpu()
        u32 = lambda a: (a & 0xFFFFFFFF).double()   # noqa: E731  (differences mod 2^32)
        gspan = u32(st[:, 9] - st[:, 0])
        ghz = float((u32(st[:, 8] - st[:, 1]) / gspan.clamp(min=1)).median())
        f = 1.0 / (1e3 * ghz) if ghz > 0.5 else 1.0 / 1755.0
        med = lambda a: float(u32(a).median()) * f                 # noqa: E731
        mx = lambda a: float(u32(a).max()) * f                     # noqa: E731
        g0 = st[:, 0]
        line = (f"S{d['S']} r{r} d{b} m{mm} ({ctas} CTAs): {ghz:.3f} GHz, {len(set(st[:, 10].tolist()))} SMs; "
                f"skew {float(u32(g0 - g0.min()).max()) / 1e3:.2f}, span {float(u32(st[:, 9] - g0.min()).max()) / 1e3:.2f}; "
                f"loop0 {med(st[:, 4] - st[:, 1]):.2f}/{mx(st[:, 4] - st[:, 1]):.2f}, copies {med(st[:, 2] - st[:, 1]):.2f}, "
                f"precorr {med(st[:, 3] - st[:, 1]):.2f}, ")
        if r == 2 and d["S"] > 1:
            line += (f"push-issue {med(st[:, 13] - st[:, 4]):.2f}/{mx(st[:, 13] - st[:, 4]):.2f}, "
                     f"barrier {med(st[:, 5] - st[:, 13]):.2f}/{mx(st[:, 5] - st[:, 13]):.2f}, ")
        else:
            line += f"tile+sync {med(st[:, 5] - st[:, 4]):.2f}/{mx(st[:, 5] - st[:, 4]):.2f}, "
        line += (f"bar2 {med(st[:, 6] - st[:, 5]):.2f}, loads {med(st[:, 14] - st[:, 6]):.2f}/{mx(st[:, 14] - st[:, 6]):.2f}, "
                 f"stores {med(st[:, 7] - st[:, 14]):.2f}, final {med(st[:, 7] - st[:, 6]):.2f}, exit {med(st[:, 8] - st[:, 7]):.2f}, "
                 f"cta {med(st[:, 8] - st[:, 1]):.2f}/{mx(st[:, 8] - st[:, 1]):.2f}")
        parts.append(line)
    log(f"dgrs stamps {layer} m{m} bm{d['BM']}m4 (med/max us): " + " | ".join(parts))


def prepare(weights):
    t0 = time.monotonic()
    dgrs_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgrs_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for d in DGRS_DIAGS:
        lab.section(f"dgrs diag {d['layer']} S{d['S']}", sec_dgrs_diag, d, min_left=10.0)
        lab.section(f"dgrs stamps {d['layer']} S{d['S']}", sec_stamps, d, min_left=5.0)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgrs":
            log(f"V30 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

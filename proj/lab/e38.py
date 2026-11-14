"""PE38 (P41, DG12-RS v9, pinned copy proj/dg/pinned/dgrs_v9.py): (1) do the ring corrections (DIAG 512) remove the
precompute's ~0.9 us drag on the main loop? (2) does an instance specialized for red 2 / 2 splits (constant row share, no
runtime mode branches, other modes' code dropped) shrink the finalize's fixed cost (the code-size question)? unpooler m32 and o
m32, BM32 S2 red 2 MATH 4: d0 / d512 / d576 (= 512 + timestamps), generic and specialized; then the m32 sweep with d512,
generic and specialized (unpooler S2 r2, pooler S3 r0 / r2, o S2 r2, qkv S4 r2). Winners >= 3% enter CHOICE (fallback: v10's
entry), printed as "V31 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_stamps(lab, d):
    layer, m = d["layer"], d["m"]
    n = FULL[layer][0]
    ctas = n // 128 * -(-m // d["BM"]) * d["S"]
    parts = []
    for b, mm, r, sp in d["runs"]:
        if not b & 64:
            continue
        c = dgrs_diag_cfg(d, b, mm, r, sp)
        bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
        tag = f"d{b}{'S' if sp else 'G'}"
        if bufs is None:
            parts.append(f"{tag}: no buffers")
            continue
        st = (bufs["part"][: 16 * ctas].view(torch.int32).view(ctas, 16).to(torch.int64) & 0xFFFFFFFF).cpu()
        u32 = lambda a: (a & 0xFFFFFFFF).double()   # noqa: E731  (differences mod 2^32)
        gspan = u32(st[:, 9] - st[:, 0])
        ghz = float((u32(st[:, 8] - st[:, 1]) / gspan.clamp(min=1)).median())
        f = 1.0 / (1e3 * ghz) if ghz > 0.5 else 1.0 / 1755.0
        med = lambda a: float(u32(a).median()) * f                 # noqa: E731
        mx = lambda a: float(u32(a).max()) * f                     # noqa: E731
        g0 = st[:, 0]
        parts.append(
            f"{tag} ({ctas} CTAs): {ghz:.3f} GHz, {len(set(st[:, 10].tolist()))} SMs; skew {float(u32(g0 - g0.min()).max()) / 1e3:.2f}, "
            f"span {float(u32(st[:, 9] - g0.min()).max()) / 1e3:.2f}; loop0 {med(st[:, 4] - st[:, 1]):.2f}/{mx(st[:, 4] - st[:, 1]):.2f}, "
            f"copies {med(st[:, 2] - st[:, 1]):.2f}, push-issue {med(st[:, 13] - st[:, 4]):.2f}/{mx(st[:, 13] - st[:, 4]):.2f}, "
            f"barrier {med(st[:, 5] - st[:, 13]):.2f}/{mx(st[:, 5] - st[:, 13]):.2f}, bar2 {med(st[:, 6] - st[:, 5]):.2f}, "
            f"loads {med(st[:, 14] - st[:, 6]):.2f}/{mx(st[:, 14] - st[:, 6]):.2f}, stores {med(st[:, 7] - st[:, 14]):.2f}, "
            f"final {med(st[:, 7] - st[:, 6]):.2f}, exit {med(st[:, 8] - st[:, 7]):.2f}, cta {med(st[:, 8] - st[:, 1]):.2f}/"
            f"{mx(st[:, 8] - st[:, 1]):.2f}")
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
            log(f"V31 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

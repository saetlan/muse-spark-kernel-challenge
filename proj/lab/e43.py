"""PE43 (P43 v2: F12K HL on the router's BF16 hi + lo fragments, and F12K on aligned whole-tile grids): v11b byte for byte + the F12K v2 section
(proj/f12m/f12k.py) with PE42's winners put in CHOICE first (fallback: v11b's entry; each checked against FP64 and launched
once here, outside any capture), then F12K HL candidates on the router at m = 1 / 8 / 16 / 32 (16 row tiles of 8 MB hi + lo:
the same few-tile problem). Router winners >= 3% enter CHOICE; every F12K entry is printed as an "SK CHOICE" line, so the
check's own score measures v11b + PE42's winners + the router winners."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 170.0
PE42_WINNERS = {}   # filled from PE42's "SK CHOICE" lines by make_e43.py


def e43_targets(sms):
    """Router hi + lo (split tiles, fix-up), then aligned F12 grids: G divides the tile count, so every tile stays whole
    (pieces = 1, no fix-up) and a CTA streams G / tiles whole tiles back to back, the next tile's loads issued before the
    current epilogue (PE42 split tiles and lost on the fix-up, PF60)."""
    def c(nt, ksw, d, g, hl=False):
        out = dict(kind="f12k", NT=nt, KSW=ksw, D=d, G=g)
        if hl:
            out["HL"] = True
        return out
    return {
        ("router", 16): [c(2, 8, 2, 128, 1), c(2, 8, 2, 256, 1), c(2, 8, 2, 2 * sms, 1), c(2, 4, 2, 512, 1), c(2, 4, 2, 4 * sms, 1)],
        ("router", 8): [c(1, 8, 2, 128, 1), c(1, 8, 2, 256, 1), c(1, 4, 2, 512, 1), c(1, 8, 2, 2 * sms, 1), c(1, 4, 2, 4 * sms, 1)],
        ("router", 32): [c(4, 8, 1, 128, 1), c(4, 8, 1, 256, 1), c(4, 4, 1, 512, 1), c(4, 8, 1, 2 * sms, 1)],
        ("router", 1): [c(1, 8, 2, 128, 1), c(1, 8, 2, 256, 1), c(1, 4, 2, 512, 1)],
        ("o", 8): [c(1, 8, 2, 256), c(1, 8, 2, 128), c(1, 4, 4, 256), c(1, 4, 4, 512)],
        ("o", 16): [c(2, 8, 2, 256), c(2, 8, 2, 128), c(2, 4, 4, 256), c(2, 4, 4, 512)],
        ("o", 1): [c(1, 8, 2, 256), c(1, 8, 2, 128), c(1, 4, 4, 512)],
        ("pooler", 8): [c(1, 8, 2, 128), c(1, 4, 4, 256), c(1, 8, 2, 256)],
        ("pooler", 16): [c(2, 8, 2, 128), c(2, 4, 4, 256), c(2, 8, 2, 256)],
        ("unpooler", 16): [c(2, 8, 2, 256), c(2, 8, 2, 128), c(2, 4, 4, 512)],
        ("qkv_gate", 16): [c(2, 8, 2, 72), c(2, 4, 4, 144)],
        ("qkv_gate", 8): [c(1, 8, 2, 72), c(1, 4, 4, 144)],
    }


F12K_LAB_INSTANCES = tuple(sorted({f12k_inst(c) for cs in e43_targets(132).values() for c in cs}
                                  | {f12k_inst(c) for c in PE42_WINNERS.values()}, key=str))


def e43_commit_winners(state):
    """PE42's winners into CHOICE: each checked on fresh normal / outlier / probe inputs against FP64 (lab_ok) and launched
    once here; a winner that fails keeps v11b's entry."""
    for (layer, m), c in sorted(PE42_WINNERS.items()):
        n, k = lab_n(layer, m), FULL[layer][1]
        cand = dict(c, fallback=CHOICE.get((layer, m)))
        oks = []
        for seed, dist, probe in ((5100 + m, "normal", False), (5200 + m, "outliers", False), (5300 + m, "normal", True)):
            x = lab_input(m, k, seed, dist=dist, probe_all=probe)
            oks.append(lab_ok(lab_errors(run_choice(cand, x, layer, n, state), x, state[layer][:n]), layer))
        if all(oks):
            CHOICE[(layer, m)] = cand
        log(f"PE42 winner {layer} m{m} {cfg_name(cand)}: " + ("in CHOICE" if all(oks) else f"WRONG {oks} -> keep v11b's entry"))
    torch.cuda.synchronize()


def prepare(weights):
    t0 = time.monotonic()
    f12k_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v11b) took {time.monotonic() - t0:.1f}s")
    targets = e43_targets(torch.cuda.get_device_properties(0).multi_processor_count)
    f12k_setup_lab(state, t0, {**targets, **{k: [c] for k, c in PE42_WINNERS.items()}})
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    if F12K_MOD is not None:
        lab.section("pe42 winners", lambda lab_: e43_commit_winners(state), min_left=-1e9)
    sweep = Sweep(state)
    for layer, m in targets:
        lab.section(f"f12k {layer} m{m}", sec_f12k, sweep, targets, layer, m, min_left=8.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    lab.section("f12k counters", sec_f12k_cnt, min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12k":
            log(f"SK CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

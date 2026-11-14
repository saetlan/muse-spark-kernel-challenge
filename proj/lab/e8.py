"""PE8: F12V (12-bit lossless weights decoded to FP32 in registers, CUDA-core GEMV) at m = 1 on the four
BF16 layers, timed by the judge on top of v4. In prepare(): v4's prepare plus f12v_setup (pack + nvcc),
registers per instance and escape statistics, then F12V_LAB_CANDIDATES per layer through the gate
(FP64, probe rows, all-probe input, outliers) against SGLang and v4's choice; winners >= 3% faster go
into CHOICE (keeping v4's kernel as their fallback) and are printed as "V5 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_f12v_info(lab, state):
    rows = []
    for inst in F12V_INSTANCES:
        v = int(getattr(F12V_MOD, f12v_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("f12v regs (R/RG/KSW/NS/D): " + " ".join(rows))
    log("f12v escapes: " + ", ".join(f"{name} {pl['rate'] * 100:.4f}% max {pl['max_row']}/row"
                                     for name, pl in state["f12v"].items())
        + f"; layers without planes: {[n for n in F12V_LAYERS if n not in state['f12v']]}")


def sec_f12v(lab, sweep, layer):
    fallback = CHOICE.get((layer, 1))
    cands = [dict(kind="f12v", fallback=fallback, **c) for c in F12V_LAB_CANDIDATES.get(layer, [])]
    if cands and layer in sweep.state["f12v"]:
        sweep.shape(lab, layer, 1, cands)
        rec = sweep.results[(layer, 1)]
        n, k = lab_n(layer, 1), FULL[layer][1]
        times = " ".join(f"{c['R']}/{c['RG']}/{c['KSW']}/{c['D']}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
        log(f"f12v {layer} m1 (sglang {rec['sglang']:.2f}, v4 {rec['cur_us']:.2f} us; 12-bit bytes {n * k * 1.5 / 1e6:.1f} MB) "
            f"R/RG/KSW/D:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"v4 prepare + f12v_setup ({len(F12V_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    lab = Lab(budget)
    sweep = Sweep(state)
    lab.section("f12v info", sec_f12v_info, state)
    for layer in ("pooler", "qkv_gate", "o", "unpooler"):
        lab.section(f"f12v {layer}", sec_f12v, sweep, layer, min_left=15.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] == "f12v":
            log(f"V5 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

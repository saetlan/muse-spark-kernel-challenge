"""PE27 (P40 diagnosis, PF44): where DG12's ~0.8 us per 64-k stage goes. The same kernel with parts removed (DIAG bits:
1 = no decode, 2 = no raw-plane TMA, 4 = no proxy fence) on unpooler m128 128x64 s7 and pooler m128 64x64 s9, each timed
like the sweep (CUDA graph, cold L2); DIAG 0 and 4 are also checked against FP64 (the others are wrong by design). Nothing
enters CHOICE."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DG12_BUILD_DEADLINE_S = 245.0
DG12_MOD = None
DG12_OK = set()


def dg12_inst(c, n, k):
    return (c["BM"], c["BN"], c["ST"], c["SWD"], c["NMATH"], n, k, c.get("DIAG", 0))


def dg12_setup_lab(state, t_start):
    global DG12_MOD
    state["dg12"] = {}
    for layer in sorted({layer for (layer, m) in DG12_DIAG}):
        pk = dg12_pack(state[layer])
        if pk is not None:
            state["dg12"][layer] = {key: pk[key] for key in ("qc", "lc", "scale", "offs8", "ent8")}
    insts = tuple(dict.fromkeys(dg12_inst(c, FULL[layer][0], FULL[layer][1]) for (layer, m), cs in DG12_DIAG.items()
                                for c in cs))
    t0 = time.monotonic()
    mod, errors = dg12_build(insts, timeout=max(10.0, DG12_BUILD_DEADLINE_S - (time.monotonic() - t_start)))
    for name, exc in list(errors.items())[:2]:
        errs = [ln.strip() for ln in str(exc).splitlines() if "error" in ln.lower()][:4]
        log(f"dg12 {name} FAILED ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs))
    ok = [inst for inst in insts if dg12_name(inst) not in errors]
    for inst in ok:
        getattr(mod, dg12_name(inst) + "_attr")()
    log(f"dg12 build: {len(ok)}/{len(insts)} instances ok in {time.monotonic() - t0:.1f}s")
    if ok:
        DG12_MOD = mod
        DG12_OK.update(dg12_name(inst) for inst in ok)


def sec_diag(lab, state, layer, m):
    n, k = FULL[layer]
    pk = state["dg12"].get(layer)
    if DG12_MOD is None or pk is None:
        return
    x = lab_input(m, k, 4200 + m)
    w = state[layer][:n]
    ref_us = lab.time_graph(lambda: run_choice(CHOICE.get((layer, m)), x, layer, n, state)) if CHOICE.get((layer, m)) \
        else lab.time_graph(lambda: sglang_path(x, layer, n, state))
    parts = []
    for c in DG12_DIAG[(layer, m)]:
        name = dg12_name(dg12_inst(c, n, k))
        if name not in DG12_OK:
            parts.append(f"d{c['DIAG']}:FAILED")
            continue
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        fn = getattr(DG12_MOD, name)

        def run(fn=fn, out=out):
            fn(x, pk["qc"], pk["lc"], pk["scale"], pk["offs8"], pk["ent8"], out)
            return out
        try:
            run()
            us = lab.time_graph(run)
            tag = ""
            if c["DIAG"] in (0, 4):
                errs = lab_errors(run(), x, w)
                tag = "ok" if lab_ok(errs, layer) else f"WRONG {errs[0]:.1e}"
            parts.append(f"d{c['DIAG']}:{us:.2f}{('(' + tag + ')') if tag else ''}")
        except Exception as exc:
            parts.append(f"d{c['DIAG']}:ERR {type(exc).__name__}")
    c0 = DG12_DIAG[(layer, m)][0]
    log(f"diag {layer} m{m} {c0['BM']}x{c0['BN']}s{c0['ST']} (base {cfg_name(CHOICE.get((layer, m)))} {ref_us:.2f} us): "
        + " ".join(parts) + "  [bits: 1 no decode, 2 no raw TMA, 4 no fence]")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dg12_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    for layer, m in DG12_DIAG:
        lab.section(f"diag {layer} m{m}", sec_diag, state, layer, m, min_left=6.0)
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

"""PE26 (P40 step 2c): DG12 with the batched shared-memory decoder (PF43) and BM 128 variants, on chunk-major raw planes (PF42: a stage's raw data = one contiguous block, TMA'd as 128-B
rows), otherwise as PE24: DeepGEMM's sm90 BF16 GEMM with the 12-bit weight load (proj/dg/dg12.py; planes from
proj/dg/dg12_layout.py packed in prepare()), plain launch (PF41), on all four BF16 layers at m = 32/128/256 with
DeepGEMM's block shapes (stages recomputed for the raw planes) plus a smaller-BN alternative at m = 128/256; timed
next to the base (v10: cuBLAS / DeepGEMM / Triton). Winners >= 3% enter CHOICE with the base kernel as fallback,
printed as "V21 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DG12_BUILD_MAX_START_S = 160.0
DG12_BUILD_DEADLINE_S = 245.0
DG12_MOD = None
DG12_OK = set()


def dg12_inst(c, n, k):
    return (c["BM"], c["BN"], c["ST"], c["SWD"], c["NMATH"], n, k)


def dg12_setup_lab(state, t_start):
    global DG12_MOD
    state["dg12"] = {}
    t0 = time.monotonic()
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        try:
            pk = dg12_pack(state[layer])
        except Exception as exc:
            log(f"dg12 pack {layer} FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-200:]}")
            pk = None
        if pk is not None:
            state["dg12"][layer] = {key: pk[key] for key in ("qc", "lc", "scale", "offs8", "ent8")}
    log(f"dg12 pack: {sorted(state['dg12'])} in {time.monotonic() - t0:.1f}s; escapes "
        + ", ".join(f"{k} {int(v['ent8'].shape[0])}" for k, v in state["dg12"].items()))
    if time.monotonic() - t_start > DG12_BUILD_MAX_START_S:
        log(f"dg12 build skipped: {time.monotonic() - t_start:.0f}s already used")
        return
    insts = tuple(dict.fromkeys(dg12_inst(c, FULL[layer][0], FULL[layer][1]) for (layer, m), cs in DG12_TARGETS.items()
                                for c in cs))
    t0 = time.monotonic()
    try:
        mod, errors = dg12_build(insts, timeout=max(10.0, DG12_BUILD_DEADLINE_S - (time.monotonic() - t_start)))
    except Exception as exc:
        log(f"dg12 build FAILED to start ({type(exc).__name__}): {' '.join(str(exc).split())[-500:]}")
        return
    shown = 0
    for name, exc in errors.items():
        text = str(exc)
        if shown < 2:
            errs = [ln.strip() for ln in text.splitlines() if "error" in ln.lower()][:5]
            log(f"dg12 {name} FAILED ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs)
                + " || tail: " + " ".join(text.split())[-500:])
        shown += 1
    ok = [inst for inst in insts if dg12_name(inst) not in errors]
    regs = []
    for inst in ok:
        try:
            getattr(mod, dg12_name(inst) + "_attr")()
            v = int(getattr(mod, dg12_name(inst) + "_info")())
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}:{v // 1000000}r" + (f"+{v % 1000000}B" if v % 1000000 else ""))
        except Exception as exc:
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}: attr FAILED {' '.join(str(exc).split())[-120:]}")
    log(f"dg12 build: {len(ok)}/{len(insts)} instances ok in {time.monotonic() - t0:.1f}s; regs {' '.join(regs)}")
    if ok:
        DG12_MOD = mod
        DG12_OK.update(dg12_name(inst) for inst in ok)


def run_dg12(c, x, layer, n, state):
    m, k = x.shape
    pk = state.get("dg12", {}).get(layer)
    if DG12_MOD is None or pk is None or dg12_name(dg12_inst(c, n, k)) not in DG12_OK or n != FULL[layer][0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(DG12_MOD, dg12_name(dg12_inst(c, n, k)))(x.contiguous(), pk["qc"], pk["lc"], pk["scale"], pk["offs8"], pk["ent8"], out)
    return out


_run_choice_pre_dg12 = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dg12":
        return run_dg12(c, x, layer, n, state)
    return _run_choice_pre_dg12(c, x, layer, n, state)


def sec_dg12(lab, sweep, layer, m):
    if DG12_MOD is None or layer not in sweep.state.get("dg12", {}):
        return
    n, k = FULL[layer]
    cands = [dict(kind="dg12", fallback=CHOICE.get((layer, m)), **c) for c in DG12_TARGETS[(layer, m)]
             if dg12_name(dg12_inst(c, n, k)) in DG12_OK]
    if not cands:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = n * k * 1.5
    times = " ".join(f"{c['BM']}x{c['BN']}s{c['ST']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"dg12 {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) BMxBN s stages:us(12-bit TB/s) {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dg12_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in DG12_TARGETS:
        lab.section(f"dg12 {layer} m{m}", sec_dg12, sweep, layer, m, min_left=6.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dg12":
            log(f"V21 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state



# ---------------------------------------------------------------------------------------------
# Lab L14 (check runs only): S2c (cuda_ver3.py: S2b with the producer's page entries carried by
# the cp.async ring, make_s2c.py) vs S2b (cuda_ver2.py) between the Triton prologue and _combine,
# on the six verify shapes against the Triton path and SGLang: error vs the FP32 reference,
# graph timing with a cold L2.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
_MODS = {}


def _build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


def _ver_nsplit(cfg, max_pages, ctas_per_sm):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // 64, MAX_SPLITS))



def _ver2_run(data, ctas_per_sm, stages, kind="ver3"):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    device = qkv.device
    tokens = batch * length
    max_pages = data["page_table"].shape[1]
    nsplit = _ver_nsplit(cfg, max_pages, ctas_per_sm)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(tokens,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                         q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                         HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    getattr(_MODS[kind], "attend_" + kind)(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                              data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                              cfg["page_size"], cfg["window"], cfg["sm_scale"], stages)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1414)
    try:
        for kind, decl, src in (("ver3", CUDA_VER3_DECL, CUDA_VER3_SRC), ("ver2", CUDA_VER2_DECL, CUDA_VER2_SRC)):
            _MODS[kind] = _build("okc_l14_" + kind, decl, src,
                                 ["attend_" + kind, "attend_" + kind + "_setup", "attend_" + kind + "_attrs"])
            rc = getattr(_MODS[kind], "attend_" + kind + "_setup")()
            _log(f"{kind} setup {rc}{'  <-- refused' if rc else ''}")
            for st in (2, 3, 4):
                a = getattr(_MODS[kind], "attend_" + kind + "_attrs")(st)
                _log(f"{kind} s{st} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 8)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L14 start: budget {_LAB_BUDGET_S:.0f}s; line = shape variant: us TB/s err (nsplit)")
    variants = [(2, 3), (4, 3), (2, 4), (4, 4), (8, 3), (3, 3), (6, 3)]
    anchor = {16: (4, 4), 32: (8, 3), 1: (1, 3)}      # S2b's best per batch in L11
    for batch, window in [(16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)]:
        name = f"{'S' if window else 'F'}v{batch}"
        if lab.left() < 20:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, 8, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv = float((data["cache_seqlens"].clamp(max=window + 8) if window else data["cache_seqlens"]).sum()) * 256
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} triton path: {us:.1f}us {kv / us / 1e6:.2f}TB/s")
            us = lab.time_graph(lambda: sglang_path(data))
            _log(f"{name} sglang path: {us:.1f}us {kv / us / 1e6:.2f}TB/s")
            run(data)                                  # our new rows back in the cache
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for kind, cps, stages in [("ver2",) + anchor[batch]] + [("ver3", c, st) for c, st in variants]:
            tag = f"{'S2b' if kind == 'ver2' else 'S2c'} c{cps} s{stages}"
            try:
                ns = _ver_nsplit(data["config"], data["page_table"].shape[1], cps)
                err = float((_ver2_run(data, cps, stages, kind).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ver2_run(data, cps, stages, kind))
                _log(f"{name} {tag}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
            except Exception as exc:
                _log(f"{name} {tag}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

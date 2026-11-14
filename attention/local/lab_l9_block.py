

# ---------------------------------------------------------------------------------------------
# Lab L9 (check runs only): D2 = A5b's compute fed from a warp-private cp.async ring
# (attention/local/cuda_dec3.py) against A5b (cuda_dec2.py) and the submission's path on the six
# decode shapes: error vs the FP32 reference, graph timing with a cold L2, warps x stages x CTAs.
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


def _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _dec2_run(data, ctas_per_sm, warps):
    """Triton prologue, A5b attention, Triton combine (q_len 1, groups of 8)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _MODS["main"].attend_dec2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                              data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                              cfg["page_size"], cfg["window"], cfg["sm_scale"], warps)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out



def _dec3_run(data, ctas_per_sm, warps, stages):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _MODS["dec3"].attend_dec3(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                              data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                              cfg["page_size"], cfg["window"], cfg["sm_scale"], warps, stages)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(9009)
    try:
        _MODS["main"] = _build("okc_l9_dec2", CUDA_DEC2_DECL, CUDA_DEC2_SRC, ["attend_dec2", "attend_dec2_attrs"])
        _MODS["dec3"] = _build("okc_l9_dec3", CUDA_DEC3_DECL, CUDA_DEC3_SRC,
                               ["attend_dec3", "attend_dec3_setup", "attend_dec3_attrs"])
        rc = _MODS["dec3"].attend_dec3_setup()
        _log(f"dec3 setup {rc}{'  <-- smem limit refused: D2 variants skipped' if rc else ''}")
        if rc:
            _MODS.pop("dec3")
        for w, st in ((4, 3), (4, 4), (8, 2), (8, 3)) if "dec3" in _MODS else ():
            a = _MODS["dec3"].attend_dec3_attrs(w, st)
            _log(f"D2 w{w} s{st} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L9 start: budget {_LAB_BUDGET_S:.0f}s")
    variants = [(4, 4, 3), (8, 4, 3), (16, 4, 3), (4, 4, 4), (8, 4, 4), (4, 8, 2), (8, 8, 2), (4, 8, 3), (8, 8, 3)]
    for batch, window in [(16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)]:
        name = f"{'S' if window else 'F'}d{batch}"
        if lab.left() < 20:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} v2.2 path: {us:.1f}us {kv / us / 1e6:.2f}TB/s")
            best = (8, 4) if (batch, window) == (32, 0) else (4, 4)
            us = lab.time_graph(lambda: _dec2_run(data, *best))
            _log(f"{name} A5b c{best[0]} w{best[1]}: {us:.1f}us {kv / us / 1e6:.2f}TB/s")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for cps, warps, stages in variants if "dec3" in _MODS else []:
            try:
                err = float((_dec3_run(data, cps, warps, stages).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _dec3_run(data, cps, warps, stages))
                _log(f"{name} D2 c{cps} w{warps} s{stages}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
            except Exception as exc:
                _log(f"{name} D2 c{cps} w{warps} s{stages}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

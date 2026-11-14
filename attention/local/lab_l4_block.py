

# ---------------------------------------------------------------------------------------------
# Lab L4 (check runs only): A5, the CUDA decode kernel (attention/local/cuda_dec.py, appended
# above), against the submission's Triton path on the six decode shapes at the scored sizes.
# Per variant: error vs the FP32 reference, time of prologue + attend + combine under a CUDA
# graph with a cold L2, KV bandwidth. Build once with load_inline (sm_90a).
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
_DEC = []


def _dec_build():
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name="okc_attn_dec_l4", cpp_sources=CUDA_DEC_DECL, cuda_sources=CUDA_DEC_SRC,
                      functions=["attend_dec", "attend_dec_attrs"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix="okc_l4_"), verbose=False)
    _log(f"build {_time.monotonic() - t:.1f}s; attrs (regs, local B, smem B): "
         f"w4 {mod.attend_dec_attrs(4)} w8 {mod.attend_dec_attrs(8)}")
    _DEC.append(mod)


def _dec_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _dec_run(data, ctas_per_sm, warps):
    """run() with the CUDA decode kernel in place of _attend (q_len 1, groups of 8)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _DEC[0].attend_dec(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                       data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                       cfg["page_size"], cfg["window"], cfg["sm_scale"], warps)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4004)
    _dec_build()
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L4 start: budget {_LAB_BUDGET_S:.0f}s; line = shape variant: us TB/s err-vs-FP32 (nsplit)")
    shapes = [(16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)]
    variants = [("cuda c2 w4", 2, 4), ("cuda c4 w4", 4, 4), ("cuda c8 w4", 8, 4), ("cuda c16 w4", 16, 4),
                ("cuda c2 w8", 2, 8), ("cuda c4 w8", 4, 8), ("cuda c8 w8", 8, 8)]
    for batch, window in shapes:
        name = f"{'S' if window else 'F'}d{batch}"
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv_bytes = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} triton TUNE: {us:.1f}us {kv_bytes / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for label, cps, warps in variants:
            if lab.left() < 10:
                _log("out of budget")
                break
            try:
                ns = _dec_nsplit(data["config"], data["page_table"].shape[1], cps, warps)
                out = _dec_run(data, cps, warps).float()
                err = float((out - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _dec_run(data, cps, warps))
                _log(f"{name} {label}: {us:.1f}us {kv_bytes / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
            except Exception as exc:
                _log(f"{name} {label}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

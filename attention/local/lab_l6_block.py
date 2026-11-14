

# ---------------------------------------------------------------------------------------------
# Lab L6 (check runs only): S1b = one fused CUDA kernel per decode layer
# (attention/local/cuda_fused.py, appended above: prologue fold + A5b attention + last-CTA
# ticket merge + norm-gate) on the six decode shapes against the submission's path, with the
# error vs the FP32 reference and graph timing with a cold L2. Tickets: a persistent zeroed
# int32 buffer the kernel resets itself after every call (graph-safe).
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 300.0
_MODS = {}
_TICKETS = []


def _build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


def _fused_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _fused_run(data, ctas_per_sm, warps):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    device = data["qkv"].device
    if not _TICKETS:
        _TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=device))
    max_pages = data["page_table"].shape[1]
    nsplit = _fused_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _MODS["fused"].fused_dec(data["qkv"], data["positions"], data["cache_loc"],
                             _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"],
                             data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                             data["cache_seqlens"], o_part, m_part, l_part, _TICKETS[0], out, batch, hq, hkv,
                             max_pages, nsplit, cfg["page_size"], cfg["window"], int(rope),
                             cfg["qk_scale_factor"], cfg["sm_scale"], warps)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(6006)
    try:
        _MODS["fused"] = _build("okc_l6_fused", CUDA_FUSED_DECL, CUDA_FUSED_SRC, ["fused_dec", "fused_dec_attrs"])
        m = _MODS["fused"]
        _log(f"fused attrs (regs, local B, smem B): w4 {m.fused_dec_attrs(4)} w8 {m.fused_dec_attrs(8)}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L6 start: budget {_LAB_BUDGET_S:.0f}s; line = shape variant: us TB/s err-vs-FP32 (nsplit)")
    variants = [(2, 4), (4, 4), (8, 4), (2, 8), (4, 8), (8, 8)]
    for batch, window in [(1, 2048), (16, 2048), (32, 2048), (16, 0), (32, 0), (1, 0)]:
        name = f"{'S' if window else 'F'}d{batch}"
        if lab.left() < 20:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} v2.2 path: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for cps, warps in variants:
            try:
                ns = _fused_nsplit(data["config"], data["page_table"].shape[1], cps, warps)
                # The fused kernel writes the new rows itself: re-run it, then compare with the
                # reference computed from the cache (the rows it wrote must match the prologue's).
                out = _fused_run(data, cps, warps).float()
                err = float((out - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _fused_run(data, cps, warps))
                tick = int(_TICKETS[0][: batch * (data["config"]["num_kv_heads"] // 2)].abs().sum().item())
                _log(f"{name} fused c{cps} w{warps}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns}) tickets {tick}")
            except Exception as exc:
                _log(f"{name} fused c{cps} w{warps}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()



# ---------------------------------------------------------------------------------------------
# Lab L8 (check runs only): A5b and the S1b fused kernel after the local-memory fix (the
# accumulator was indexed with a lane-dependent value at the merge, so it lived in local memory:
# L5 reported 256 B of local memory for A5b). Both on the six decode shapes against the
# submission's path, error vs the FP32 reference, graph timing with a cold L2. Any kernel that
# still reports local memory is flagged.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
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
    gen = torch.Generator(device=dev).manual_seed(8008)
    try:
        _MODS["main"] = _build("okc_l8_dec2", CUDA_DEC2_DECL, CUDA_DEC2_SRC, ["attend_dec2", "attend_dec2_attrs"])
        _MODS["fused"] = _build("okc_l8_fused", CUDA_FUSED_DECL, CUDA_FUSED_SRC, ["fused_dec", "fused_dec_attrs"])
        for name, attrs in (("A5b", _MODS["main"].attend_dec2_attrs), ("fused", _MODS["fused"].fused_dec_attrs)):
            for w in (4, 8):
                a = attrs(w)
                _log(f"{name} w{w} attrs (regs, local B, smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L8 start: budget {_LAB_BUDGET_S:.0f}s")
    variants = [(2, 4), (4, 4), (8, 4), (4, 8), (8, 8)]
    for batch, window in [(16, 0), (32, 0), (1, 2048), (16, 2048), (32, 2048), (1, 0)]:
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
        for label, fn in (("A5b", _dec2_run), ("fused", _fused_run)):
            for cps, warps in variants:
                try:
                    out = fn(data, cps, warps).float()
                    err = float((out - ref).norm() / ref.norm())
                    us = lab.time_graph(lambda: fn(data, cps, warps))
                    _log(f"{name} {label} c{cps} w{warps}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                except Exception as exc:
                    _log(f"{name} {label} c{cps} w{warps}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

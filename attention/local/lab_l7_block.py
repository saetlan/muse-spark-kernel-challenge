

# ---------------------------------------------------------------------------------------------
# Lab L7 (check runs only): S2a = the CUDA verify kernel (attention/local/cuda_ver.py, appended
# above) between the Triton prologue and _combine, on the six verify shapes (q_len 8), against
# both the submission's Triton path (run) and SGLang's path (sglang_path, what v2.2 uses on Fv16,
# Fv32, Sv32), with the error vs the FP32 reference and graph timing with a cold L2.
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


def _ver_run(data, ctas_per_sm, stages):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert length == 8 and hq // hkv == 8 and hkv % 2 == 0
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
    _MODS["ver"].attend_ver(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                            data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                            cfg["page_size"], cfg["window"], cfg["sm_scale"], stages)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7007)
    try:
        _MODS["ver"] = _build("okc_l7_ver", CUDA_VER_DECL, CUDA_VER_SRC, ["attend_ver", "attend_ver_attrs"])
        m = _MODS["ver"]
        _log(f"ver attrs (regs, local B, smem B): s3 {m.attend_ver_attrs(3)} s4 {m.attend_ver_attrs(4)} "
             f"s6 {m.attend_ver_attrs(6)}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 8)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L7 start: budget {_LAB_BUDGET_S:.0f}s; line = shape variant: us TB/s err-vs-FP32 (nsplit)")
    variants = [(2, 3), (4, 3), (8, 3), (2, 4), (4, 4), (4, 6), (8, 6)]
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
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} triton path: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
            err = float((sglang_path(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: sglang_path(data))
            _log(f"{name} sglang path: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for cps, stages in variants:
            try:
                ns = _ver_nsplit(data["config"], data["page_table"].shape[1], cps)
                err = float((_ver_run(data, cps, stages).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ver_run(data, cps, stages))
                _log(f"{name} cuda c{cps} s{stages}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
            except Exception as exc:
                _log(f"{name} cuda c{cps} s{stages}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    try:
        _sec_fused(lab, k_cache, v_cache, gen)
    except Exception as exc:
        _log(f"fused section FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()


# ---- L7 part 2: the S1b fused decode kernel after the AF20 fix, the RoPE table and the CTA-wide
# merge (cuda_fused.py appended above), on the sliding and bs16/32 full decode shapes (<= 128 slices).
_TICKETS = []


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
    assert nsplit <= 128, nsplit   # the fused merge's limit
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




def _sec_fused(lab, k_cache, v_cache, gen):
    _MODS["fused"] = _build("okc_l7_fused", CUDA_FUSED_DECL, CUDA_FUSED_SRC, ["fused_dec", "fused_dec_attrs"])
    for w in (4, 8):
        a = _MODS["fused"].fused_dec_attrs(w)
        _log(f"fused w{w} attrs (regs, local B, smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    for batch, window in [(1, 2048), (16, 2048), (32, 2048), (16, 0), (32, 0)]:
        name = f"{'S' if window else 'F'}d{batch}"
        if lab.left() < 15:
            _log("fused: out of budget")
            return
        data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
        run(data)
        ref = _lab_reference(data)
        kv = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
        us = lab.time_graph(lambda: run(data))
        _log(f"{name} v2.2 path: {us:.1f}us {kv / us / 1e6:.2f}TB/s")
        for cps, warps in ((2, 4), (4, 4), (8, 4), (4, 8), (8, 8)):
            try:
                if _fused_nsplit(data["config"], data["page_table"].shape[1], cps, warps) > 128:
                    continue
                err = float((_fused_run(data, cps, warps).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _fused_run(data, cps, warps))
                tick = int(_TICKETS[0][: batch].abs().sum().item())
                _log(f"{name} fused c{cps} w{warps}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} tickets {tick}")
            except Exception as exc:
                _log(f"{name} fused c{cps} w{warps}: FAILED {_why(exc)}")



# =============================================================================================
# v3: CUDA kernels (attention/local/cuda_dec2.py, cuda_fused.py, cuda_ver.py, appended above)
# with hybrid dispatch. One load_inline module (sm_90a), built on the first eager call next to
# the Triton warm-up; if the build fails, every shape keeps its v2.2 path (Triton or SGLang).
# CUDA_ROUTE maps a scored shape class (full layer, verify block, batch class) to its CUDA
# kernel and launch settings, from the labs' judge-H200 timings; classes not listed stay on v2.2.
# =============================================================================================
import tempfile as _tempfile

_CUDA = {}
_TICKETS = []

# class -> (kernel, ctas per SM, warps or stages), from the labs' judge-H200 timings:
# L8 (AE10): Fd16 A5b 167.8 us vs v2.2 183.7; Fd32 A5b 311.2 vs 369.1.
CUDA_ROUTE = {
    (True, False, 16): ("dec2", 4, 4),
    (True, False, 32): ("dec2", 8, 4),
}


def _cuda_build():
    from torch.utils.cpp_extension import load_inline
    src = (CUDA_DEC2_SRC + "\n" + CUDA_FUSED_SRC + "\n" + CUDA_VER_SRC)
    decl = CUDA_DEC2_DECL + "\n" + CUDA_FUSED_DECL + "\n" + CUDA_VER_DECL
    try:
        _CUDA["mod"] = load_inline(
            name="okc_attention_v3", cpp_sources=decl, cuda_sources=src,
            functions=["attend_dec2", "attend_dec2_attrs", "fused_dec", "fused_dec_attrs", "attend_ver",
                       "attend_ver_attrs"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=_tempfile.mkdtemp(prefix="okc_attention_v3_"), verbose=False)
    except Exception as exc:  # keep every shape on its v2.2 path
        print(f"v3: CUDA build failed, staying on v2.2 paths: {type(exc).__name__}: {str(exc)[-300:]}")


def _nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // min_keys, MAX_SPLITS))


def _prologue_q(data):
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = cfg["batch"] * cfg["q_len"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(tokens,)](data["qkv"], data["positions"], data["cache_loc"],
                         _inv_freq(cfg["rope_theta"] if rope else 1.0, device), q, data["k_cache"], data["v_cache"],
                         data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, ROPE=rope,
                         QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    return q


def _merge(data, o_part, m_part, l_part, nsplit):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    _combine[(batch * length, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit,
                                   HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)),
                                   num_warps=4)
    return out


def _cuda_run(data, route):
    kind, cps, knob = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    max_pages = data["page_table"].shape[1]
    mod = _CUDA["mod"]
    if kind == "fused":
        nsplit = _nsplit(cfg, max_pages, cps, 32 * knob)
        o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        rope = bool(cfg["rope_theta"])
        mod.fused_dec(data["qkv"], data["positions"], data["cache_loc"],
                      _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                      data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], o_part, m_part,
                      l_part, _TICKETS[0], out, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                      int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], knob)
        return out
    q = _prologue_q(data)
    if kind == "dec2":
        nsplit = _nsplit(cfg, max_pages, cps, 32 * knob)
        o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        mod.attend_dec2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                        data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                        cfg["page_size"], cfg["window"], cfg["sm_scale"], knob)
    else:  # "ver"
        nsplit = _nsplit(cfg, max_pages, cps, 64)
        tokens = batch * length
        o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        mod.attend_ver(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                       data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                       cfg["page_size"], cfg["window"], cfg["sm_scale"], knob)
    return _merge(data, o_part, m_part, l_part, nsplit)


def _cuda_ok(cfg, route):
    """The CUDA kernels need groups of 8 query heads, paired KV heads, q_len 1 (fused/dec2) or 8
    (ver), and the fused merge at most 128 slices."""
    kind, cps, knob = route
    if cfg["num_q_heads"] // cfg["num_kv_heads"] != 8 or cfg["num_kv_heads"] % 2:
        return False
    if kind in ("fused", "dec2") and cfg["q_len"] != 1:
        return False
    if kind == "ver" and cfg["q_len"] != 8:
        return False
    return True


_kernel_fn_v22 = kernel_fn
_warm_up_v22 = _warm_up


def _warm_up():  # noqa: F811 (v3: Triton warm-up, then the CUDA build and a launch of every routed class)
    _warm_up_v22()
    _cuda_build()
    if "mod" not in _CUDA:
        return
    dev = torch.device("cuda")
    _TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    for cls, route in CUDA_ROUTE.items():
        full, verify, bclass = cls
        length = 8 if verify else 1
        data = dummy_inputs(bclass, length, 0 if full else 2048, scored_max_pages(bclass, length), dev)
        _cuda_run(data, route)


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    route = CUDA_ROUTE.get(shape_class(cfg))
    if route is not None and "mod" in _CUDA and _cuda_ok(cfg, route):
        return _cuda_run(data, route)
    return _kernel_fn_v22(data)

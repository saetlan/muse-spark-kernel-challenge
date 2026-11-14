

# =============================================================================================
# v4: CUDA kernels with hybrid dispatch over v2.2 (Triton or SGLang paths, above). Kinds:
#   "dec2": A5b, register-direct decode (cuda_dec2.py), between the Triton prologue and combine;
#   "dec3": D2, per-warp cp.async ring decode (cuda_dec3.py), same glue;
#   "dec4": D3, D2 with page entries carried by the ring (cuda_dec4.py, make_pt_ring.py), same glue;
#   "ud":   UD, one launch per decode layer (cuda_unified.py): query fold, ring streaming,
#           new-row patch, cluster DSMEM merge, second level, RMSNorm x gate, BF16 out;
#   "ver2": S2b, verify with a producer warp and mbarrier stages (cuda_ver2.py), Triton glue.
# One load_inline module (sm_90a) holds only the routed kinds, built on the first eager call
# next to the Triton warm-up. Each route is then launched once on synthetic inputs of its class
# (random FP8 cache, our own data) and compared with the Triton path; a route whose build,
# setup, launch or result fails is dropped, so its class keeps its v2.2 path.
# CUDA_ROUTE maps a scored class (full layer, verify block, batch class) to its kernel and
# launch settings, from the labs' judge-H200 timings; classes not listed stay on v2.2.
# =============================================================================================
import tempfile as _tempfile

_CUDA = {}
_TICKETS = []

# class -> (kind, settings...):
#   ("dec2", ctas per SM, warps) | ("dec3" or "dec4", ctas per SM, warps, stages)
#   ("ud", cluster size, ctas per SM, warps, stages) | ("ver2", ctas per SM, stages)
CUDA_ROUTE = {
    (True, False, 16): ("dec4", 6, 4, 3),   # L12 (AE17): D3 125.8 us vs D2 155.6 (L9b), v2.2 182.3
    (True, False, 32): ("dec4", 6, 4, 3),   # L12 (AE17): D3 214.0 us vs D2 263.3 (L9b), v2.2 372.2
}

_KINDS = {   # kind -> (CUDA source, declarations, functions, setup function or None)
    "dec2": (CUDA_DEC2_SRC, CUDA_DEC2_DECL, ["attend_dec2"], None),
    "dec3": (CUDA_DEC3_SRC, CUDA_DEC3_DECL, ["attend_dec3", "attend_dec3_setup"], "attend_dec3_setup"),
    "dec4": (CUDA_DEC4_SRC, CUDA_DEC4_DECL, ["attend_dec4", "attend_dec4_setup"], "attend_dec4_setup"),
    "ud": (CUDA_UD_SRC, CUDA_UD_DECL, ["ud", "ud_setup"], "ud_setup"),
    "ver2": (CUDA_VER2_SRC, CUDA_VER2_DECL, ["attend_ver2", "attend_ver2_setup"], "attend_ver2_setup"),
}
SELF_CHECK_TOL = 2e-2   # relative L2 vs the Triton path (both ~2e-3 from FP32; a bug is O(1))
_WARM_DEVICE = "cuda"   # the CPU glue test (test_v4_glue.py) runs the warm-up on "cpu"


def _self_check_tol(route):   # per kind (v6 raises it for FP8-query kinds)
    return SELF_CHECK_TOL


def _cuda_build():
    from torch.utils.cpp_extension import load_inline
    kinds = sorted({route[0] for route in CUDA_ROUTE.values()})
    if not kinds:
        return
    try:
        mod = load_inline(
            name="okc_attention_v4_" + "_".join(kinds), cpp_sources="\n".join(_KINDS[k][1] for k in kinds),
            cuda_sources="\n".join(_KINDS[k][0] for k in kinds),
            functions=[f for k in kinds for f in _KINDS[k][2]],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=_tempfile.mkdtemp(prefix="okc_attention_v4_"), verbose=False)
    except Exception as exc:  # every class keeps its v2.2 path
        print(f"v4: CUDA build failed, staying on v2.2 paths: {type(exc).__name__}: {str(exc)[-300:]}")
        return
    _CUDA["mod"] = mod
    _CUDA["refused"] = set()
    for k in kinds:
        setup = _KINDS[k][3]
        if setup is not None and getattr(mod, setup)() != 0:
            print(f"v4: {setup} refused (shared memory or cluster attribute); {k} routes dropped")
            _CUDA["refused"].add(k)


def _nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // min_keys, MAX_SPLITS))


def _ud_clusters(cfg, max_pages, cs, ctas_per_sm, warps):
    """Clusters per (request, head pair): one for sliding layers (the window fits one cluster),
    else enough for ctas_per_sm CTAs per SM, at least one warp chunk (32 keys) per warp."""
    if cfg["window"]:
        return 1
    tokens = max_pages * cfg["page_size"]
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    nc = max(1, math.ceil(ctas_per_sm * NUM_SMS / (programs * cs)))
    return max(1, min(nc, tokens // (cs * 32 * warps)))


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
    kind = route[0]
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    max_pages = data["page_table"].shape[1]
    mod = _CUDA["mod"]
    if kind == "ud":
        _, cs, cps, warps, stages = route
        nc = _ud_clusters(cfg, max_pages, cs, cps, warps)
        rope = bool(cfg["rope_theta"])
        cpart_o = torch.empty((batch * hq * nc, 64), dtype=torch.float32, device=device)
        cpart_m = torch.empty((batch * hq * nc,), dtype=torch.float32, device=device)
        cpart_l = torch.empty_like(cpart_m)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        mod.ud(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
               data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
               data["cache_seqlens"], cpart_o, cpart_m, cpart_l, _TICKETS[0], out, batch, hq, hkv, max_pages,
               cs * nc, cfg["page_size"], cfg["window"], int(rope), cfg["qk_scale_factor"], cfg["sm_scale"],
               warps, stages, cs)
        return out
    q = _prologue_q(data)
    if kind == "ver2":
        _, cps, stages = route
        nsplit = _nsplit(cfg, max_pages, cps, 64)
    else:   # dec2 / dec3 / dec4: one warp chunk (32 keys) per warp at least
        cps, warps = route[1], route[2]
        nsplit = _nsplit(cfg, max_pages, cps, 32 * warps)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
            cfg["sm_scale"])
    if kind == "dec2":
        mod.attend_dec2(*args, route[2])
    elif kind == "dec3":
        mod.attend_dec3(*args, route[2], route[3])
    elif kind == "dec4":
        mod.attend_dec4(*args, route[2], route[3])
    else:
        mod.attend_ver2(*args, route[2])
    return _merge(data, o_part, m_part, l_part, nsplit)


def _cuda_ok(cfg, route):
    """Every CUDA kernel needs groups of 8 query heads and paired KV heads; the decode kinds take
    q_len 1, ver2 q_len 8."""
    kind = route[0]
    if cfg["num_q_heads"] // cfg["num_kv_heads"] != 8 or cfg["num_kv_heads"] % 2:
        return False
    return cfg["q_len"] == (8 if kind == "ver2" else 1)


def _self_check_inputs(bclass, length, full, device):
    """Synthetic inputs of a class (our own random data, never the judge's): a random FP8 cache
    so every key matters, a context that spans the window on sliding layers."""
    data = dummy_inputs(bclass, length, 0 if full else 2048, scored_max_pages(bclass, length), device,
                        ctx=3000 if full else 2300)
    gen = torch.Generator(device=device).manual_seed(4040 + bclass + length)
    for name in ("k_cache", "v_cache"):
        t = data[name]
        data[name] = (torch.randn(t.shape, generator=gen, device=device) * 0.5).to(t.dtype)
    return data


_kernel_fn_v22 = kernel_fn
_warm_up_v22 = _warm_up


def _warm_up():  # noqa: F811 (v4: Triton warm-up, then the CUDA build and a checked launch per route)
    _warm_up_v22()
    _cuda_build()
    if "mod" not in _CUDA:
        CUDA_ROUTE.clear()
        return
    dev = torch.device(_WARM_DEVICE)
    _TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    for cls, route in sorted(CUDA_ROUTE.items()):
        full, verify, bclass = cls
        try:
            if route[0] in _CUDA["refused"]:
                raise RuntimeError("setup refused")
            data = _self_check_inputs(bclass, 8 if verify else 1, full, dev)
            if not _cuda_ok(data["config"], route):
                raise RuntimeError("config not supported")
            ours = _cuda_run(data, route).float()
            want = run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-6))
            if not err < _self_check_tol(route):
                raise RuntimeError(f"differs from the Triton path: rel. L2 {err:.2e}")
        except Exception as exc:
            print(f"v4: route {cls} -> {route} dropped: {type(exc).__name__}: {str(exc)[-200:]}")
            CUDA_ROUTE.pop(cls)


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    route = CUDA_ROUTE.get(shape_class(cfg))
    if route is not None and "mod" in _CUDA and _cuda_ok(cfg, route):
        return _cuda_run(data, route)
    return _kernel_fn_v22(data)



# =============================================================================================
# v6: v5 + the V4 verify kernel (cuda_ver4.py: S2c's producer warp and page-entry ring, FP8
# consumers: Q.K and P.V on mma.m16n8k32.e4m3, queries rounded to E4M3 as FA3 does and the
# judge's bound allows) as CUDA kind "ver4", between the Triton prologue and _combine; and D3
# routes with a whole-wave split ("dec4", ctas per SM, warps, stages, "w"; AF33); and UD3
# (cuda_ud3.py, make_ud3.py: one launch per sliding decode layer, AF32) as ("ud3", cs, warps, stages).
# Routes ("ver4", ctas per SM, stages, m-tiles per consumer warp, min blocks per SM) come from
# the L16 lab's judge-H200 timings. FP8 queries put the self-check's distance from the Triton
# path at ~0.06 (the judge's own FP8-query reference sits there too); a layout bug is O(1).
# =============================================================================================
_KINDS["ver4"] = (CUDA_VER4_SRC, CUDA_VER4_DECL, ["attend_ver4", "attend_ver4_setup", "attend_ver4_attrs"],
                  "attend_ver4_setup")
_KINDS["ud3"] = (CUDA_UD3_SRC, CUDA_UD3_DECL, ["ud3", "ud3_setup", "ud3_attrs"], "ud3_setup")
FP8_KINDS = {"ver4"}
SELF_CHECK_TOL_FP8 = 0.15
CUDA_ROUTE.update(V6_ROUTES)

_cuda_ok_v5 = _cuda_ok
_cuda_run_v5 = _cuda_run
MERGE_CB = 256   # merge splits in tiles of up to MERGE_CB (8 warps above 64): fewer serial passes (AF35)


def _merge(data, o_part, m_part, l_part, nsplit):  # noqa: F811
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    cb = min(MERGE_CB, triton.next_power_of_2(nsplit))
    _combine[(batch * length, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit,
                                   HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=8 if cb > 64 else 4)
    return out


def _self_check_tol(route):  # noqa: F811
    return SELF_CHECK_TOL_FP8 if route[0] in FP8_KINDS else SELF_CHECK_TOL


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "ud3":      # sliding decode only: one cluster per (request, KV-head pair)
        return (cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1 and cfg["window"] > 0)
    if route[0] == "ver4":
        return cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v5(cfg, route)


def _nsplit_w(cfg, max_pages, ctas_per_sm, min_keys):
    """Whole-wave split (AF33): floor(ctas_per_sm x SMs / programs) CTAs per request, so the grid
    fills whole waves of resident CTAs instead of spilling a few into one more wave."""
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(ctas_per_sm * NUM_SMS // programs, tokens // min_keys, MAX_SPLITS))


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud3":      # ("ud3", cluster size, warps, stages): the whole sliding decode layer
        _, cs, warps, stages = route
        cfg = data["config"]
        batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
        device = data["qkv"].device
        rope = bool(cfg["rope_theta"])
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        _CUDA["mod"].ud3(data["qkv"], data["positions"], data["cache_loc"],
                         _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                         data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                         hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                         cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
        return out
    if route[0] == "dec4" and len(route) == 5:      # ("dec4", ctas per SM, warps, stages, "w"): whole waves
        _, cps, warps, stages, _w = route
        cfg = data["config"]
        batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
        max_pages = data["page_table"].shape[1]
        q = _prologue_q(data)
        nsplit = _nsplit_w(cfg, max_pages, cps, 32 * warps)
        o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=q.device)
        m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=q.device)
        l_part = torch.empty_like(m_part)
        _CUDA["mod"].attend_dec4(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                                 data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                                 cfg["page_size"], cfg["window"], cfg["sm_scale"], warps, stages)
        return _merge(data, o_part, m_part, l_part, nsplit)
    if route[0] != "ver4":
        return _cuda_run_v5(data, route)
    _, cps, stages, mt, minb = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_ver4(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                             data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                             cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, mt, minb)
    return _merge(data, o_part, m_part, l_part, nsplit)

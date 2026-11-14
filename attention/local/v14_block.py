# =============================================================================================
# v14: v13 + UD4 (cuda_ud4.py, make_ud4.py: UD3 with row-owner finalize and the new token's K / V prefetched at entry,
# AF59) as CUDA kind "ud4": one launch per sliding decode layer, one cluster of cs CTAs per (request, KV-head pair).
# Routes ("ud4", cs, warps, stages) come from the L34 lab's judge-H200 timings; checked at warm-up against the Triton
# path (UD3's tolerance: FP16 queries, not an FP8 kind).
# =============================================================================================
_KINDS["ud4"] = (CUDA_UD4_SRC, CUDA_UD4_DECL, ["ud4", "ud4_setup", "ud4_attrs"], "ud4_setup")
CUDA_ROUTE.update(V14_ROUTES)   # PDL_ROUTE kept: a UD4 route dropped at warm-up falls back to the PDL chain

_cuda_ok_v13 = _cuda_ok
_cuda_run_v13 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "ud4":
        return (cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1 and cfg["window"] > 0)
    return _cuda_ok_v13(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "ud4":
        return _cuda_run_v13(data, route)
    _, cs, warps, stages = route
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _CUDA["mod"].ud4(data["qkv"], data["positions"], data["cache_loc"],
                     _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                     data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                     hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                     cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
    return out

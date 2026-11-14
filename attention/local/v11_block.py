# =============================================================================================
# v11: v10 + UW2 (cuda_uw2.py, make_uw2.py) as CUDA kind "uw2": one launch per sliding verify layer (UV4's in-kernel
# prologue and cluster merge around W2c's SS-wgmma consumers), one cluster of cs CTAs per (request, KV-head pair).
# Routes ("uw2", stages, pp, cs) come from the L30 lab's judge-H200 timings; checked at warm-up against the Triton
# path (FP8 tolerance).
# =============================================================================================
_KINDS["uw2"] = (CUDA_UW2_SRC, CUDA_UW2_DECL, ["uw2", "uw2_setup", "uw2_attrs"], "uw2_setup")
FP8_KINDS.add("uw2")
CUDA_ROUTE.update(V11_ROUTES)

_cuda_ok_v10 = _cuda_ok
_cuda_run_v10 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "uw2":
        return (cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
                and cfg["window"] > 0)
    return _cuda_ok_v10(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "uw2":
        return _cuda_run_v10(data, route)
    _, stages, pp, cs = route
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
    _CUDA["mod"].uw2(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                     data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                     data["cache_seqlens"], out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"],
                     cfg["window"], int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
    return out

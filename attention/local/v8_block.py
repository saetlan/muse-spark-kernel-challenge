

# =============================================================================================
# v8: v7 + V5 (cuda_ver5.py, make_ver5.py) as CUDA kind "ver5": V4's verify kernel with one softmax
# round per 64-key stage, edge-only masks, the scale fused into the exponent's FMA and an optional
# register cap for two CTAs per SM (L17: verify is consumer-bound, AF37), between the Triton
# prologue and _combine like ver4. Routes ("ver5", ctas per SM, stages, min blocks per SM) come from
# the L18 lab's judge-H200 timings; checked at warm-up against the Triton path (FP8 tolerance).
# =============================================================================================
_KINDS["ver5"] = (CUDA_VER5_SRC, CUDA_VER5_DECL, ["attend_ver5", "attend_ver5_setup", "attend_ver5_attrs"],
                  "attend_ver5_setup")
_KINDS["ver5t"] = (CUDA_VER5T_SRC, CUDA_VER5T_DECL, ["attend_ver5t", "attend_ver5t_setup", "attend_ver5t_attrs"],
                   "attend_ver5t_setup")   # V5t (cuda_ver5t.py): V re-laid once per stage by the producer (A56)
FP8_KINDS.update({"ver5", "ver5t"})
CUDA_ROUTE.update(V8_ROUTES)

_cuda_ok_v7 = _cuda_ok
_cuda_run_v7 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] in ("ver5", "ver5t"):
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v7(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] not in ("ver5", "ver5t"):
        return _cuda_run_v7(data, route)
    kind, cps, stages, minb = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    fn = _CUDA["mod"].attend_ver5 if kind == "ver5" else _CUDA["mod"].attend_ver5t
    fn(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
       data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
       cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, minb)
    return _merge(data, o_part, m_part, l_part, nsplit)

# =============================================================================================
# v10: v9 + W2c (cuda_w2.py, make_w2.py) as CUDA kind "w2": verify on SS wgmma FP8 with Q and P in shared memory
# and FA3's intra-warpgroup overlap (optional ping-pong of the two warpgroups), between the Triton prologue and
# _combine like w1n, with whole-wave splits. Routes ("w2", ctas per SM, stages, pp) come from the L28 lab's
# judge-H200 timings; checked at warm-up against the Triton path (FP8 tolerance).
# =============================================================================================
_KINDS["w2"] = (CUDA_W2_SRC, CUDA_W2_DECL, ["attend_w2", "attend_w2_setup", "attend_w2_attrs"], "attend_w2_setup")
FP8_KINDS.add("w2")
CUDA_ROUTE.update(V10_ROUTES)

_cuda_ok_v9 = _cuda_ok
_cuda_run_v9 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "w2":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v9(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "w2":
        return _cuda_run_v9(data, route)
    _, cps, stages, pp = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                           cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp)
    return _merge(data, o_part, m_part, l_part, nsplit)

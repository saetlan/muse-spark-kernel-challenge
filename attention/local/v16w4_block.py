# =============================================================================================
# v15 (W4): v14 + W4 (cuda_w4.py, make_w4.py, AF68: verify with consumer pairs that copy and transpose their own stages,
# no producer warps) as CUDA kind "w4". Routes ("w4", stages, npair, minb) come from the L45 lab's judge-H200 timings;
# splits as the lab: whole waves at 2 pairs per SM, a multiple of npair; checked at warm-up (FP8 tolerance).
# =============================================================================================
_KINDS["w4"] = (CUDA_W4_SRC, CUDA_W4_DECL, ["attend_w4", "attend_w4_setup", "attend_w4_attrs"], "attend_w4_setup")
FP8_KINDS.add("w4")
CUDA_ROUTE.update(V15_ROUTES)

_cuda_ok_v14 = _cuda_ok
_cuda_run_v14 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "w4":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v14(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "w4":
        return _cuda_run_v14(data, route)
    _, stages, npair, minb = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 2, 64)
    nsplit = max(npair, nsplit // npair * npair)
    rows = batch * length * hq * nsplit
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((rows,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w4(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                           o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                           cfg["sm_scale"], stages, npair, minb)
    return _merge(data, o_part, m_part, l_part, nsplit)

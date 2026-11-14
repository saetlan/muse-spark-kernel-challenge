# =============================================================================================
# v12: v11 + W3 (cuda_w3.py, make_w3.py) as CUDA kind "w3": full verify on SS wgmma FP8 with P.V accumulated into O in
# place, two copier warps and 4 consumer warpgroups (two pairs take alternate 64-key stages; setmaxnreg moves registers
# from the producer warpgroup to the consumers), between the Triton prologue and _combine like w2, whole-wave splits,
# nsplit x 2 partials per row. Routes ("w3", stages, pp, nwg, rc) come from the L31 lab's judge-H200 timings; checked
# at warm-up against the Triton path (FP8 tolerance).
# =============================================================================================
_KINDS["w3"] = (CUDA_W3_SRC, CUDA_W3_DECL, ["attend_w3", "attend_w3_setup", "attend_w3_attrs"], "attend_w3_setup")
FP8_KINDS.add("w3")
CUDA_ROUTE.update(V12_ROUTES)

_cuda_ok_v11 = _cuda_ok
_cuda_run_v11 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "w3":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v11(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "w3":
        return _cuda_run_v11(data, route)
    _, stages, pp, nwg, rc = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    nparts = nsplit * (nwg // 2)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nparts, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nparts,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w3(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                           cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)

# =============================================================================================
# v13: v12 + W3b (cuda_w3b.py from make_wholerow.py: W3 with the copier reading whole 128 B rows per instruction, AF56)
# as CUDA kind "w3b" on full verify bs16 / bs32 (replacing v12's "w3" routes there). Same consumers as W3: P.V into O in
# place, 4 consumer warpgroups, setmaxnreg; nsplit x 2 partials per row merged by _combine. Routes ("w3b", stages, pp,
# nwg, rc) come from the L32 lab's judge-H200 timings; checked at warm-up against the Triton path (FP8 tolerance).
# =============================================================================================
_KINDS["w3b"] = (CUDA_W3B_SRC, CUDA_W3B_DECL, ["attend_w3b", "attend_w3b_setup", "attend_w3b_attrs"], "attend_w3b_setup")
FP8_KINDS.add("w3b")
CUDA_ROUTE.update(V13_ROUTES)

_cuda_ok_v12 = _cuda_ok
_cuda_run_v12 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "w3b":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v12(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "w3b":
        return _cuda_run_v12(data, route)
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
    _CUDA["mod"].attend_w3b(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                           cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)

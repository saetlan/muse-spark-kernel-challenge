# =============================================================================================
# v15: v14 + the copier with page entries 2S stages ahead (make_entry_lead.py, AF66): W3e (cuda_w3e.py, from W3b) as kind
# "w3e" and UW2e (cuda_uw2e.py, from UW2) as kind "uw2e". In W3b / UW2 the copier's per-stage wait for entries also
# waited for rows S - 1 stages back; with the entries 2S ahead it blocks only on empty slots. Everything else is W3b's /
# UW2's. Routes come from the L44 lab's judge-H200 timings; checked at warm-up against the Triton path (FP8 tolerance).
# =============================================================================================
_KINDS["w3e"] = (CUDA_W3E_SRC, CUDA_W3E_DECL, ["attend_w3e", "attend_w3e_setup", "attend_w3e_attrs"], "attend_w3e_setup")
_KINDS["uw2e"] = (CUDA_UW2E_SRC, CUDA_UW2E_DECL, ["uw2e", "uw2e_setup", "uw2e_attrs"], "uw2e_setup")
FP8_KINDS.add("w3e")
FP8_KINDS.add("uw2e")
CUDA_ROUTE.update(V15_ROUTES)

_cuda_ok_v14 = _cuda_ok
_cuda_run_v14 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] in ("w3e", "uw2e"):
        ok = cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
        return ok and (route[0] == "w3e" or cfg["window"] > 0)
    return _cuda_ok_v14(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] not in ("w3e", "uw2e"):
        return _cuda_run_v14(data, route)
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    if route[0] == "uw2e":
        _, stages, pp, cs = route
        device = data["qkv"].device
        rope = bool(cfg["rope_theta"])
        out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
        _CUDA["mod"].uw2e(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                          data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                          data["cache_seqlens"], out, batch, hq, hkv, max_pages, cfg["page_size"], cfg["window"],
                          int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
        return out
    _, stages, pp, nwg, rc = route
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    nparts = nsplit * (nwg // 2)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nparts, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nparts,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w3e(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                            data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                            cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)

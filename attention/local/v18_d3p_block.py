# =============================================================================================
# v18: D3p as kind "dec4p" (cuda_dec4p.py, make_pdl.py: D3 as a PDL dependent of the Triton prologue `_prologue_p`, page
# entries issued before griddepcontrol.wait, `_combine_p` launched with launch_pdl; L38 on the judge's H200: Fd16 -1.8%,
# Fd32 -2.0% vs D3 'w'). Routes ("dec4p", ctas per SM, warps, stages, early trigger); checked at warm-up like every route.
# =============================================================================================
_KINDS["dec4p"] = (CUDA_DEC4P_SRC, CUDA_DEC4P_DECL, ["attend_dec4p", "attend_dec4p_setup"], "attend_dec4p_setup")
CUDA_ROUTE.update(V18_D3P_ROUTES)

_cuda_ok_v17 = _cuda_ok
_cuda_run_v17 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "dec4p":
        return (cfg["q_len"] == 1 and cfg["window"] == 0 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_v17(cfg, route)


def _prologue_q_pdl(data, early):   # v18: v5's PDL prologue (Triton, triggers its dependents)
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=qkv.device)
    _prologue_p[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, qkv.device),
                          q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                          HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS,
                          TRIGGER_EARLY=bool(early), num_warps=4)
    return q


def _merge_pdl(data, o_part, m_part, l_part, nsplit):   # v18: v5's combine as a PDL dependent
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    cb = min(MERGE_CB, triton.next_power_of_2(nsplit))
    _combine_p[(batch, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit,
                            HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=8 if cb > 64 else 4, launch_pdl=True)
    return out


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "dec4p":
        return _cuda_run_v17(data, route)
    _, cps, warps, stages, early = route
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    max_pages = data["page_table"].shape[1]
    q = _prologue_q_pdl(data, early)
    nsplit = _nsplit_w(cfg, max_pages, cps, 32 * warps)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    getattr(_CUDA["mod"], "attend_dec4p")(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                                         cfg["page_size"], cfg["window"], cfg["sm_scale"], warps, stages, int(early))
    return _merge_pdl(data, o_part, m_part, l_part, nsplit)

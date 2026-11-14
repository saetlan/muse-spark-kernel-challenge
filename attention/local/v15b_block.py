# =============================================================================================
# v15: v14 + the L46-L49 lab winners: D3 configs (kind dec4, v14's kernel), UD5 (cuda_ud5.py, make_ud5.py: UD4 with
# AF70's fixes and clusters 2..16; only the routed variants are instantiated) and/or UD4c (UD4's kernel at other cluster
# sizes), UW2v (cuda_uw2v.py, make_uw2v.py: UW2 with 16-byte DSMEM pushes), UD6 (make_ud6.py: UD5 with groups of
# clusters and a ticketed second level: one launch per full decode layer), UW2g (make_uw2g.py: UW2 + UV4's second level). Routes from the labs' judge-H200 timings;
# every CUDA route is checked at warm-up against the Triton path and dropped on a mismatch (v14's fallbacks stay).
# =============================================================================================
_KINDS["ud5"] = (CUDA_UD5_SRC, CUDA_UD5_DECL, ["ud5", "ud5_setup", "ud5_attrs"], "ud5_setup")
_KINDS["ud4c"] = (CUDA_UD4C_SRC, CUDA_UD4C_DECL, ["ud4c", "ud4c_setup", "ud4c_attrs"], "ud4c_setup")
_KINDS["uw2v"] = (CUDA_UW2V_SRC, CUDA_UW2V_DECL, ["uw2v", "uw2v_setup", "uw2v_attrs"], "uw2v_setup")
_KINDS["ud6"] = (CUDA_UD6_SRC, CUDA_UD6_DECL, ["ud6", "ud6_setup", "ud6_attrs"], "ud6_setup")
_KINDS["uw2g"] = (CUDA_UW2G_SRC, CUDA_UW2G_DECL, ["uw2g", "uw2g_setup", "uw2g_attrs"], "uw2g_setup")
if "CUDA_UW2H_SRC" in globals():   # v16: UW2h (make_uw2h.py: UW2 with a half-size FP16 push)
    _KINDS["uw2h"] = (CUDA_UW2H_SRC, CUDA_UW2H_DECL, ["uw2h", "uw2h_setup", "uw2h_attrs"], "uw2h_setup")
    FP8_KINDS.add("uw2h")
if "CUDA_UD8_SRC" in globals():   # v17: UD8 (make_ud8.py: UD5 with the owners' merge over every warp)
    _KINDS["ud8"] = (CUDA_UD8_SRC, CUDA_UD8_DECL, ["ud8", "ud8_setup", "ud8_attrs"], "ud8_setup")
if "CUDA_UD9_SRC" in globals():   # v18: UD9 (make_ud9.py: UD8 with a compact receive buffer)
    _KINDS["ud9"] = (CUDA_UD9_SRC, CUDA_UD9_DECL, ["ud9", "ud9_setup", "ud9_attrs"], "ud9_setup")
if "CUDA_UD7_SRC" in globals():   # v16: UD7 (make_ud7.py: UD6 with multi-row owners, clusters 1..16)
    _KINDS["ud7"] = (CUDA_UD7_SRC, CUDA_UD7_DECL, ["ud7", "ud7_setup", "ud7_attrs"], "ud7_setup")
FP8_KINDS.update({"uw2v", "uw2g"})
_V15_KINDS = ("ud5", "ud4c", "uw2v", "ud6", "uw2g", "ud7", "uw2h", "ud8", "ud9")
CUDA_ROUTE.update(V15_ROUTES)   # PDL_ROUTE kept: a route dropped at warm-up falls back to v14's PDL chain / Triton path

_cuda_ok_v14 = _cuda_ok
_cuda_run_v14 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] in ("ud6", "ud7"):   # any context (full or sliding); tickets: bp * cs <= 4096 (v4's _TICKETS)
        return (cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 1
                and cfg["batch"] * (cfg["num_kv_heads"] // 2) * route[1] <= 4096)
    if route[0] == "uw2g":
        return (cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
                and cfg["batch"] * (cfg["num_kv_heads"] // 2) * route[3] <= 4096)
    if route[0] in ("ud5", "ud4c", "ud8", "ud9"):
        return (cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1 and cfg["window"] > 0)
    if route[0] in ("uw2v", "uw2h"):
        return (cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
                and cfg["window"] > 0)
    return _cuda_ok_v14(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] not in _V15_KINDS:
        return _cuda_run_v14(data, route)
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    inv = _inv_freq(cfg["rope_theta"] if rope else 1.0, device)
    if route[0] == "uw2g":   # gpart: written by every cluster before the last one reads it (v6's uv4 glue)
        _, stages, pp, cs, ncl = route
        bp = batch * (hkv // 2)
        out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
        gpart = torch.empty((bp * ncl * 128 * 68 if ncl > 1 else 4,), dtype=torch.float32, device=device)
        _CUDA["mod"].uw2g(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
                          data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq, hkv,
                          data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                          cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs, ncl, gpart, _TICKETS[0])
        return out
    if route[0] in ("ud6", "ud7"):
        _, cs, warps, stages, groups = route
        bp = batch * (hkv // 2)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        gpart = torch.empty((bp * groups * 16 * 68 if groups > 1 else 4,), dtype=torch.float32, device=device)
        fn = _CUDA["mod"].ud6 if route[0] == "ud6" else _CUDA["mod"].ud7
        fn(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
                         data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                         hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                         cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs, groups, gpart, _TICKETS[0])
        return out
    if route[0] in ("uw2v", "uw2h"):
        _, stages, pp, cs = route
        out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
        (_CUDA["mod"].uw2v if route[0] == "uw2v" else _CUDA["mod"].uw2h)(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
                          data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq, hkv,
                          data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                          cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
        return out
    _, cs, warps, stages = route
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    fn = getattr(_CUDA["mod"], route[0])   # only the routed kinds are compiled: look up this route's function alone
    fn(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
       data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
       hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
       cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
    return out

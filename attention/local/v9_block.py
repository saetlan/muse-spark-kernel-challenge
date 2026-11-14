

# =============================================================================================
# v9: v8 + W1-N (cuda_w1n.py, make_w1n.py) as CUDA kind "w1n": verify on wgmma FP8 (QGMMA; FP8 mma.sync
# is emulated on Hopper, AF41), between the Triton prologue and _combine like ver4 / ver5, with
# whole-wave splits. W1-N is the W1 text that L22-L24 found right on the GPU (AF46: the plain W1 text
# computes garbage; the same text with dump code guarded off by a null pointer is right on every block).
# Routes ("w1n", ctas per SM, stages) come from the L24 lab's judge-H200 timings;
# checked at warm-up against the Triton path (FP8 tolerance). Also the L20 tune results: TUNE_UPDATE
# (Triton PDL launch settings) and SGLANG_FALLBACK (classes that leave the PDL chain for SGLang).
# =============================================================================================
_KINDS["w1n"] = (CUDA_W1N_SRC, CUDA_W1N_DECL, ["attend_w1n", "attend_w1n_setup"], "attend_w1n_setup")
FP8_KINDS.add("w1n")
CUDA_ROUTE.update(V9_ROUTES)
TUNE.update(TUNE_UPDATE)
for _cls in SGLANG_FALLBACK:
    PDL_ROUTE.discard(_cls)

_cuda_ok_v8 = _cuda_ok
_cuda_run_v8 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "w1n":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_v8(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "w1n":
        return _cuda_run_v8(data, route)
    _, cps, stages = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w1n(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                            data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                            cfg["page_size"], cfg["window"], cfg["sm_scale"], stages)
    return _merge(data, o_part, m_part, l_part, nsplit)

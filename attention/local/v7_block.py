

# =============================================================================================
# v7: v6 + UV4 (cuda_uv4.py, make_uv4.py) as CUDA kind "uv4": one launch per verify layer
# (q_len 8; sliding, or full where one cluster per request fills the GPU): V4's FP8 attention with the prologue (query fold, the block's keys and
# values written to the cache) and the merge (cluster push, RMSNorm x sigmoid(gate), BF16 out)
# inside the kernel, one cluster of cs CTAs per (request, KV-head pair). Routes ("uv4", cluster
# size, stages, m-tiles per consumer warp[, clusters per request: second level through global
# memory and a ticket per (request pair, rank), for full verify bs1]) come from the L17 lab's
# judge-H200 timings; the
# warm-up checks each against the Triton path (FP8 queries: tolerance SELF_CHECK_TOL_FP8).
# =============================================================================================
_KINDS["uv4"] = (CUDA_UV4_SRC, CUDA_UV4_DECL, ["uv4", "uv4_setup", "uv4_attrs"], "uv4_setup")
FP8_KINDS.add("uv4")
CUDA_ROUTE.update(V7_ROUTES)

_cuda_ok_v6 = _cuda_ok
_cuda_run_v6 = _cuda_run
_self_check_inputs_v4 = _self_check_inputs


def _self_check_inputs(bclass, length, full, device):  # noqa: F811
    """v4's synthetic inputs with disjoint cache rows per request: dummy_inputs' page table wraps
    (arange % pages), so one request's new-token slot is another's past key. The 3-kernel paths write
    every new row before any read; kernels that write the new rows in-kernel (UD3, UV4) race with
    the other requests' reads there and failed the self-check for that reason alone (L16: UD3 Sd16 /
    Sd32 dropped at 0.29 / 0.37). The judge's pages are disjoint per request (generate_input)."""
    data = _self_check_inputs_v4(bclass, length, full, device)
    batch = data["config"]["batch"]
    span = int(data["cache_seqlens"].max())          # ctx + length rows per request, pages = batch x span
    rows, max_pages = data["k_cache"].shape[0], data["page_table"].shape[1]
    table = (torch.arange(batch, device=device)[:, None] * span + torch.arange(max_pages, device=device)[None, :]) % rows
    data["page_table"] = table.to(torch.int32)
    data["cache_loc"] = torch.gather(table.long(), 1, data["positions"].view(batch, length)).reshape(-1)
    return data


def _cuda_ok(cfg, route):  # noqa: F811
    if route[0] == "uv4":      # verify: one cluster per (request, KV-head pair)
        return (cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 8)
    return _cuda_ok_v6(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] != "uv4":
        return _cuda_run_v6(data, route)
    cs, stages, mt = route[1:4]
    ncl = route[4] if len(route) > 4 else 1         # clusters per request (second level when > 1)
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
    gpart = torch.empty((batch * (hkv // 2) * ncl * 128 * 68 if ncl > 1 else 4,), dtype=torch.float32, device=device)
    _CUDA["mod"].uv4(data["qkv"], data["positions"], data["cache_loc"],
                     _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                     data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq, hkv,
                     data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                     cfg["qk_scale_factor"], cfg["sm_scale"], stages, mt, cs, ncl, gpart, _TICKETS[0])
    return out

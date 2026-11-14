# v17 = v15 + MoE's scored E25 expert path (replaces v16's E19 subset). parts/moe25.py (local/make_embed_subset.py
# --which e25) holds the 45 definitions of moe/scored/submission_m13t_e25.py (scored 1.6279x on muse_moe_w4a16, job
# b5da76a6e7cb4267) that its run path needs, prefixed m25_. E25's own tuner picked, on its problem: M11b (E24: M11's
# kernel on M13's bank and rotate-and-mask decode, ~130 instead of ~190 instructions per warp per chunk) for GEMM1 and
# GEMM2 (PDL) at 1-32 tokens, M13's CUDA post-norm (PDL); judged bs1 32.71 / bs8 79.88 / bs16 111.87 / bs32 158.12 us vs
# E19's 37.46 / 95.77 / 129.65 / 185.60. v17 builds its own extension with only those four M11b instances and the two
# M19 instances v16 used (E21/E22's CR / SR kernels and G2P emptied: their lists are read at call time; wgmma compiled
# out), M13's bank from our M11 bank (m11_bank / m11_shuffle are identical in E25 and here), a state of its own, and a
# per-class choice (V17_CHOICE: "m11b" or "m19") a lab fills. Classes without a choice keep v15's paths (our M11 with
# the PDL GEMM2 + CUDA post-norm at <= 32 tokens, Marlin above).

V17_CHOICE = {}  # class bound (fewer than) -> "m11b" | "m19", from a lab's printed V17_CHOICE
V17_FORCE = False  # a lab sets it: build and check even with no choice
V17_CLASSES = ((5, 1), (9, 5), (17, 9), (33, 17), (192, 65), (321, 192))  # (bound, lowest token count)
V17_M19_INSTANCES = ((1, 2, 2, 2, 4), (0, 2, 2, 4, 4))  # (gated, tt, nt, P, MB): GEMM1 g2 n2 P2 MB4, GEMM2 t2 n2 P4 MB4
V17_M11B_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4))  # E25's four small-batch picks
_V17_POST = {"kind": "m13", "pdl": 1}
V17_CFGS = {  # E25's printed M13_CONFIGS (classes 5 / 9-33 for M11b; 65 / 192 for M19 without the hot split)
    "m19": {"s1": {"kind": "m19", "nt": 2, "c": 4, "pfd": 0, "p": 2, "mb": 4},
            "s2": {"kind": "m19", "nt": 2, "c": 4, "pfd": 0, "p": 4, "mb": 4, "tt": 2, "pdl": 1, "pf": 0}, "post": _V17_POST},
    "m11b_t1": {"s1": {"kind": "m11b", "nt": 1, "d": 4, "ks": 4, "c": 8},
                "s2": {"kind": "m11b", "tt": 1, "nt": 1, "d": 4, "ks": 4, "c": 4, "pdl": 1}, "post": _V17_POST},
    "m11b": {"s1": {"kind": "m11b", "nt": 1, "d": 4, "ks": 2, "c": 8},
             "s2": {"kind": "m11b", "tt": 2, "nt": 1, "d": 4, "ks": 4, "c": 8, "pdl": 1}, "post": _V17_POST},
}
V17_FUNCTIONS = ["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                 "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                 "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon",
                 "m11p_gemm", "m18_gemm", "m18_setup", "m18_recon", "m19_gemm", "m19_setup", "m11_sr_setup",
                 "m11f_gemm2_post", "m11b_gemm"]  # as E25's m13_build

# The instance lists E25's m13_source reads at call time: only what v17 uses.
m25_M11B_INSTANCES = V17_M11B_INSTANCES
m25_M13_M11_CR_INSTANCES = ()
m25_M13_M11_SR_INSTANCES = ()
m25_M11F_INSTANCES = ()


def v17_source():
    return m25_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(), m16_instances=(),
                          m17_instances=(), m18_instances=(), m19_instances=V17_M19_INSTANCES)


def v17_build():
    return _v9_timed("e25_nvcc", load_inline)(
        name="moe25_ext", cpp_sources=m25__M11_CPP + m25__M13_CPP + m25__M11F_CPP, cuda_sources=v17_source(),
        functions=V17_FUNCTIONS, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DM13_NO_WGMMA"],
        build_directory=tempfile.mkdtemp(prefix="moe25_"), verbose=False)


def v17_cfg(state, tokens):
    """E25's config for this token count if v17 serves its class, else None."""
    if "v17_moe" not in state:
        return None
    picks = state.get("v17_choice") or {}
    for below, low in V17_CLASSES:
        if low <= tokens < below:
            kind = picks.get(below)
            if kind == "m19":
                return V17_CFGS["m19"]
            if kind == "m11b":
                return V17_CFGS["m11b_t1" if below == 5 else "m11b"]
            return None
    return None


def v17_run(x, topk_ids, topk_weights, state, cfg):
    return m25_m13_run(x.contiguous(), topk_ids.to(torch.int32), topk_weights.to(torch.float32), state["v17_moe"], cfg)


_v17_m11_run = m11_run  # (v14, when built in, rebinds it to g2p_m11_run with cfg["g2p"] off)


def m11_run(x, topk_ids, topk_weights, state, cfg):  # noqa: F811 (our M11 at <= 32 tokens unless v17 serves the class)
    c = v17_cfg(state, x.shape[0])
    if c is not None:
        return v17_run(x, topk_ids, topk_weights, state, c)
    return _v17_m11_run(x, topk_ids, topk_weights, state, cfg)


_v17_d1_experts = _experts


def _experts(x, topk_weights, topk_ids, state):  # noqa: F811 (D1's Marlin path unless v17 serves the class)
    c = v17_cfg(state, x.shape[0])
    if c is not None:
        return v17_run(x, topk_ids, topk_weights, state, c)
    return _v17_d1_experts(x, topk_weights, topk_ids, state)


def v17_case(tokens, device, gen):
    """Expert-stage inputs: normalized-scale activations, top-8 of 256 with about a quarter of the routes on this GPU's
    32 experts, FP32 weights summing to 1."""
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=device)).to(torch.bfloat16)
    sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
    sc[:, :LOCAL_EXPERTS] += 0.35
    ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
    tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
    return x, ids, tw / tw.sum(-1, keepdim=True)


def v17_setup(state, weights):
    """Build, bank, check M11b against our M11 path at 1 / 8 / 16 / 32 tokens and M19 against Marlin at 128 / 256
    (< 2e-2; also every kernel's first launch, outside any capture), then install V17_CHOICE. Any failure leaves
    v15's paths."""
    import time as _v17_time
    state["v17_choice"] = {}
    picks = {below: kind for below, kind in V17_CHOICE.items() if kind in ("m11b", "m19")}
    if not picks and not V17_FORCE:
        print("[v17] MoE E25 path: not used (no choice)")
        return
    t0 = _v17_time.monotonic()
    try:
        if "m11" not in state:
            raise RuntimeError("no M11 bank")
        if not m25_m13_scales_ok(weights, 0, LOCAL_EXPERTS):
            raise RuntimeError("an E4M3 block scale is zero, subnormal or NaN (M13's decode needs normal scales)")
        ext = v17_build()
        ext.m19_setup()
        state["v17_moe"] = {"m11_ext": ext, "m13": m25_m13_bank(state["m11"], weights["w13"].device),
                            "sms": state["sms"], "norm_w": state["norm_w"]}
        gen = torch.Generator(device=weights["w13"].device).manual_seed(71)
        worst = {}
        for tokens, kind in ((1, "m11b_t1"), (8, "m11b"), (16, "m11b"), (32, "m11b"), (128, "m19"), (256, "m19")):
            x, ids, tw = v17_case(tokens, weights["w13"].device, gen)
            mine = v17_run(x, ids, tw, state, V17_CFGS[kind])
            m11cfg = m11_config_for(state, tokens)
            ref = (_v17_m11_run(x, ids, tw, state, m11cfg) if tokens <= 32 and m11cfg is not None
                   else _v17_d1_experts(x, tw, ids, state))
            worst[kind[:4]] = max(worst.get(kind[:4], 0.0), _v3_rel(mine, ref))
        torch.cuda.synchronize()
        bad = {k: v for k, v in worst.items() if not v < 2e-2}
        if bad:
            raise RuntimeError(f"E25 path differs from ours: {bad}")
        state["v17_choice"] = picks
        print(f"[v17] MoE E25 path: built ({_v17_time.monotonic() - t0:.1f}s), worst rel err M11b vs M11 "
              f"{worst.get('m11b', 0):.1e} (1-32 tokens), M19 vs Marlin {worst.get('m19', 0):.1e} (128 / 256); "
              f"installed {picks}")
    except Exception as exc:
        state.pop("v17_moe", None)
        print(f"[v17] MoE E25 path: OFF ({type(exc).__name__}: {str(exc)[-200:]})")


# Prepare time (as v16): bf16_proj's P9 build (~31 s) serves only the router at 16 tokens, which runs on the side
# stream behind the pooler; the router there goes back to bf16_proj v2's judged Triton choice, still hidden.
pj_P9_ENABLED = False
pj_CHOICE[("router", 16)] = {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4,
                             'stages': 4}

# v16 = v15 + MoE's scored E19 large-batch expert path at 65-320 tokens (our verify 16 / 32, Marlin until now).
# parts/moe19.py (local/make_embed_subset.py) holds the 37 E19 definitions that path needs, prefixed moe_. E19's
# own tuner picked, at 65 / 192 / 321 (its printed M13_CONFIGS, scored job 9df74163eea14095: bs128 308.9 us 1.476x,
# bs256 443.4 us 1.379x vs SGLang): GEMM1 M19 n2 c4 P2 MB4, GEMM2 M19 t2 n2 c4 P4 MB4 PDL-launched, M13's CUDA
# post-norm PDL-launched (M19 = M14 with the weight-code ring in shared memory, mma.sync; no wgmma). v16 builds its own
# extension with only those two M19 instances (wgmma compiled out, as E19's fallback build), M13's bank from our M11
# bank (m11_bank / m11_shuffle are identical in E19 and here), and a state of its own (E19's wrappers look kernels up on
# state["m11_ext"], which is our M11 extension). On only in the token classes a lab picked (V16_M19).

V16_M19 = {}  # class bound (fewer than) -> True, from a lab's printed V16_M19; bounds 192 (65-191), 321 (192-320)
V16_FORCE = False  # a lab sets it: build and check even with no picks (a clean file with no picks skips the build)
# (class bound, lowest token count): MoE E20 (check 1630e29e, 1.5009x) also puts its bs32 class [17, 33) on the same
# M19 config (in-tuner 167.7 vs M11's 183.7 us); [33, 65) is not a decode-layer shape and stays on Marlin.
V16_CLASSES = ((33, 17), (192, 65), (321, 192))
V16_INSTANCES = ((1, 2, 2, 2, 4), (0, 2, 2, 4, 4))  # (gated, tt, nt, P, MB): GEMM1 g2 n2 P2 MB4, GEMM2 t2 n2 P4 MB4
V16_CFG = {"s1": {"kind": "m19", "nt": 2, "c": 4, "pfd": 0, "p": 2, "mb": 4},
           "s2": {"kind": "m19", "nt": 2, "c": 4, "pfd": 0, "p": 4, "mb": 4, "tt": 2, "pdl": 1, "pf": 0},
           "post": {"kind": "m13", "pdl": 1}}
V16_FUNCTIONS = ["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                 "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                 "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon",
                 "m11p_gemm", "m18_gemm", "m18_setup", "m18_recon", "m19_gemm", "m19_setup"]  # as E19's m13_build


def v16_source():
    return moe_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(), m16_instances=(),
                          m17_instances=(), m18_instances=(), m19_instances=V16_INSTANCES)


def v16_build():
    return _v9_timed("m19_nvcc", load_inline)(
        name="moe19_ext", cpp_sources=moe__M11_CPP + moe__M13_CPP, cuda_sources=v16_source(),
        functions=V16_FUNCTIONS, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DM13_NO_WGMMA"],
        build_directory=tempfile.mkdtemp(prefix="moe19_"), verbose=False)


def v16_on(state, tokens):
    picks = state.get("v16_m19") or {}
    if "v16_moe" not in state:
        return False
    return any(low <= tokens < below and picks.get(below) for below, low in V16_CLASSES)


_v16_d1_experts = _experts


def _experts(x, topk_weights, topk_ids, state):  # noqa: F811 (D1's Marlin path unless v16 serves this size)
    if v16_on(state, x.shape[0]):
        return moe_m13_run(x.contiguous(), topk_ids.to(torch.int32), topk_weights.to(torch.float32), state["v16_moe"],
                           V16_CFG)
    return _v16_d1_experts(x, topk_weights, topk_ids, state)


_v16_m11_run = m11_run  # (v14 rebound it to g2p_m11_run; cfg["g2p"] stays off)


def m11_run(x, topk_ids, topk_weights, state, cfg):  # noqa: F811 (M11 at <= 32 tokens unless v16 serves the size)
    if v16_on(state, x.shape[0]):
        return moe_m13_run(x.contiguous(), topk_ids.to(torch.int32), topk_weights.to(torch.float32), state["v16_moe"],
                           V16_CFG)
    return _v16_m11_run(x, topk_ids, topk_weights, state, cfg)


def v16_case(tokens, device, gen):
    """Expert-stage inputs at `tokens`: normalized-scale activations, top-8 of 256 with about a quarter of the
    routes on this GPU's 32 experts (a judge-like share at 128 / 256 tokens, MF19), FP32 weights summing to 1."""
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=device)).to(torch.bfloat16)
    sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
    sc[:, :LOCAL_EXPERTS] += 0.35
    ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
    tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
    return x, ids, tw / tw.sum(-1, keepdim=True)


def v16_setup(state, weights):
    """Build, bank, check against D1's Marlin path at 128 and 256 tokens (< 2e-2; this is also every kernel's first
    launch, outside any capture), then install V16_M19. Any failure leaves the Marlin path (v15)."""
    import time as _v16_time
    state["v16_m19"] = {}
    picks = {below: bool(on) for below, on in V16_M19.items() if on}
    if not picks and not V16_FORCE:
        print("[v16] MoE E19 M19 path: not used (no picks)")
        return
    t0 = _v16_time.monotonic()
    try:
        if "m11" not in state:
            raise RuntimeError("no M11 bank")
        if not moe_m13_scales_ok(weights, 0, LOCAL_EXPERTS):
            raise RuntimeError("an E4M3 block scale is zero, subnormal or NaN (M13's decode needs normal scales)")
        ext = v16_build()
        ext.m19_setup()
        moe = {"m11_ext": ext, "m13": moe_m13_bank(state["m11"], weights["w13"].device), "sms": state["sms"],
               "norm_w": state["norm_w"]}
        state["v16_moe"] = moe
        gen = torch.Generator(device=weights["w13"].device).manual_seed(67)
        worst = 0.0
        for tokens in (32, 128, 256):  # 32: vs M11 (the path it replaces there); 128 / 256: vs Marlin
            x, ids, tw = v16_case(tokens, weights["w13"].device, gen)
            mine = moe_m13_run(x, ids, tw, moe, V16_CFG)
            m11cfg = m11_config_for(state, tokens)
            ref = (_v16_m11_run(x, ids, tw, state, m11cfg) if tokens <= 32 and m11cfg is not None
                   else _v16_d1_experts(x, tw, ids, state))
            worst = max(worst, _v3_rel(mine, ref))
        torch.cuda.synchronize()
        if not worst < 2e-2:
            raise RuntimeError(f"M19 path differs from the Marlin path: rel err {worst:.1e}")
        state["v16_m19"] = picks
        print(f"[v16] MoE E19 M19 path: built ({_v16_time.monotonic() - t0:.1f}s), worst rel err vs M11 / Marlin "
              f"{worst:.1e} at 32 / 128 / 256 tokens; installed {picks}")
    except Exception as exc:
        state.pop("v16_moe", None)
        print(f"[v16] MoE E19 M19 path: OFF ({type(exc).__name__}: {str(exc)[-200:]})")


# Prepare time for the M19 and G2P builds: bf16_proj's P9 (its own nvcc build, ~31 s in DE17) serves only the router
# at 16 tokens, which runs on the side stream behind the ~14 us pooler (DE17 profile: P9 5.8 + routing 2.7 us). The
# router there goes back to bf16_proj v2's judged Triton choice (proj/scored/submission_v2.py), still hidden; the
# projections' self-check in prepare compares it with SGLang like every other (layer, m) choice.
pj_P9_ENABLED = False
pj_CHOICE[("router", 16)] = {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4,
                             'stages': 4}

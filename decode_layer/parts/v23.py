# v23 = v22 with MoE's scored E49 (parts/moe49.py = make_embed_subset.py --which e49: moe/scored/submission_m13t_e49.py,
# md5 264dceac, scored 1.6958x on muse_moe_w4a16, job 6edb977e1d414efc) in place of E36: same glue hooks; E49 adds the
# dynamic-deal M11b (dyn_kernel: per-role __device__ counters reset by the last CTA) and hot-first GEMM1 / GEMM2 (hot
# launch first, the dynamic light side second with PDL and a late wait), split threshold 8. s128 / s256 = E49's bs128 /
# bs256 picks with M19 instead of its wgmma M18 for the GEMM2 hot side (E48: 379.6 vs 378.2 us at bs256). Rest as v22.
# Built under time pressure before the close: no lab; v17_setup checks M11b vs M11 and both splits vs Marlin before any
# capture and turns the path off on a mismatch.

# v22 = v20b with MoE's scored E36 (parts/moe36.py = make_embed_subset.py --which e36: moe/scored/submission_m13t_e36.py,
# md5 1ba91ab2, scored 1.6520x on muse_moe_w4a16, job 73e71876d57147a7) in place of E25 (parts/moe25.py). E36 keeps
# E25's bank, scale check, post-norm and m13_run (identical text) and adds E26's hot / cold split (experts with more than
# thr pairs run in a second, PDL launch; M11b takes a sel argument = thr | hot side << 16, 0 = every expert), E35's late
# wait on M19's hot GEMM1 (pf bit 29: it reads only x, so it computes during the cold launch's last wave) and E36's on
# GEMM2 (the cold GEMM2 signals its dependents only after its own wait, pf bit 15, set by m13_gemm2 when the hot GEMM2
# has lw; the hot GEMM2 then waits at its end). Our small classes run the same M11b shapes (sel 0) and the same M19
# cold instances; two hot M19 instances are added. New per-class kinds (V17_CHOICE values, from a lab):
#   "s128" at 65-191 tokens: MoE's picked split at bs128, cold M11b + hot M19 for GEMM1; GEMM2 cold M11b (lad, sel bit
#          18, set by m13_gemm2) + hot M19 with the late wait: the M19-only option MoE's E36 tuner timed within ~0.3% of
#          its M18 hot pick (296.3 us best at bs128; M18 is wgmma, compiled out here);
#   "s256" at 192-320 tokens: E36's pick at bs256, M19 cold + M19 hot with the late wait on GEMM1 and GEMM2.
# MoE's judge: bs128 1.434x -> 1.485-1.515x across E26-E36 (the split); E36's GEMM2 late wait -0.7% in-tuner at bs256.

m49_M11B_INSTANCES = V17_M11B_INSTANCES  # E36's m13_source reads these at call time: only what v22 uses
m49_M11B_OCC_INSTANCES = ()
m49_M11B_EARLY_INSTANCES = ()
m49_M11B_PF_INSTANCES = ()
m49_M11B_NU_INSTANCES = ()
m49_M11B_RECON_SHAPES = ()
m49_M11B_DYN_INSTANCES = V17_M11B_INSTANCES  # E49's dynamic-deal builds: our four shapes
m49_M13_M11_CR_INSTANCES = ()
m49_M13_M11_SR_INSTANCES = ()
m49_M11F_INSTANCES = ()
V22_M19_HOT = ((1, 2, 4, 2, 3), (0, 2, 4, 2, 4))  # (gated, tt, nt, P, MB): hot GEMM1 n4 c3 P2 MB3, hot GEMM2 t2 n4 P2 MB4
V17_M19_INSTANCES = V17_M19_INSTANCES + V22_M19_HOT
_V23_G1 = {"kind": "m11b", "nt": 1, "d": 4, "ks": 2, "c": 4, "dyn": 1, "hf": 1, "thr": 8,
           "hot": {"kind": "m19", "nt": 4, "c": 3, "p": 2, "mb": 3}}
_V23_G2 = {"kind": "m11b", "tt": 2, "nt": 1, "d": 4, "ks": 4, "c": 4, "dyn": 1, "pdl": 1, "hf": 1, "thr": 8,
           "hot": {"kind": "m19", "tt": 2, "nt": 4, "c": 4, "p": 2, "mb": 4}}
V22_CFGS = {  # E49's picks at bs128 and bs256 (hot-first, dynamic-deal M11b light side, thr 8), M19 for the GEMM2 hot side
    "s128": {"s1": _V23_G1, "s2": _V23_G2, "post": _V17_POST},
    "s256": {"s1": dict(_V23_G1), "s2": dict(_V23_G2), "post": _V17_POST},
}
V22_KIND_CLASS = {"s128": 192, "s256": 321}  # the class each split config serves (its bound in V17_CLASSES)


def v17_source():  # noqa: F811 (E36's source: M11b with sel, M19 cold + hot)
    return m49_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(), m16_instances=(),
                          m17_instances=(), m18_instances=(), m19_instances=V17_M19_INSTANCES)


def v17_build():  # noqa: F811
    return _v9_timed("e49_nvcc", load_inline)(
        name="moe49_ext", cpp_sources=m49__M11_CPP + m49__M13_CPP + m49__M11F_CPP, cuda_sources=v17_source(),
        functions=V17_FUNCTIONS, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DM13_NO_WGMMA"],
        build_directory=tempfile.mkdtemp(prefix="moe49_"), verbose=False)


def v17_cfg(state, tokens):  # noqa: F811 (+ the split kinds)
    if "v17_moe" not in state:
        return None
    picks = state.get("v17_choice") or {}
    for below, low in V17_CLASSES:
        if low <= tokens < below:
            kind = picks.get(below)
            if kind in V22_CFGS and V22_KIND_CLASS[kind] == below:
                return V22_CFGS[kind]
            if kind == "m19":
                return V17_CFGS["m19"]
            if kind == "m11b":
                return V17_CFGS["m11b_t1" if below == 5 else "m11b"]
            return None
    return None


def v17_run(x, topk_ids, topk_weights, state, cfg):  # noqa: F811
    return m49_m13_run(x.contiguous(), topk_ids.to(torch.int32), topk_weights.to(torch.float32), state["v17_moe"], cfg)


def v17_setup(state, weights):  # noqa: F811 (E36's path; checks + the split configs)
    """Build, bank, check M11b against our M11 path at 1 / 8 / 16 / 32 tokens, M19 against Marlin at 128 / 256, and the
    split configs against Marlin at 128 (s128) / 256 (s256) (< 2e-2; also every kernel's first launch, outside any
    capture), then install V17_CHOICE. Any failure leaves v15's paths."""
    import time as _v22_time
    state["v17_choice"] = {}
    picks = {b: k for b, k in V17_CHOICE.items() if k in ("m11b", "m19") or V22_KIND_CLASS.get(k) == b}
    if not picks and not V17_FORCE:
        print("[v23] MoE E49 path: not used (no choice)")
        return
    t0 = _v22_time.monotonic()
    try:
        if "m11" not in state:
            raise RuntimeError("no M11 bank")
        if not m49_m13_scales_ok(weights, 0, LOCAL_EXPERTS):
            raise RuntimeError("an E4M3 block scale is zero, subnormal or NaN (M13's decode needs normal scales)")
        ext = v17_build()
        ext.m19_setup()
        state["v17_moe"] = {"m11_ext": ext, "m13": m49_m13_bank(state["m11"], weights["w13"].device),
                            "sms": state["sms"], "norm_w": state["norm_w"]}
        gen = torch.Generator(device=weights["w13"].device).manual_seed(71)
        worst = {}
        cases = ((1, V17_CFGS["m11b_t1"], "m11b"), (8, V17_CFGS["m11b"], "m11b"), (16, V17_CFGS["m11b"], "m11b"),
                 (32, V17_CFGS["m11b"], "m11b"), (128, V17_CFGS["m19"], "m19"), (256, V17_CFGS["m19"], "m19"),
                 (128, V22_CFGS["s128"], "s128"), (256, V22_CFGS["s256"], "s256"))
        for tokens, cfg, name in cases:
            x, ids, tw = v17_case(tokens, weights["w13"].device, gen)
            mine = v17_run(x, ids, tw, state, cfg)
            m11cfg = m11_config_for(state, tokens)
            ref = (_v17_m11_run(x, ids, tw, state, m11cfg) if tokens <= 32 and m11cfg is not None
                   else _v17_d1_experts(x, tw, ids, state))
            worst[name] = max(worst.get(name, 0.0), _v3_rel(mine, ref))
        torch.cuda.synchronize()
        bad = {k: v for k, v in worst.items() if not v < 2e-2}
        if bad:
            raise RuntimeError(f"E49 path differs from ours: {bad}")
        state["v17_choice"] = picks
        print(f"[v23] MoE E49 path: built ({_v22_time.monotonic() - t0:.1f}s), worst rel err M11b vs M11 "
              f"{worst['m11b']:.1e} (1-32 tokens), M19 vs Marlin {worst['m19']:.1e}, split s128 {worst['s128']:.1e} / "
              f"s256 {worst['s256']:.1e} vs Marlin; installed {picks}")
    except Exception as exc:
        state.pop("v17_moe", None)
        print(f"[v23] MoE E49 path: OFF ({type(exc).__name__}: {str(exc)[-200:]})")


def v23_setup(state, weights):
    """v17_setup (above, in v17's slot) did the work; this logs which classes run E36's split."""
    split = {b: k for b, k in (state.get("v17_choice") or {}).items() if k in V22_CFGS}
    print(f"[v23] E49 split: {split or 'none'} (M19 instances {len(V17_M19_INSTANCES)}, M11b {len(V17_M11B_INSTANCES)})")


# bf16_proj PE45 (its message, Nov 5 ~03:15 UTC; judge lab job 96cdbcfa: F12M o m8 9.87 -> 9.47 us, unpooler m16 19.98 ->
# 19.20, qkv m16 16.78 -> 16.37, pooler m16 19.39 -> 19.02): pad F12M's split-K reduction buffer from row stride MT to
# MT + 3, so the fragment writes and the KSW-sum reads stop being 4-16-way bank-conflicted. Every access is array-indexed
# (part[ks][row][tok], part[p][row][tok]); same summation order, bit-identical outputs; static shared memory per CTA
# stays <= 20.5 KB for our five instances (occupancy unchanged). Our F12M is bf16_proj v6's source (parts/proj6.py).
V22_F12M_OLD = "__shared__ float part[KSW][RB][MT];"
V22_F12M_NEW = "__shared__ float part[KSW][RB][MT + 3];"
assert pj_F12M_SRC.count(V22_F12M_OLD) == 1 and V22_F12M_NEW not in pj_F12M_SRC
pj_F12M_SRC = pj_F12M_SRC.replace(V22_F12M_OLD, V22_F12M_NEW)


# Attention: the attention session's scored v18 (parts/attn18.py = make_attn.py --which attn18 from
# attention/scored/v18_submission.py, md5 5e6976a1, scored 1.8031x on muse_attention_decode, job 0b3cd2ff; v17 1.7973x,
# v15 1.7764x) in place of v14: UD8 on sliding decode 1 / 16 (UD5 = UD4 + padded warp partials, cache_loc at entry,
# 16-byte pushes; UD8 spreads each owned row's merge over every warp) and UD9 on sliding decode 32 (UD8 with a compact
# receive buffer: clusters of 8 at 3 CTAs per SM, one wave; its judge 11.83 vs 13.96 us), UW2h on sliding verify 1 / 16
# (FP16 pushes), the multi-row combine for calls with <= 4 splits (Sv32), D3 at 1 CTA per SM on full decode 1, and D3p on
# full decode 16 / 32 (D3 as a PDL dependent of the Triton prologue, PDL combine; Fd32 202.7 vs 210.7 us): its chain
# starts with a plain launch, so it still waits for our qkv projection. Its kinds UW2h / UD8 / UD9 register under
# `"CUDA_X_SRC" in globals()`: make_attn.py prefixes those strings (else they would not register here and the CUDA
# build would fall back). parts/v20.py's import-time assert covers the trimmed sources (17 unrouted kinds emptied).
V9_ATT_NAME = "attention v18"


def v20_setup(state, weights):  # noqa: F811 (v20's line, naming attention v18)
    routes = {f"{'F' if full else 'S'}{'v' if verify else 'd'}{b}": r[0]
              for (full, verify, b), r in sorted(att_CUDA_ROUTE.items())}
    print(f"[v20] attention v18 {'on' if state.get('v4_att') else 'OFF'} ({len(V6_ATT_CASES)} self-check cases); "
          f"CUDA module {'built' if 'mod' in att__CUDA else 'MISSING'}, routes after warm-up {routes}")

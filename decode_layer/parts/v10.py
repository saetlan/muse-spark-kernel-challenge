# v10 = v9 + the MoE session's E15 / E16 PDL path at the smallest M11 classes (parts/m11_pdl.py, from
# local/make_m11pdl.py): GEMM2 PDL-launched (it scans the routing while GEMM1 finishes, then waits before its
# activation loads) and M13's CUDA post-norm, PDL-launched (it reads the routing, then waits for GEMM2). Judged on
# muse_moe_w4a16: bs1 38.8 -> 35.8 us (MF16); at bs8 the post-norm alone passed MoE's 2% margin (MF17). Here a lab
# times each option on the whole layer and installs it per M11 class (V10_PDL); missing classes run as v9.

# M11 class bound (the `below` of V3_M11_CONFIGS) -> {"s2": 0 / 1 (GEMM2 with PDL), "post": None / 0 / 1 (CUDA
# post-norm, plain / PDL)}. Filled from a lab's printed V10_PDL; missing: as v9 (Triton post-norm, plain GEMM2).
V10_PDL = {}
V10_PDL_CLASSES = (5, 9)  # the classes whose GEMM2 config has an M11P instance (M11P_INSTANCES)


def v3_m11_setup(state, weights):
    """parts/v3.py's M11 build (no M12 instances), with parts/m11_pdl.py's PDL GEMM2 instances and launchers."""
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = _v9_timed("m11_nvcc", load_inline)(
        name="m11pdl_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(M11_INSTANCES, ()),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m11p_gemm", "m11_post_norm_cuda"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m11_"), verbose=False)
    state["m11"] = m11_bank(weights, 0, 32)


def v10_with(cfg, opt):
    """An M11 config with a V10_PDL option applied (a new dict; the base config is not changed)."""
    out = dict(cfg, s2=dict(cfg["s2"], pdl=int(opt.get("s2", 0))))
    if opt.get("post") is not None:
        out["post"] = {"pdl": int(opt["post"])}
    return out


def v10_install(state, picks):
    """state["m11_configs"] with picks (below -> option) applied to their classes; others as v9's."""
    base = state.get("v10_base_configs", state["m11_configs"])
    state["v10_base_configs"] = base
    state["m11_configs"] = tuple((below, v10_with(cfg, picks[below]) if cfg is not None and below in picks else cfg)
                                 for below, cfg in base)


def v10_setup(state, weights):
    """Check every PDL option against the plain path at its class (same kernels, so the outputs agree to
    rounding), warm the installed ones (no first launch inside a capture), then install V10_PDL. Any failure
    leaves every class as v9."""
    state["v10_pdl"], state["v10_ok"] = {}, False
    if "m11" not in state:
        print("[v10] PDL path: OFF (no M11)")
        return
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(47)
    try:
        worst, msgs = 0.0, []
        for below in V10_PDL_CLASSES:
            cfg = dict(state["m11_configs"]).get(below)
            if cfg is None:
                continue
            tokens = below - 1
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            sc[:, :LOCAL_EXPERTS] += 0.5
            ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            ref = m11_run(x, ids, tw, state, cfg)
            for opt in ({"s2": 0, "post": 0}, {"s2": 0, "post": 1}, {"s2": 1, "post": 1}, {"s2": 1, "post": None}):
                err = _v3_rel(m11_run(x, ids, tw, state, v10_with(cfg, opt)), ref)
                worst = max(worst, err)
            msgs.append(f"T{tokens}")
        torch.cuda.synchronize()
        if not worst < 1e-2:
            raise RuntimeError(f"PDL options differ from the plain path: rel err {worst:.1e}")
        picks = {below: dict(opt) for below, opt in V10_PDL.items() if below in V10_PDL_CLASSES}
        v10_install(state, picks)
        state["v10_pdl"], state["v10_ok"] = picks, True
        for below in picks:
            m11_warm(state, dict(state["m11_configs"])[below])
        print(f"[v10] PDL path: on ({'/'.join(msgs)} checked, worst rel err vs plain {worst:.1e}; installed {picks})")
    except Exception as exc:
        if "v10_base_configs" in state:
            state["m11_configs"] = state["v10_base_configs"]
        state["v10_pdl"] = {}
        print(f"[v10] PDL path: OFF ({type(exc).__name__}: {str(exc)[-200:]})")

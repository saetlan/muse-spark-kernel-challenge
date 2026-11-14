from torch.utils.cpp_extension import load_inline

"""F12M launch recon (PF41 follow-up): v10's own F12M kernels (CS = 1, F12M_SRC unchanged) in a second module whose
wrappers take a cluster size: LVC 0 = plain <<<>>> launch (= v10), LVC 1 / 2 = cudaLaunchKernelEx with a cluster-
dimension attribute of 1 / 2 CTAs (the kernel uses no cluster feature, so only the launch differs). DeepGEMM lost ~30%
to the 1-CTA attribute (PE23); if F12M loses the same, PE17's split-K (CS > 1, real clusters) was penalized by the
launch and a cluster-free split-K deserves a test; if LVC 2 costs nothing, clusters of 2 are safe for DG12-RS's
split-K reduction. Lab only: these entries never enter CHOICE."""

F12MLV_MOD = None
F12MLV_TARGETS = {
    # LVC "cs2": PE17's split-K over a real cluster of 2 (288 CTAs), to compare with LVC 2 (same launch, no split)
    ("qkv_gate", 16): [dict(NT=2, RTW=1, RG=1, KSW=8, D=2, F12=True, LVC=v) for v in (0, 1, 2, "cs2")],
    ("pooler", 16): [dict(NT=2, RTW=1, RG=1, KSW=8, D=2, F12=True, LVC=v) for v in (0, 1, 2)],
    ("qkv_gate", 8): [dict(NT=1, RTW=1, RG=1, KSW=8, D=4, F12=True, LVC=v) for v in (0, 2)],
}


def f12mlv_inst(c, k):
    if c["LVC"] == "cs2":
        return (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // 2, c["D"], c["F12"], 2)
    return (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"], c["D"], c["F12"])


def f12mlv_fn(c, k):
    inst = f12mlv_inst(c, k)
    return f12m_name(inst) if c["LVC"] == "cs2" else f12mlv_name(inst)


def f12mlv_name(inst):
    return "lv_" + f12m_name(inst)


def f12mlv_source(insts, cs_insts):
    """v10's F12M_SRC with f12m_source's own wrappers for the split-K instances, then the LVC wrappers (F12M_SRC ends with
    its @@INSTANCES@@ marker, so wrappers appended after it sit at global scope like f12m_source's)."""
    wrappers = []
    for inst in insts:
        nt, rtw, rg, ksw, ns, d, f12, cs, xd = f12m_unpack(inst)[:9]   # v10's F12M: 9 parameters (no TG)
        assert cs == 1 and xd == d and f12
        rb, threads, k = 16 * rg * rtw, 32 * rg * ksw, 64 * ns * ksw
        tmpl = f"{nt}, {rtw}, {rg}, {ksw}, {ns}, {d}, true, 1, {xd}"
        args = ("reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()), "
                "reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(m)")
        wrappers.append(f"""
void {f12mlv_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows,
                         int64_t cdim) {{
  TORCH_CHECK(n_rows > 0 && n_rows % {rb} == 0 && m >= 1 && m <= {8 * nt}, "f12mlv: rows or tokens out of range");
  TORCH_CHECK(x.numel() == m * {k} && out.numel() == m * n_rows && rec.numel() >= n_rows * 8
              && wp.numel() * wp.element_size() >= (n_rows / 16) * {ns * ksw * 1536}LL, "f12mlv: sizes");
  TORCH_CHECK(cdim >= 0 && cdim <= 2 && (cdim == 0 || (n_rows / {rb}) % cdim == 0), "f12mlv: cluster size");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (cdim == 0) {{
    f12m::f12m_kernel<{tmpl}><<<static_cast<unsigned>(n_rows / {rb}), {threads}, 0, stream>>>({args});
  }} else {{
    cudaLaunchConfig_t cfg = {{}};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = static_cast<unsigned>(cdim);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(static_cast<unsigned>(n_rows / {rb}), 1, 1);
    cfg.blockDim = dim3({threads}, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, f12m::f12m_kernel<{tmpl}>, {args});
    TORCH_CHECK(err == cudaSuccess, "f12mlv cluster launch failed: ", cudaGetErrorString(err));
  }}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}""")
    assert F12M_SRC.rstrip().endswith("@@INSTANCES@@")
    return f12m_source(cs_insts) + "\n" + "\n".join(wrappers) + "\n"


def f12mlv_setup():
    global F12MLV_MOD
    every = [(c, FULL[layer][1]) for (layer, m), cs in F12MLV_TARGETS.items() for c in cs]
    insts = tuple(dict.fromkeys(f12mlv_inst(c, k) for c, k in every if c["LVC"] != "cs2"))
    cs_insts = tuple(dict.fromkeys(f12mlv_inst(c, k) for c, k in every if c["LVC"] == "cs2"))
    decls = [f"void {f12mlv_name(i)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, "
             f"int64_t n_rows, int64_t cdim);" for i in insts] + f12m_decls(cs_insts)
    t0 = time.monotonic()
    try:
        F12MLV_MOD = load_inline(
            name="okc_proj_f12mlv", cpp_sources="\n".join(decls), cuda_sources=f12mlv_source(insts, cs_insts),
            functions=[dcl.split("(")[0].split()[-1] for dcl in decls],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
            build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mlv_"), verbose=False)
        info = " ".join(f"{f12m_name(i)}:{int(getattr(F12MLV_MOD, f12m_name(i) + '_info')()) // 1000000}r" for i in cs_insts)
        log(f"f12mlv build ok ({len(insts)} + {len(cs_insts)} split-K instances) in {time.monotonic() - t0:.1f}s {info}")
    except Exception as exc:
        F12MLV_MOD = None
        log(f"f12mlv build FAILED after {time.monotonic() - t0:.1f}s ({type(exc).__name__}): {_error_text(exc)}")


def run_f12mlv(c, x, layer, n, state):
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MLV_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if c["LVC"] == "cs2":
        getattr(F12MLV_MOD, f12mlv_fn(c, k))(pl["p"], pl["rec"], x, out, m, n)
    else:
        getattr(F12MLV_MOD, f12mlv_fn(c, k))(pl["p"], pl["rec"], x, out, m, n, c["LVC"])
    return out


_run_choice_pre_f12mlv = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mlv":
        return run_f12mlv(c, x, layer, n, state)
    return _run_choice_pre_f12mlv(c, x, layer, n, state)


def sec_f12mlv(lab, sweep, layer, m):
    if F12MLV_MOD is None or layer not in sweep.state.get("f12m", {}):
        return
    cands = [dict(kind="f12mlv", fallback=CHOICE.get((layer, m)), **c) for c in F12MLV_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results.pop((layer, m))   # lab only: never committed
    times = " ".join(f"lvc{c['LVC']}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: str(e[1]["LVC"])))
    log(f"f12mlv {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) cluster size:us {times}")

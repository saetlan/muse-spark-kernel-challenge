


# ---------------------------------------------------------------------------------------------
# UW2 anatomy (cuda_anatuw2.py, make_anatuw2.py): phase stamps of v14's Sv1 / Sv16 kernel, cold L2, inside a CUDA graph
# like the timing. Per phase, median / max over CTAs in ns (clock deltas scaled by each CTA's globaltimer / clock
# ratio): loads = entry -> seqlens-independent loads issued; rope = -> RoPE table; fold = -> Q tile stored; new = ->
# the block's new rows; full0 = -> stage 0 full; loop = -> consumer loop done; push = -> push stores issued; cbar = the
# cluster barrier; fin = the owner's finalize; skew / span over CTAs.
# ---------------------------------------------------------------------------------------------
def _anat_uw2(lab, k_cache, v_cache, gen, cases=((1, 6, 0, 16), (16, 4, 1, 4))):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        mod = load_inline(name="okc_anat_uw2", cpp_sources=CUDA_AN_UW2_DECL, cuda_sources=CUDA_AN_UW2_SRC,
                          functions=["an_uw2", "an_uw2_setup", "an_uw2_attrs"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_anat_uw2_"), verbose=False)
        if mod.an_uw2_setup():
            raise RuntimeError("an_uw2_setup refused")
    except Exception as exc:
        _log(f"anat uw2: build FAILED {_why(exc)}")
        return
    for batch, stages, pp, cs in cases:
        if lab.left() < 20:
            _log("anat uw2: out of budget")
            break
        try:
            data = _lab_inputs(batch, 8, 2048, k_cache, v_cache, gen)
            cfg = data["config"]
            dev = data["qkv"].device
            ctas = cs * batch * (cfg["num_kv_heads"] // 2)
            an = torch.zeros((ctas * 32,), dtype=torch.int64, device=dev)
            out = torch.empty((batch, 8, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=dev)
            inv = _inv_freq(cfg["rope_theta"], dev)

            def call():
                mod.an_uw2(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
                           data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch,
                           cfg["num_q_heads"], cfg["num_kv_heads"], data["page_table"].shape[1], cfg["page_size"],
                           cfg["window"], 1, cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs, an)
            us = lab.time_graph(call)
            a = an.view(ctas, 32).cpu().double()
            gt0, c1, c28, gt29 = a[:, 0], a[:, 1], a[:, 28], a[:, 29]
            ns_per_clk = (gt29 - gt0) / (c28 - c1).clamp_min(1)
            rows = []
            for name, i, j in (("loads", 1, 2), ("rope", 2, 3), ("fold", 3, 4), ("new", 4, 5), ("full0", 5, 6),
                               ("loop", 6, 7), ("push", 7, 8), ("cbar", 8, 9), ("fin", 9, 28)):
                d = (a[:, j] - a[:, i]) * ns_per_clk
                rows.append(f"{name} {float(d.median()):.0f}/{float(d.max()):.0f}")
            skew = gt0 - gt0.min()
            span = float(gt29.max() - gt0.min())
            _log(f"anat UW2 bs{batch} s{stages} pp{pp} cs{cs} ({us:.2f}us raw graph; ns median/max over {ctas} CTAs; "
                 f"stages {int(a[0, 13])}): " + " ".join(rows) + f" | skew {float(skew.median()):.0f}/{float(skew.max()):.0f} | span {span:.0f}")
        except Exception as exc:
            _log(f"anat UW2 bs{batch}: FAILED {_why(exc)}")




# ---------------------------------------------------------------------------------------------
# UD5 anatomy (cuda_anatud5.py, make_anatud5.py): phase stamps of v15's Sd1 / Sd16 / Sd32 kernels, cold L2, inside a CUDA
# graph like the timing. Per phase: median and max over CTAs, in ns (clock deltas scaled by each CTA's own
# globaltimer / clock ratio). Phases: n = entry -> seqlens arrived; rope = -> RoPE table; pt = -> page entries
# landed; fold = -> query fragments; k0 = -> warp 0's first chunk; loop = -> warp 0's loop end (wait = its K/V wait
# share); wmerge = -> partials in smem; wts = -> CTA weights; push = -> after the push barrier; fin = leader finalize;
# skew = CTA start after the first CTA; span = first start -> last exit.
# ---------------------------------------------------------------------------------------------
def _anat_ud5(lab, k_cache, v_cache, gen, cases=((1, 16, 4, 3), (16, 8, 4, 3), (32, 4, 4, 3))):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        mod = load_inline(name="okc_anat_ud5", cpp_sources=CUDA_AN_UD5_DECL, cuda_sources=CUDA_AN_UD5_SRC,
                          functions=["an_ud5", "an_ud5_setup", "an_ud5_attrs"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_anat_ud5_"), verbose=False)
        if mod.an_ud5_setup():
            raise RuntimeError("an_ud5_setup refused")
    except Exception as exc:
        _log(f"anat ud5: build FAILED {_why(exc)}")
        return
    for batch, cs, warps, stages in cases:
        if lab.left() < 20:
            _log("anat ud5: out of budget")
            break
        try:
            data = _lab_inputs(batch, 1, 2048, k_cache, v_cache, gen)
            cfg = data["config"]
            dev = data["qkv"].device
            ctas = cs * batch * (cfg["num_kv_heads"] // 2)
            an = torch.zeros((ctas * 32,), dtype=torch.int64, device=dev)
            out = torch.empty((batch, 1, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=dev)
            inv = _inv_freq(cfg["rope_theta"], dev)

            def call():
                mod.an_ud5(data["qkv"], data["positions"], data["cache_loc"], inv, data["k_cache"], data["v_cache"],
                           data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch,
                           cfg["num_q_heads"], cfg["num_kv_heads"], data["page_table"].shape[1], cfg["page_size"],
                           cfg["window"], 1, cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs, an)
            us = lab.time_graph(call)
            a = an.view(ctas, 32).cpu().double()
            gt0, c1, c28, gt29 = a[:, 0], a[:, 1], a[:, 28], a[:, 29]
            ns_per_clk = (gt29 - gt0) / (c28 - c1).clamp_min(1)

            def ph(i, j):   # clock slot j - slot i, in ns
                d = (a[:, j] - a[:, i]) * ns_per_clk
                return d
            rows = []
            for name, i, j in (("n", 1, 2), ("rope", 2, 4), ("pt", 4, 3), ("fold", 3, 5), ("k0", 5, 6), ("loop", 6, 7),
                               ("wmerge", 7, 9), ("wts", 9, 10), ("push", 10, 11), ("fin", 11, 28)):
                d = ph(i, j)
                rows.append(f"{name} {float(d.median()):.0f}/{float(d.max()):.0f}")
            wait = a[:, 8] * ns_per_clk
            leader = (torch.arange(ctas) % cs) == 0
            fin_lead = ph(11, 28)[leader]
            skew = gt0 - gt0.min()
            span = float(gt29.max() - gt0.min())
            _log(f"anat UD5 bs{batch} cs{cs} w{warps}s{stages} ({us:.2f}us raw graph; ns median/max over {ctas} CTAs; "
                 f"chunks {int(a[0, 13])}): " + " ".join(rows) + f" | wait {float(wait.median()):.0f} | leader fin "
                 f"{float(fin_lead.median()):.0f} | skew {float(skew.median()):.0f}/{float(skew.max()):.0f} | span {span:.0f}")
        except Exception as exc:
            _log(f"anat UD5 bs{batch}: FAILED {_why(exc)}")

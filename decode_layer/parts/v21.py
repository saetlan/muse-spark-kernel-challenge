# v21 = v20 + bf16_proj v11b's DG12-RS for the pooler and unpooler at 32 tokens (parts/dgrs.py = make_embed_subset.py
# --which dgrs: proj/scored/submission_v11b.py sha cfe20913, scored 1.3511x on muse_bf16_proj, job ee5b3e65; kernel by
# muse_prefill_layer, pinned proj/dg/pinned/dgrs_v10.py). DG12-RS reads 12-bit lossless planes of the BF16 weights
# (bf16_proj's F12M encoder in natural k order: bit-exact weights after decode; a layer whose planes don't fit keeps its
# kernel) into register-A wgmma with split-K over a cluster (unpooler: S 2, push reduction) or global partials and a
# last-arriver counter (pooler: S 4); fully specialized for M 32. On bf16_proj's judge (us, m 32): unpooler 15.97 vs
# SGLang's cuBLAS 18.69, pooler 19.27 vs 21.25. In our layer at 32 tokens the unpooler is cuBLAS and the pooler bf16_proj
# v6's swap kernel (21.6 us per call in DE23's profile). Served per (layer, tokens) from a lab's V21_DGRS, each checked
# against F.linear (< 1e-2, twice: the counters must reset) and launched in prepare before any capture.

V21_DGRS = {}  # (layer, tokens) -> True where DG12-RS serves; from a lab's printed V21_DGRS
V21_FORCE = False  # a lab sets it: build and check every shape of pb_DGRS_CHOICE
V21_TOL = 1e-2


def v21_run(entry, x):
    c = entry["c"]
    out = torch.empty((x.shape[0], entry["n"]), dtype=torch.bfloat16, device=x.device)
    return pb_dgrs_run(entry["ext"], x.contiguous(), entry["wp"], entry["rec"], out, entry["bufs"], c["BM"], c["S"],
                       cps=c.get("CPS", 1), red=c.get("RED", 0), math=c.get("MATH", 2), diag=c.get("DIAG", 0))


def v21_setup(state, weights):
    """Planes, one nvcc build of the chosen specialized instances, scratch per shape, and a check of each shape against
    F.linear; then V21_DGRS (the lab's picks) is installed. Any failure leaves v20's projections."""
    import time as _v21_time
    state["v21_ready"], state["v21_on"] = {}, set()
    picks = {key for key, on in V21_DGRS.items() if on and key in pb_DGRS_CHOICE}
    want = set(pb_DGRS_CHOICE) if V21_FORCE else picks
    if not want:
        print("[v21] DG12-RS: not used (no picks)")
        return
    t0 = _v21_time.monotonic()
    try:
        if "pj_state" not in state:
            raise RuntimeError("bf16_proj's projections are off")
        planes = {}
        for layer in sorted({layer for layer, _m in want}):
            pl = pb_f12m_device_planes(state["dense"][layer], natural=True)
            if pl is None:
                print(f"[v21] DG12-RS: {layer}'s planes don't fit (a row with > 7 escapes); it keeps its kernel")
                continue
            planes[layer] = (pb_dgrs_pack(pl["p"]), pl["rec"])
        want = {key for key in want if key[0] in planes}
        if not want:
            raise RuntimeError("no layer's planes fit")
        ext = _v9_timed("dgrs_nvcc", pb_dgrs_build)(
            tuple(sorted({pb_dgrs_inst(pb_DGRS_CHOICE[key], *key) for key in want})))
        gen = torch.Generator(device=state["dense"]["pooler"].device).manual_seed(87)
        worst = {}
        for layer, m in sorted(want):
            c = pb_DGRS_CHOICE[(layer, m)]
            w = state["dense"][layer]
            n, k = w.shape
            assert (c["M"], c["N"], c["K"]) == (m, n, k), (layer, m, c, w.shape)
            entry = {"ext": ext, "wp": planes[layer][0], "rec": planes[layer][1], "c": c, "n": n,
                     "bufs": pb_dgrs_buffers(m, n, c["BM"], c["S"], w.device)}
            err = 0.0
            for _rep in range(2):  # the second call runs on the counters the first one left (they must be back to 0)
                x = torch.randn((m, k), generator=gen, device=w.device).to(torch.bfloat16)
                err = max(err, _v3_rel(v21_run(entry, x), F.linear(x, w)))
            torch.cuda.synchronize()
            flag = int(entry["bufs"]["err"].item())
            worst[f"{layer} {m}"] = err
            if err < V21_TOL and flag == 0:
                state["v21_ready"][(layer, m)] = entry
            else:
                print(f"[v21] DG12-RS {layer} m{m}: dropped (rel err {err:.1e}, error flag {flag})")
        state["v21_on"] = {key for key in picks if key in state["v21_ready"]}
        print(f"[v21] DG12-RS: built ({_v21_time.monotonic() - t0:.1f}s), rel err vs F.linear "
              + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + f"; serving {sorted(state['v21_on'])}")
    except Exception as exc:
        state["v21_ready"], state["v21_on"] = {}, set()
        print(f"[v21] DG12-RS: OFF ({type(exc).__name__}: {str(exc)[-200:]})")


_v21_project = v4_project


def v4_project(x, layer, n, cfg, state):  # noqa: F811 (DG12-RS where V21_DGRS serves the shape, else v20's)
    on = state.get("v21_on")
    if on and (layer, x.shape[0]) in on and v4_mode(cfg, state, "pj"):
        entry = state["v21_ready"][(layer, x.shape[0])]
        if n == entry["n"]:
            return v21_run(entry, x)
    return _v21_project(x, layer, n, cfg, state)

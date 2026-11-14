"""RE1 lab (check runs only): Marlin tile heights at prefill sizes, routing stats from the real
router, and a BF16 grouped-GEMM ceiling (torch._grouped_mm) for the expert GEMMs.
Appended after submission.py by local/make_lab.py; kernel_fn stays the submission's."""

RE1_BUDGET_S = 170.0
RE1_TOKENS = (753, 3012, 16384)
RE1_TILES = (8, 16, 32, 48, 64)


def re1_real_routing(weights, tokens, seed, device="cuda"):
    """Random router inputs (a shared mean direction plus noise, RMS-normalized like h) through the
    real FP32 router and bias, top-8, renormalized; rotated so experts 0-31 are the busiest GPU's."""
    gen = torch.Generator(device=device).manual_seed(seed)
    mean = torch.randn((1, HIDDEN), generator=gen, device=device)
    h = torch.randn((tokens, HIDDEN), generator=gen, device=device) + mean
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    scores = torch.sigmoid(0.1767766952966369 * (h @ weights["router"].float().T))
    ids = (scores + weights["router_bias"].float()).topk(TOP_K, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True)
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), w.float()


def re1_stats(ids):
    local = ids[ids < LOCAL_EXPERTS].long()
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS)
    return (f"local routes {int(local.numel())} ({local.numel() / ids.numel():.3f} of all), distinct "
            f"{int((counts > 0).sum())}, per expert mean {local.numel() / LOCAL_EXPERTS:.1f} max {int(counts.max())}")


def re1_routings(lab, weights):
    out = {}
    for tokens in RE1_TOKENS:
        real = re1_real_routing(weights, tokens, seed=tokens)
        synth = synthetic_routing(tokens, seed=tokens)
        log(f"routing T={tokens} real-router: {re1_stats(real[0])}")
        log(f"routing T={tokens} synthetic : {re1_stats(synth[0])}")
        out[tokens] = {"real": real, "synth": synth}
    return out


def sec_marlin_tiles(lab, state, routings):
    """_experts (both Marlin GEMMs, silu_and_mul, post-norm) per tile height, cold-L2 graph time."""
    for tokens in RE1_TOKENS:
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = (torch.randn((tokens, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        for kind, (ids, w) in routings[tokens].items():
            ref = _experts(x, w, ids, state, block_size_m=64).float()
            row = []
            for tile in RE1_TILES:
                if lab.left() < 20:
                    log(f"marlin T={tokens} {kind}: budget low, stopping")
                    return
                out = _experts(x, w, ids, state, block_size_m=tile).float()
                err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
                us = lab.time_graph(lambda: _experts(x, w, ids, state, block_size_m=tile), reps=10)
                row.append(f"{tile}:{us:.1f}us(err{err:.1e})")
            log(f"marlin T={tokens} {kind} v1-pick={moe_block_size(tokens)}: " + " ".join(row))


def sec_grouped_mm(lab, routings):
    """BF16 grouped GEMMs with the expert shapes (GEMM1 [R,4096]x[4096,8192], GEMM2 [R,4096]x[4096,4096])
    over the real-router routing's local routes, sorted by expert: the ceiling of a wgmma grouped GEMM
    without the NVFP4 decode. Random BF16 weights (timing only)."""
    gm = getattr(torch, "_grouped_mm", None)
    if gm is None:
        log("grouped_mm: torch._grouped_mm missing")
        return
    gen = torch.Generator(device="cuda").manual_seed(3)
    w13 = torch.randn((LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT), generator=gen, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((LOCAL_EXPERTS, D_EXPERT, D_EXPERT), generator=gen, device="cuda", dtype=torch.bfloat16)
    for tokens in RE1_TOKENS:
        ids = routings[tokens]["real"][0]
        local = ids[ids < LOCAL_EXPERTS].long()
        counts = torch.bincount(local, minlength=LOCAL_EXPERTS)
        for pad in (1, 16, 128):
            padded = (counts + pad - 1) // pad * pad
            rows = int(padded.sum())
            offs = padded.cumsum(0).to(torch.int32)
            a = (torch.randn((rows, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            mid = torch.empty((rows, D_EXPERT), device="cuda", dtype=torch.bfloat16)

            def run():
                h1 = gm(a, w13.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
                mid.copy_(h1[:, :D_EXPERT])
                return gm(mid, w2.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)

            try:
                y = run()
                # Spot-check the first expert's group against a plain matmul.
                n0 = int(padded[0])
                if n0:
                    h_ref = a[:n0].float() @ w13[0].float().T
                    e = float((gm(a, w13.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)[:n0].float()
                               - h_ref).norm() / h_ref.norm())
                else:
                    e = float("nan")
                us = lab.time_graph(run, reps=10)
                flops = 2.0 * int(counts.sum()) * (2 * D_EXPERT * D_EXPERT + D_EXPERT * D_EXPERT)
                log(f"grouped_mm T={tokens} pad{pad}: rows {rows}, {us:.1f}us, {flops / us / 1e6:.0f} TFLOP/s "
                    f"(real rows), group0 err {e:.1e}")
                del y
            except Exception as exc:
                log(f"grouped_mm T={tokens} pad{pad}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
    del w13, w2


def sec_env(lab):
    import importlib.util
    log(f"torch {torch.__version__}, cuda {torch.version.cuda}, triton {triton_version()}, "
        f"gpu {torch.cuda.get_device_name()}, sms {torch.cuda.get_device_properties(0).multi_processor_count}")
    for name in ("flash_attn_interface", "sgl_kernel", "deep_gemm", "flashinfer", "cutlass"):
        try:
            found = "found" if importlib.util.find_spec(name) else "missing"
        except Exception as exc:
            found = f"lookup failed ({type(exc).__name__})"
        log(f"module {name}: {found}")


def triton_version():
    import importlib.metadata
    try:
        return importlib.metadata.version("triton")
    except Exception:
        return "?"


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"submission prepare: {time.monotonic() - t0:.1f}s")
    lab = Lab(RE1_BUDGET_S)
    lab.section("env", sec_env)
    holder = {}
    lab.section("routings", lambda lab_: holder.update(r=re1_routings(lab_, weights)))
    if "r" in holder:
        lab.section("marlin_tiles", sec_marlin_tiles, state, holder["r"])
        lab.section("grouped_mm", sec_grouped_mm, holder["r"], min_left=25.0)
    holder.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"lab total {time.monotonic() - t0:.1f}s")
    return state

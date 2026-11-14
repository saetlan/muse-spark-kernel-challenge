"""DE3 lab (check runs only), on top of submission_v2 + parts/x5.py (X5 v2: weights, tile bases and token rows all
staged by bulk copies; loader-only mode). X5 (our bulk-copy mma.sync skinny
GEMM) and X10 (12-bit lossless BF16 weights) on the five dense weights, against cuBLAS. Log kept short;
summary last.

build   nvcc for the 12 X5 instances; FlashInfer frost probe (X13).
pack    plain + 12-bit packs of qkv, o, pooler, unpooler slice, unpooler; raw-tile share and bytes.
correct each pack x plain/enc at 1/8/16/32 tokens vs FP32 x @ W^T (rel err), before any timing.
time    per weight, 1/8/16 tokens: cuBLAS (bf16 out) vs X5 plain and 12-bit (1-8 tokens: nc 4/8 x 1/2 CTAs per
        SM; 16 tokens: nc 8, 1 CTA per SM), kernel only, plus loader-only (no math) at 1 token: the
        bulk-copy ceiling.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def sec_build(lab, store, deadline):
    t = time.monotonic()
    store["ext"] = x5_build()
    store["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
    log(f"x5 build {time.monotonic() - t:.1f}s; {store['sms']} SMs")
    try:
        import importlib
        m = importlib.import_module("flashinfer.fused_moe.backends.frost_low_latency")
        log(f"frost: present ({', '.join(sorted(n for n in dir(m) if not n.startswith('_'))[:12])})")
    except Exception as exc:
        log(f"frost: absent ({type(exc).__name__}: {_error_text(exc, 100)})")


def sec_pack(lab, state, store, deadline):
    dense = state["dense"]
    weights = {"qkv": dense["qkv_gate"], "o": dense["o"], "pool": dense["pooler"],
               "unps": dense["unpooler"][:UNPOOL_SLICE].contiguous(), "unp": dense["unpooler"]}
    store["packs"] = {}
    cells = []
    for name, w in weights.items():
        t = time.monotonic()
        pk = x5_pack(w)
        pk["w"] = w
        store["packs"][name] = pk
        ratio = x5_enc_bytes(pk) / (w.numel() * 2)
        cells.append(f"{name} raw {100 * pk['raw_tiles'] / pk['tiles']:.2f}% bytes {ratio:.3f} ({time.monotonic() - t:.1f}s)")
    torch.cuda.synchronize()
    log("x5 packs: " + "; ".join(cells))


def _run(store, pk, x, enc, nc, cps, out=None, mode=0):
    t = x.shape[0]
    nt = 1 if t <= 8 else 2
    return x5_gemv(x, pk, store["ext"], enc, nc, nt, store["sms"] * cps, out, mode)


def _kern(store, pk, x, out, enc, nc, cps, mode=0):
    t = x.shape[0]
    store["ext"].x5_gemv(pk["enc_b"] if enc else pk["plain"], pk["enc_i"], pk["bases"], pk["plain"], x, out,
                         pk["n"], pk["k"], t, enc, nc, 1 if t <= 8 else 2, store["sms"] * cps, mode)


def sec_correct(lab, store, deadline):
    bad = []
    worst = 0.0
    for name, pk in store["packs"].items():
        for t in (1, 8, 16):
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            ref = x.float() @ pk["w"].float().T
            for enc in (0, 1):
                for nc in ((4, 8) if t <= 8 else (8,)):
                    got = _run(store, pk, x, enc, nc, 1)
                    err = float((got - ref).norm() / ref.norm())
                    worst = max(worst, err)
                    if not err < 1e-4:
                        bad.append(f"{name} T{t} enc{enc} nc{nc} {err:.1e}")
    torch.cuda.synchronize()
    store["ok"] = not bad
    log(f"x5 correct: worst rel err {worst:.1e}" + ("" if not bad else " | BAD: " + ", ".join(bad[:8])))
    LAB_SUMMARY.append(f"x5 correct {'yes' if not bad else 'NO'} (worst {worst:.1e})")


def sec_time(lab, store, deadline):
    if not store.get("ok"):
        LAB_SUMMARY.append("x5 timing skipped (correctness failed)")
        return
    fixed = lab.empty_graph_us()
    for name, pk in store["packs"].items():
        nbytes = pk["w"].numel() * 2
        ebytes = x5_enc_bytes(pk)
        w = pk["w"]
        cells = []
        for t in (1, 8, 16):
            if lab.elapsed() > deadline:
                LAB_SUMMARY.append(f"x5 timing stopped at {name} T{t}")
                break
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            out = torch.empty((t, pk["n"]), dtype=torch.float32, device="cuda")
            try:
                cub = lab.time_graph(lambda: F.linear(x, w)) - fixed
                cfgs = [(nc, cps) for nc in (4, 8) for cps in (1, 2)] if t <= 8 else [(8, 1)]
                res = {}
                for enc in (0, 1):
                    best = None
                    for nc, cps in cfgs:
                        us = lab.time_graph(lambda: _kern(store, pk, x, out, enc, nc, cps)) - fixed
                        if best is None or us < best[0]:
                            best = (us, nc, cps)
                    res[enc] = best
                p, e = res[0], res[1]
                cell = (f"T{t} cub {cub:.1f}({nbytes / max(cub, .01) / 1e6:.2f}) plain {p[0]:.1f}({nbytes / max(p[0], .01) / 1e6:.2f})"
                        f"n{p[1]}c{p[2]} enc {e[0]:.1f}({ebytes / max(e[0], .01) / 1e6:.2f}r)n{e[1]}c{e[2]}")
                if t == 1:
                    lp = lab.time_graph(lambda: _kern(store, pk, x, out, 0, 8, 2, 1)) - fixed
                    le = lab.time_graph(lambda: _kern(store, pk, x, out, 1, 8, 2, 1)) - fixed
                    cell += f" load-only plain {lp:.1f}({nbytes / max(lp, .01) / 1e6:.2f}) enc {le:.1f}({ebytes / max(le, .01) / 1e6:.2f}r)"
                cells.append(cell)
                LAB_SUMMARY.append(f"{name}T{t} cub {cub:.1f} plain {p[0]:.1f} enc {e[0]:.1f}")
            except Exception as exc:
                lab_fail(f"time {name} T{t}", exc)
        log(f"x5 {name} [{pk['n']}x{pk['k']}] " + " | ".join(cells))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v2 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s")
    store = {}
    lab.section("build", sec_build, store)
    if "ext" in store:
        lab.section("pack", sec_pack, state, store)
        if "packs" in store:
            lab.section("correct", sec_correct, store)
            lab.section("time", sec_time, store)
    store.clear()
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state

"""DE2 lab (check runs only), on top of submission_v2 + parts/x5.py: X5 (our bulk-copy mma.sync skinny
GEMM) and X10 (12-bit lossless BF16 weights) on the five dense weights, against cuBLAS. Log kept short;
summary last.

build   nvcc for the 12 X5 instances; FlashInfer frost probe (X13).
pack    plain + 12-bit packs of qkv, o, pooler, unpooler slice, unpooler; raw-tile share and bytes.
correct each pack x plain/enc at 1/8/16/32 tokens vs FP32 x @ W^T (rel err), before any timing.
time    per weight and token count: cuBLAS (bf16 out) vs X5 plain and X5 12-bit at nc 4/8 x 1/2 CTAs
        per SM; "tot" = zero + kernel + BF16 rounding as standalone ops, "k" = kernel only.
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


def _run(store, pk, x, enc, nc, cps, out=None):
    t = x.shape[0]
    nt = 1 if t <= 8 else 2 if t <= 16 else 4
    return x5_gemv(x, pk, store["ext"], enc, nc, nt, store["sms"] * cps, out)


def sec_correct(lab, store, deadline):
    bad = []
    worst = 0.0
    for name, pk in store["packs"].items():
        for t in (1, 8, 16, 32):
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            ref = x.float() @ pk["w"].float().T
            for enc in (0, 1):
                for nc in (4, 8):
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
        for t in (1, 8, 16, 32):
            if lab.elapsed() > deadline:
                LAB_SUMMARY.append(f"x5 timing stopped at {name} T{t}")
                return
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            w = pk["w"]
            try:
                cub = lab.time_graph(lambda: F.linear(x, w)) - fixed
                best = {}
                for enc in (0, 1):
                    for nc in (4, 8):
                        for cps in (1, 2):
                            out = torch.empty((t, pk["n"]), dtype=torch.float32, device="cuda")
                            tot = lab.time_graph(lambda: _run(store, pk, x, enc, nc, cps, out).to(torch.bfloat16)) - fixed
                            if enc not in best or tot < best[enc][0]:
                                best[enc] = (tot, nc, cps)
                    tot, nc, cps = best[enc]
                    nt = 1 if t <= 8 else 2 if t <= 16 else 4
                    out = torch.empty((t, pk["n"]), dtype=torch.float32, device="cuda")
                    kern = lab.time_graph(lambda: store["ext"].x5_gemv(
                        pk["enc_b"] if enc else pk["plain"], pk["enc_i"], pk["bases"], pk["plain"], x, out,
                        pk["n"], pk["k"], t, enc, nc, nt, store["sms"] * cps)) - fixed
                    best[enc] = (tot, nc, cps, kern)
                p, e = best[0], best[1]
                line = (f"x5 {name} T{t}: cub {cub:.1f} ({nbytes / max(cub, .01) / 1e6:.2f}TB/s) | plain {p[0]:.1f} k{p[3]:.1f} "
                        f"({nbytes / max(p[3], .01) / 1e6:.2f}) nc{p[1]}c{p[2]} | enc {e[0]:.1f} k{e[3]:.1f} "
                        f"({ebytes / max(e[3], .01) / 1e6:.2f} real, {nbytes / max(e[3], .01) / 1e6:.2f} eff) nc{e[1]}c{e[2]}")
                log(line)
                LAB_SUMMARY.append(f"{name}T{t} cub {cub:.1f} plain {p[0]:.1f}/{p[3]:.1f} enc {e[0]:.1f}/{e[3]:.1f}")
            except Exception as exc:
                lab_fail(f"time {name} T{t}", exc)


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

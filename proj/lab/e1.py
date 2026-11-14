"""PE1: per scored shape, under CUDA graphs with a cold L2, time SGLang's path against
  triton  our _swap_kernel (m <= 32) and _gemm_kernel (m >= 128) over tiles, split-K, stages
  gemv    the image's hopper_bf16_gemv (m = 1, BF16 layers)
  tiny    the image's tiny_gemm_bf16 (m <= 32, BF16 layers)
Every candidate is checked against FP64 (with probe rows) before it is timed; a winner must be
correct on normal and outlier inputs and at least MIN_GAIN faster than SGLang in this lab, and
then goes into CHOICE, so the run's timed kernel_fn uses it. Everything else stays on SGLang."""

import math

LAB_BUDGET_S = 215.0
MIN_GAIN = 1.03
SCORED_M = (1, 8, 16, 32, 128, 256)
LAYER_ORDER = ("pooler", "unpooler", "qkv_gate", "o", "router")
FULL = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
        "router": (256, 8192)}


def lab_n(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else FULL[layer][0]


def lab_input(m, k, seed, dist="normal"):
    """Like the judge's generate_input: standard normal, ~1 row in 8 a single power of two."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda")
    if dist == "outliers":
        idx = torch.randperm(k, generator=gen, device="cuda")[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device="cuda") < 1 / 8
    if m >= 8:
        probes[torch.randint(m, (1,), generator=gen, device="cuda")] = True
    cols = torch.randint(k, (m,), generator=gen, device="cuda")
    signs = torch.randint(2, (m,), generator=gen, device="cuda") * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device="cuda")
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16)


def lab_errors(out, x, w):
    """(relative L2, worst row relative to max(own norm, 0.1 x mean), worst probe row) vs FP64."""
    ref = x.double() @ w.double().T
    ref = ref.to(torch.float32 if w.dtype == torch.float32 else torch.bfloat16).double()
    if out.shape != ref.shape or out.dtype != (torch.float32 if w.dtype == torch.float32 else torch.bfloat16):
        return math.inf, math.inf, math.inf
    o = out.double()
    if not torch.isfinite(o).all():
        return math.inf, math.inf, math.inf
    diff = o - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    row = float((diff.norm(dim=-1) / torch.maximum(rows_ref, 0.1 * rows_ref.mean())).max())
    probe = ((x != 0).sum(-1) == 1).nonzero().flatten()
    pr = float((diff[probe].norm(dim=-1) / rows_ref[probe].clamp(min=1e-30)).max()) if len(probe) else 0.0
    return rel, row, pr


def lab_ok(errs, layer):
    tol = 1e-4 if layer == "router" else 1e-2
    rel, row, pr = errs
    # Half the judge's bounds, so a winner has margin on the judge's own inputs.
    return rel <= 0.5 * tol and row <= 2.5 * tol and pr <= 0.5e-3


def triton_candidates(layer, m, n, k):
    router = layer == "router"
    cands = []
    if m <= 32:
        if router and m <= 8:
            modes = (2,)  # exact FP32 below 9 tokens, as SGLang computes it
        else:
            modes = (1,) if router else (0,)
        bm = 16 if m <= 16 else 32
        for mode in modes:
            for bn in (16, 32, 64):
                tiles = -(-n // bn)
                for bk in ((64, 128) if mode == 2 else (128, 256)):
                    for split in (1, 2, 4, 8, 16, 32):
                        if k % (split * bk) or not (64 <= tiles * split <= 1200):
                            continue
                        cands.append({"kind": "swap", "BM": bm, "BN": bn, "BK": bk, "split": split,
                                      "mode": mode, "warps": 4, "stages": 4})
    else:
        mode = 1 if router else 0
        tiles_m = -(-m // 128)
        for bm, bn, bk, warps, stages in ((128, 128, 64, 8, 4), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4),
                                          (128, 256, 64, 8, 3), (64, 64, 64, 4, 4), (256, 128, 64, 8, 3)):
            if bm > max(64, m):
                continue
            tiles = -(-m // bm) * -(-n // bn)
            for split in (1, 2, 4, 8, 16):
                if k % (split * bk) or (split > 1 and tiles * split > 1200) or (split > 1 and tiles >= 132):
                    continue
                cands.append({"kind": "gemm", "BM": bm, "BN": bn, "BK": bk, "split": split, "mode": mode,
                              "warps": warps, "stages": stages})
        del tiles_m
    return cands


def image_candidates(layer, m, n, k):
    if layer == "router" or m > 32:
        return []
    cands = []
    if m == 1:
        cands.append({"kind": "gemv"})
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm
        max_m = 16 if m <= 16 else 32
        if can_use_tiny_gemm(n, k, max_m):
            cands.append({"kind": "tiny", "max_m": max_m})
    except Exception as exc:
        log(f"tiny_gemm unavailable: {_error_text(exc, 200)}")
    return cands


def cfg_name(c):
    if c["kind"] in ("swap", "gemm"):
        return (f"{c['kind']} BM{c['BM']} BN{c['BN']} BK{c['BK']} s{c['split']} m{c['mode']} "
                f"w{c['warps']} st{c['stages']}")
    return c["kind"] + (f" max_m{c['max_m']}" if "max_m" in c else "")


class _Sweep:
    def __init__(self, state):
        self.state = state
        self.results = {}  # (layer, m) -> {"sglang": us, "best": (us, cfg), "by_kind": {...}}
        self.logged = set()  # distinct failure messages already printed (the log keeps 16 KB)

    def note(self, key, msg):
        if key not in self.logged and len(self.logged) < 8:
            self.logged.add(key)
            log(msg)

    def shape(self, lab, layer, m, cands):
        n, k = lab_n(layer, m), FULL[layer][1]
        st = self.state
        x = lab_input(m, k, 1000 + m)
        w = st[layer][:n]
        rec = self.results.setdefault((layer, m), {"by_kind": {}, "fails": 0, "wrong": 0, "best": None})
        if "sglang" not in rec:
            rec["sglang"] = lab.time_graph(lambda: sglang_path(x, layer, n, st))
        for c in cands:
            if lab.left() < 8:
                log(f"{layer} m{m}: out of budget after {len(rec['by_kind'])} kinds")
                break
            try:
                errs = lab_errors(run_choice(c, x, layer, n, st), x, w)
            except Exception as exc:
                rec["fails"] += 1
                text = _error_text(exc, 300)
                self.note(("fail", c["kind"], text[-120:]), f"{layer} m{m} {cfg_name(c)}: FAILED {type(exc).__name__}: {text}")
                continue
            if not lab_ok(errs, layer):
                rec["wrong"] += 1
                self.note(("wrong", layer, c["kind"], c.get("mode")),
                          f"{layer} m{m} {cfg_name(c)}: WRONG rel {errs[0]:.2e} row {errs[1]:.2e} probe {errs[2]:.2e}")
                continue
            us = lab.time_graph(lambda: run_choice(c, x, layer, n, st))
            kind = c["kind"]
            if kind not in rec["by_kind"] or us < rec["by_kind"][kind][0]:
                rec["by_kind"][kind] = (us, c, errs)
            if rec["best"] is None or us < rec["best"][0]:
                rec["best"] = (us, c, errs)

    def commit(self, lab):
        """Re-time SGLang and each winner side by side, check the winner on outliers, fill CHOICE."""
        st = self.state
        for (layer, m), rec in sorted(self.results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            n, k = lab_n(layer, m), FULL[layer][1]
            line = f"{layer:9s} m{m:<4d} sglang {rec['sglang']:7.2f}"
            for kind, (us, c, _) in sorted(rec["by_kind"].items()):
                line += f" | {kind} {us:7.2f}"
            if rec["fails"] or rec["wrong"]:
                line += f" | fails {rec['fails']} wrong {rec['wrong']}"
            best = rec["best"]
            if best is None:
                log(line + " -> sglang (no correct candidate)")
                continue
            c = best[1]
            x = lab_input(m, k, 2000 + m)
            x_out = lab_input(m, k, 3000 + m, dist="outliers")
            ok = lab_ok(lab_errors(run_choice(c, x_out, layer, n, st), x_out, st[layer][:n]), layer)
            t_s = min(rec["sglang"], lab.time_graph(lambda: sglang_path(x, layer, n, st)))
            t_c = min(best[0], lab.time_graph(lambda: run_choice(c, x, layer, n, st)))
            gbs = (n * k * st[layer].element_size()) / (t_c * 1e-6) / 1e12
            line += f" -> best {cfg_name(c)} {t_c:.2f} us ({t_s / t_c:.3f}x, {gbs:.2f} TB/s of weights)"
            if ok and t_s / t_c >= MIN_GAIN:
                CHOICE[(layer, m)] = c
                log(line + " CHOSEN")
            else:
                log(line + (" (outliers WRONG)" if not ok else " (below MIN_GAIN)") + " -> sglang")


def sec_env(lab):
    import importlib.util
    log(f"torch {torch.__version__} triton {triton.__version__} gpu {torch.cuda.get_device_name()} "
        f"sms {torch.cuda.get_device_properties(0).multi_processor_count}")
    for mod in ("sglang.kernels.ops.gemm.tiny_gemm", "sglang.kernels.ops.gemm.hopper_bf16_gemv", "cutlass", "tilelang"):
        try:
            log(f"{mod}: {'found' if importlib.util.find_spec(mod) else 'missing'}")
        except Exception as exc:
            log(f"{mod}: {type(exc).__name__}")


def sec_triton(lab, sweep, m):
    for layer in LAYER_ORDER:
        n, k = lab_n(layer, m), FULL[layer][1]
        sweep.shape(lab, layer, m, triton_candidates(layer, m, n, k))


def sec_image(lab, sweep, m):
    for layer in LAYER_ORDER:
        if lab.left() < 40:  # one JIT compile can take tens of seconds
            log(f"image kernels m{m}: skipped from {layer}, {lab.left():.0f}s left")
            return
        n, k = lab_n(layer, m), FULL[layer][1]
        cands = image_candidates(layer, m, n, k)
        if cands:
            sweep.shape(lab, layer, m, cands)


def prepare(weights):
    state = _submission_prepare(weights)
    lab = Lab(LAB_BUDGET_S)
    sweep = _Sweep(state)
    lab.section("env", sec_env)
    for m in SCORED_M:
        lab.section(f"triton m{m}", sec_triton, sweep, m, min_left=20.0)
    for m in (1, 8, 16, 32):
        lab.section(f"image m{m}", sec_image, sweep, m, min_left=40.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    log(f"CHOICE has {len(CHOICE)} of {len(SCORED_M) * len(LAYER_ORDER)} scored shapes; "
        f"lab took {LAB_BUDGET_S - lab.left():.0f}s")
    return state

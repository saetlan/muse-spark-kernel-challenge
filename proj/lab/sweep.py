"""Sweep helpers shared by the lab sections from PE2 on (PE1's lab/e1.py keeps its own copy).

A Sweep times, per (layer, m), SGLang's path and the base submission's current CHOICE entry as
references, then each candidate that is correct against FP64 (half the judge's bounds, probe rows
included). commit() re-checks each winner on outlier inputs, re-times it next to the references
and puts it in CHOICE only if it beats the faster reference by MIN_GAIN, so the run's timed
kernel_fn uses it."""

import math

MIN_GAIN = 1.03
FULL = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
        "router": (256, 8192)}


def lab_n(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else FULL[layer][0]


def lab_input(m, k, seed, dist="normal", probe_all=False):
    """Like the judge's generate_input: standard normal, ~1 row in 8 a single power of two
    (every row with probe_all, so the probe check runs at any m)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda")
    if dist == "outliers":
        idx = torch.randperm(k, generator=gen, device="cuda")[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device="cuda") < 1 / 8
    if m >= 8:
        probes[torch.randint(m, (1,), generator=gen, device="cuda")] = True
    if probe_all:
        probes[:] = True
    cols = torch.randint(k, (m,), generator=gen, device="cuda")
    signs = torch.randint(2, (m,), generator=gen, device="cuda") * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device="cuda")
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16)


def lab_errors(out, x, w):
    """(relative L2, worst row relative to max(own norm, 0.1 x mean), worst probe row) vs FP64."""
    want = torch.float32 if w.dtype == torch.float32 else torch.bfloat16
    ref = (x.double() @ w.double().T).to(want).double()
    if not isinstance(out, torch.Tensor) or out.shape != ref.shape or out.dtype != want:
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
    return rel <= 0.5 * tol and row <= 2.5 * tol and pr <= 0.5e-3


def cfg_name(c):
    if c is None:
        return "sglang"
    if c["kind"] in ("swap", "gemm"):
        return (f"{c['kind']} BM{c['BM']} BN{c['BN']} BK{c['BK']} s{c['split']} m{c['mode']} "
                f"w{c['warps']} st{c['stages']}")
    if c["kind"] == "fma":
        return f"fma BN{c['BN']} s{c['split']} w{c['warps']}"
    if c["kind"] == "fi_mm":
        return f"fi_mm {c['backend']}"
    if c["kind"] == "dg":
        return "dg native" if c["native"] else "dg cublaslt"
    if c["kind"] == "p9":
        return f"p9 NT{c['NT']} RTW{c['RTW']} RG{c['RG']} KSW{c['KSW']} D{c['D']} CS{c['CS']}"
    if c["kind"] == "rg":
        return f"rg RTC{c['RTC']} NTW{c['NTW']} CS{c['CS']} XS{c['XS']} D{c['D']}"
    if c["kind"] == "sx" and c.get("W"):
        return f"sxw NB{c['NB']} WGS{c['WGS']} KSW{c['KSW']} NS{c['NS']} P{c['P']} CS{c['CS']}"
    if c["kind"] == "sx":
        return (f"sx NT8{c['NT8']} TG{c['TG']} RTW{c['RTW']} RGN{c['RGN']} KSW{c['KSW']} NS{c['NS']} D{c['D']} "
                f"XS{c['XS']} CS{c['CS']}")
    if c["kind"] == "wgf":
        return f"wgf KS{c['KS']} ST{c['ST']} TT{c['TT']} D{c['D']}"
    if c["kind"] == "wg":
        return f"wg KS{c['KS']} ST{c['ST']} TT{c['TT']}"
    if c["kind"] == "f12m":
        return (f"f12m NT{c['NT']} RTW{c['RTW']} RG{c['RG']} KSW{c['KSW']} D{c['D']}"
                + (f" CS{c['CS']}" if c.get("CS", 1) > 1 else "") + ("" if c["F12"] else " bf16"))
    if c["kind"] == "f12v":
        return f"f12v R{c['R']} RG{c['RG']} KSW{c['KSW']} D{c['D']}"
    if c["kind"] in ("fi_wsk", "fi_direct"):
        return f"{c['kind']} {c['tactic']}"
    return c["kind"] + (f" max_m{c['max_m']}" if "max_m" in c else "")


class Sweep:
    def __init__(self, state):
        self.state = state
        self.results = {}
        self.logged = set()

    def note(self, key, msg):
        if key not in self.logged and len(self.logged) < 8:
            self.logged.add(key)
            log(msg)

    def shape(self, lab, layer, m, cands):
        n, k = lab_n(layer, m), FULL[layer][1]
        st = self.state
        x = lab_input(m, k, 1000 + m)
        x_probe = lab_input(m, k, 1500 + m, probe_all=True)
        w = st[layer][:n]
        rec = self.results.get((layer, m))
        if rec is None:
            cur = CHOICE.get((layer, m))
            rec = {"best": None, "n": 0, "fails": 0, "wrong": 0, "current": cur,
                   "sglang": lab.time_graph(lambda: sglang_path(x, layer, n, st))}
            rec["cur_us"] = lab.time_graph(lambda: run_choice(cur, x, layer, n, st)) if cur else rec["sglang"]
            self.results[(layer, m)] = rec
        for c in cands:
            if lab.left() < 8:
                log(f"{layer} m{m}: out of budget after {rec['n']} candidates")
                break
            try:
                errs = lab_errors(run_choice(c, x, layer, n, st), x, w)
                errs_p = lab_errors(run_choice(c, x_probe, layer, n, st), x_probe, w)
                errs = (max(errs[0], errs_p[0]), max(errs[1], errs_p[1]), max(errs[2], errs_p[2]))
            except Exception as exc:
                rec["fails"] += 1
                text = _error_text(exc, 300)
                self.note(("fail", c["kind"], text[-120:]), f"{layer} m{m} {cfg_name(c)}: FAILED {type(exc).__name__}: {text}")
                continue
            if not lab_ok(errs, layer):
                rec["wrong"] += 1
                self.note(("wrong", layer, c["kind"]),
                          f"{layer} m{m} {cfg_name(c)}: WRONG rel {errs[0]:.2e} row {errs[1]:.2e} probe {errs[2]:.2e}")
                continue
            us = lab.time_graph(lambda: run_choice(c, x, layer, n, st))
            rec["n"] += 1
            rec.setdefault("all", []).append((us, c))
            if rec["best"] is None or us < rec["best"][0]:
                rec["best"] = (us, c)

    def commit(self, lab, top=None):
        st = self.state
        for (layer, m), rec in sorted(self.results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            n, k = lab_n(layer, m), FULL[layer][1]
            cur = rec["current"]
            line = (f"{layer:9s} m{m:<4d} sglang {rec['sglang']:6.2f} | v1 {cfg_name(cur)} {rec['cur_us']:6.2f}"
                    f" | {rec['n']} ok")
            if rec["fails"] or rec["wrong"]:
                line += f", {rec['fails']} failed, {rec['wrong']} wrong"
            if rec["best"] is None:
                log(line + " -> keep v1")
                continue
            c = rec["best"][1]
            x = lab_input(m, k, 2000 + m)
            x_out = lab_input(m, k, 3000 + m, dist="outliers")
            ok = lab_ok(lab_errors(run_choice(c, x_out, layer, n, st), x_out, st[layer][:n]), layer)
            t_s = min(rec["sglang"], lab.time_graph(lambda: sglang_path(x, layer, n, st)))
            t_v = t_s if cur is None else min(rec["cur_us"], lab.time_graph(lambda: run_choice(cur, x, layer, n, st)))
            t_c = min(rec["best"][0], lab.time_graph(lambda: run_choice(c, x, layer, n, st)))
            ref = min(t_s, t_v)
            gbs = (n * k * st[layer].element_size()) / (t_c * 1e-6) / 1e12
            line += f" -> best {cfg_name(c)} {t_c:.2f} us ({ref / t_c:.3f}x of the faster reference, {gbs:.2f} TB/s)"
            if ok and ref / t_c >= MIN_GAIN:
                CHOICE[(layer, m)] = c
                log(line + " CHOSEN")
            else:
                log(line + (" (outliers WRONG)" if not ok else " (below MIN_GAIN)") + " -> keep v1")

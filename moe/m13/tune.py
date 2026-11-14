"""Self-tuning prepare (appended after the M13 section, whose prepare becomes _m13_prepare).

The queue makes lab -> config -> scored-run cycles slow, so prepare() measures its own options on the
judge's GPU, on synthetic inputs only (nothing derived from any call's inputs is ever kept), and picks per
token class the fastest config that passes an FP32 check stricter than the judge's at every size of the
class (judge-like, skewed and outlier routings): M11's E3c configs and v1 are the incumbents; M13 (1-64
tokens) and M14 (33-320 tokens) with or without PDL and with an L2 prefetch distance are the candidates.
Any failure leaves the incumbents in place. A compact log (bandwidth recon, every timing, every
decision) is printed for the run's log tail. Budgeted so prepare() as a whole stays under ~250 s."""

import statistics
import time

TUNE_BUDGET_S = 170.0
TUNE_PREPARE_LIMIT_S = 255.0  # prepare() gets 300 s
TUNE_REPS = 20  # E8/E9 tuned in ~5 s of the 170 s budget: sample more
TUNE_ROUTINGS = 5  # E46: was 3; near-tie picks (MF41-MF44) need more routings
# E46: candidate families that never won a class in E21-E45 are skipped on the GPU (the dry run keeps them all)
TUNE_PRUNE = True
TUNE_MAX_ERR, TUNE_MAX_ROW_ERR = 1e-2, 5e-2  # the judge allows 1.5e-2 and 7.5e-2
TUNE_MARGIN = 0.95  # a candidate must be >= 5% faster than the incumbent (E12: +4-5% in-tuner was -2% judged)
TUNE_MARGIN_PDL = 0.98  # the incumbent's own kernels with a PDL tail (same kernels; E15: bs8 98.0 vs 101.0 just missed 3%)
TUNE_FLUSH_BYTES = 512 << 20
# class upper bound -> representative batch size
TUNE_CLASSES = ((5, 1), (9, 8), (17, 16), (33, 32), (65, 64), (192, 128), (321, 256))
# M13 candidates (E5's best stages, MF8) x L2 prefetch distance x PDL (GEMM2, post-norm)
TUNE_M13_S1 = (dict(kind="m13", nt=1, c=4),)
TUNE_M13_S2 = (dict(kind="m13", tt=2, nt=1, c=4),)  # E8: t1n1c5 lost to t2n1c4 in every class
# Start-up L2 prefetches (E10): npre 255 = the warp's own next groups (lane-parallel, at kernel start);
# nxq = GEMM1 prefetches the first nxq/4 of each GEMM2 warp's weight range (GEMM2 then starts from L2).
TUNE_M13_S1_PF = (dict(npre=0, nxq=0),)  # E12: every start-up prefetch was slower (MF12/MF13)
TUNE_M13_S2_PF = (dict(npre=0),)
TUNE_PFD = (0,)  # E8: L2 prefetch-ahead 2/4 was slower everywhere (MF10)
TUNE_PDL = ((0, 0), (1, 1))
# E18 hot/cold split (MF19): experts with > T pairs run as a second launch on a wide-pass kernel (decoded 2-4x
# instead of ~8x at 16 pairs per pass); the cold part keeps the stage search's best M14/M18 stage.
TUNE_SPLIT_T = (16, 32, 48)  # E26: + 48 (at bs256 T 16 sends about half the experts to the hot side, MF21)
# E20: M19 with 32-pair passes (4 or 3 CTAs per SM) as the hot kernel; E18/E19 never chose M18 g2.8 or p1.16
TUNE_HOT1 = (dict(kind="m19", nt=4, c=4, p=2, mb=4), dict(kind="m19", nt=4, c=3, p=2, mb=3), dict(kind="m14", nt=4, c=3, p=4))
TUNE_HOT2 = (dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4), dict(kind="m18", tt=2, nt=8, c=2, p=2),
             dict(kind="m14", tt=1, nt=4, c=4, p=4))
# E26 (T34): M11b as the cold side at 128/256 tokens (it beat M19 by 11% at bs32, where experts also carry 1-15
# tokens); cold GEMM1 / GEMM2 stages (8- or 16-pair passes), thresholds, and E25's best hot pairs (+ M19 at MB 4,
# which the hot launch never got before E26: its MB bit wasn't passed)
# E32: + the 16-pair shapes that fit 4 CTAs per SM (g2.2d2k4 128 r, p2.2d2k4 115 r; E26 offered only n2d4k2, 182-184 r,
# 2 CTAs), so cold experts with 9-16 pairs are decoded once; T 24 added
TUNE_COLD11B_S1 = (dict(kind="m11b", nt=1, d=4, ks=2, c=8), dict(kind="m11b", nt=2, d=4, ks=2, c=8),
                   dict(kind="m11b", nt=2, d=2, ks=4, c=8))
TUNE_COLD11B_S2 = (dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8), dict(kind="m11b", tt=2, nt=2, d=4, ks=2, c=8),
                   dict(kind="m11b", tt=2, nt=2, d=2, ks=4, c=8))
TUNE_COLD11B_T = (8, 16, 24, 32)
TUNE_HOT_PAIRS = ((TUNE_HOT1[1], TUNE_HOT2[2]), (TUNE_HOT1[1], TUNE_HOT2[1]), (TUNE_HOT1[0], TUNE_HOT2[0]))
# E27 (MF29): M11b at 5-6 CTAs per SM. Base = E26's picks per class (s1, s2); bs128's cold side with E26's hot pair,
# bs256 with M11b cold under E26's hot M19 pair. Variants replace s1, s2 or both with an occupancy instance of the same
# shape (M11B_OCC_INSTANCES), grid c = minb or 2 minb.
_B1 = (dict(kind="m11b", nt=1, d=4, ks=4, c=8), dict(kind="m11b", tt=1, nt=1, d=4, ks=4, c=4, pdl=1))
_B8 = (dict(kind="m11b", nt=1, d=4, ks=2, c=8), dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8, pdl=1))
TUNE_OCC_BASE = {1: _B1, 8: _B8, 16: _B8, 32: _B8,
                 128: (dict(_B8[0], thr=16, hot=dict(TUNE_HOT1[1])), dict(_B8[1], thr=16, hot=dict(TUNE_HOT2[1]))),
                 256: (dict(_B8[0], thr=16, hot=dict(TUNE_HOT1[1])), dict(_B8[1], thr=16, hot=dict(TUNE_HOT2[0])))}


def _early_variants(st, gated):
    """E30: st's M11b shape with the early ring refill (M11B_EARLY_INSTANCES), uncapped and at 128 registers, same c."""
    tt = 2 if gated else st["tt"]
    return [dict(st, early=1, occ=(mb if mb > 1 else 0)) for g, t, nt, d, ks, mb in M11B_EARLY_INSTANCES
            if (g, t, nt, d, ks) == (int(gated), tt, st["nt"], st["d"], st["ks"])]


def _occ_variants(st, gated):
    """st's M11b shape at 5-6 CTAs per SM (M11B_OCC_INSTANCES), each at grid c = minb and 2 minb."""
    tt = 2 if gated else st["tt"]
    out = []
    for g, t, nt, d, ks, mb in M11B_OCC_INSTANCES:
        if (g, t, nt, ks) == (int(gated), tt, st["nt"], st["ks"]):
            for c in (mb, 2 * mb):
                out.append(dict(st, d=d, occ=mb, c=c))
    return out


def tlog(msg):
    print(f"[tune] {msg}", flush=True)


class _Tuner:
    def __init__(self, budget):
        self.t0 = time.monotonic()
        self.budget = budget
        self.flush = None

    def left(self):
        return self.budget - (time.monotonic() - self.t0)

    def time_graph(self, fn, reps=TUNE_REPS):
        """Median us of a CUDA-graph replay of fn, each after an L2 flush (as the judge times)."""
        if self.flush is None:
            self.flush = torch.empty(TUNE_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            self.flush.sum()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)


def _routing(bs, seed):
    """The reference's default routing (top-8 of uniform + heavy-tailed popularity, busiest GPU = 0..31)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((bs, 256), generator=gen, device="cuda")
    z = torch.randn((256,), generator=gen, device="cuda")
    u = torch.rand((256,), generator=gen, device="cuda").clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    vals, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return ids.to(torch.int32).contiguous(), (vals / vals.sum(-1, keepdim=True)).float().contiguous()


def _case(bs, seed, skew=False, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if outliers:  # 1% of channels scaled 30x, like the judge's skewed test
        cols = torch.randperm(4096, generator=gen, device="cuda")[:41]
        x[:, cols] = (x[:, cols].float() * 30.0).to(torch.bfloat16)
    if skew:  # every token on experts 0-7
        ids = torch.arange(TOP_K, dtype=torch.int32, device="cuda").repeat(bs, 1).contiguous()
        w = torch.rand((bs, TOP_K), generator=gen, device="cuda")
        w = (w / w.sum(-1, keepdim=True)).contiguous()
    else:
        ids, w = _routing(bs, seed)
    return x, ids, w


class _Ref:
    """FP32 reference from the raw NVFP4 weights (as reference.py), experts dequantized on demand."""

    def __init__(self, weights):
        self.w = weights
        self.cache = {}

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (_tune_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                             _tune_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.cache[e]

    def full(self, x, ids, tw):
        flat = ids.reshape(-1).long()
        nw = 1.0 + self.w["post_norm_w"].float()
        out = torch.zeros((ids.shape[0], 4096), device="cuda")
        for e in [e for e in torch.unique(flat).tolist() if e < 32]:
            p = (flat == e).nonzero().flatten()
            h = x[p // TOP_K].float() @ self.expert(e)[0].T
            act = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]  # FP32, as reference.ref_kernel
            y = (act @ self.expert(e)[1].T) * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, p // TOP_K, y * tw.reshape(-1)[p].unsqueeze(1))
        return out


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _tune_dequant(packed, sf, gs):
    """NVFP4 -> FP32 [rows, K] (e2m1 * e4m3 block scale * global scale), as nvfp4.dequantize."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 0x7).long()]
    vals = torch.where((codes & 0x8) != 0, -mag, mag)
    rows, k = vals.shape
    return (vals.view(rows, k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)).view(rows, k)


def _errors(out, ref):
    out, ref = out.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    diff = out - ref
    err = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    return err, float((diff.norm(dim=-1) / torch.maximum(rows_ref, floor)).max())


def _clean(state):
    bank = state["m13"]
    dirty = [k for k in ("scr13", "scr2", "ctr13", "ctr2", "ctrf") if bool(bank[k].ne(0).any())]
    for k in dirty:
        bank[k].zero_()
    return dirty


def _run(state, cfg, x, ids, w):
    if cfg is None:
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                "config": {"bs": x.shape[0], "local_start": 0, "local_count": 32}}
        return _m11_kernel_fn(data, state)
    return m13_run(x, ids, w, state, cfg)


def _pruned(cfg, bs):
    """E46: True for a family that never won (E22-E45 logs): M11 CR/SR/G2P, M13, early refill, pre-wait prefetch,
    M11b at 5-6 CTAs per SM, M18 as the hot GEMM1, M11b GEMM2 t1 above 1 token, M11b GEMM1 K split 4 at 8-32 tokens
    (its base is k2), and M11b splits at <= 32 tokens."""
    s1, s2 = cfg["s1"], cfg["s2"]
    if s1.get("kind") == "m13" or s2.get("kind") in ("m13", "m11f") or s1.get("cr") or s1.get("sr") or s2.get("cr") or s2.get("sr"):
        return True
    if s1.get("early") or s2.get("early") or s2.get("pf") or s1.get("occ", 0) > 4 or s2.get("occ", 0) > 4:
        return True
    if s1.get("hot", {}).get("kind") == "m18":
        return True
    if bs > 1 and s2.get("kind") == "m11b" and s2.get("tt") == 1:
        return True
    if bs in (8, 16, 32) and s1.get("kind") == "m11b" and s1.get("ks") == 4:
        return True
    return bs <= 32 and bool(s1.get("thr"))


def _sname(st):
    if st.get("kind") in (None, "m11p", "m11f", "m11b"):  # M11's stage dicts (E3c; m11f = G2P, E23; m11b, E24)
        return (f"{st.get('kind', 'm11')}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}d{st['d']}k{st['ks']}"
                f"c{st['c']}{'R' if st.get('cr') else ''}{'S' if st.get('sr') else ''}{'o' + str(st['occ']) if st.get('occ') else ''}{'E' if st.get('early') else ''}{'F' if st.get('pf') else ''}{'U' if st.get('nu') else ''}{'Y' if st.get('dyn') else ''}{'H' if st.get('hf') else ''}"
                + (f"|>{st['thr']}:{_sname(st['hot'])}" if st.get("thr") else ""))  # E26: M11b as a split's cold side
    extra = f"b{st['beta']}" if st["kind"] in ("m15", "m17") else f"pf{st.get('pfd', 0)}"
    if st.get("p"):
        extra += f"P{st['p']}"
    if st.get("db"):
        extra += "D"
    if st.get("mb"):
        extra += f"B{st['mb']}"
    if st.get("npre"):
        extra += f"s{st['npre']}"
    if st.get("nxq"):
        extra += f"x{st['nxq']}"
    if st.get("lw"):
        extra += "W"
    if st.get("thr"):
        extra += f"|>{st['thr']}:{_sname(st['hot'])}"
    return f"{st['kind']}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}c{st['c']}{extra}"


def _name(cfg):
    if cfg is None:
        return "incumbent"
    s1, s2 = cfg["s1"], cfg["s2"]
    return f"{_sname(s1)}+{_sname(s2)} pdl{s2.get('pdl', 0)}{cfg['post']['pdl']}"


def _valid(state, ref, cfg, lo, hi):
    """Worst (error, row error) over the class's sizes and routings; None if scratch was left dirty."""
    sizes = sorted({b for b in (1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 191, 192, 256, 257, 320) if lo <= b < hi})
    cases = [(b, False, False) for b in sizes] + [(b, True, False) for b in sizes] + [(sizes[-1], True, True)]
    worst = (0.0, 0.0)
    for b, skew, outl in cases:
        x, ids, w = _case(b, 3100 + b + 50 * skew + 25 * outl, skew, outl)
        err, row = _errors(_run(state, cfg, x, ids, w), ref.full(x, ids, w))
        worst = (max(worst[0], err), max(worst[1], row))
    torch.cuda.synchronize()
    return None if _clean(state) else worst


def _bw_recon(tuner, state):
    """Bandwidth recon for the log: torch sum and a plain streaming kernel over contiguous expert codes; M13
    GEMM1 at about bs1 and bs8 and M14 GEMM1 (n2 c3 P4) at bs128 and bs256, each full and in the recon modes
    (loads only / decode without MMA / MMA without decode), so the log says what binds at each size."""
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    len16 = bank["w13"][0].numel() // 4
    parts = []
    for n in (3, 9):
        mb = n * len16 * 16 / 1e6
        t_torch = tuner.time_graph(lambda: bank["w13"][:n].view(torch.int64).sum())
        experts = torch.arange(n, dtype=torch.int32, device="cuda")
        best = max((mb / tuner.time_graph(lambda: ext.m13_stream(bank["w13"], experts, len16, sink, u, c * sms)), u, c)
                   for u in (8, 16) for c in (4, 8, 16))
        parts.append(f"{mb:.0f}MB torch {mb / t_torch:.2f} stream {best[0]:.2f} (U{best[1]} c{best[2]})")
    tlog("bw TB/s (incl. graph launch): " + "; ".join(parts))
    # E17: where M18's ~2,000 cycles per chunk go (MF17); M14 for reference, same routing. M18 modes: 1 = ring + stages
    # + barrier only, 2 = decode + store without wgmma, 3 = wgmma without decode, 4 = full with __syncwarp for the barrier.
    for bs in (128, 256):
        x, ids, w = _case(bs, 900 + bs)
        flat = ids.reshape(-1)
        ne = int((torch.bincount(flat.long(), minlength=256)[:32] > 0).sum())
        mb = ne * 18.87
        out = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")

        def guarded(name, fn):  # E17: one refused launch lost every number; now each one fails alone
            try:
                return fn()
            except Exception as exc:
                tlog(f"skeleton bs{bs} {name} FAILED {type(exc).__name__}: {str(exc)[-160:]}")
                _clean(state)
                return float("nan")

        def g1(st):
            return guarded(f"{st['kind']} P{st['p']} pfd{st.get('pfd', 0)}",
                           lambda: tuner.time_graph(lambda: m13_gemm1(x, flat, state, st)))
        t14 = g1(dict(kind="m14", nt=2, c=3, p=2))
        t14l = guarded("m14 loads", lambda: tuner.time_graph(lambda: ext.m14_recon(bank["w13"], bank["w13_s"], bank["c13"], flat, x, out, bank["scr13"],
                                                      bank["ctr13"], flat.numel(), 0, bank["rt_w13"], 0, bank["inter"], 1, 2, 2,
                                                      3 * sms, 4, 1)))
        t18 = g1(dict(kind="m18", nt=4, c=3, p=2))
        t18p4 = g1(dict(kind="m18", nt=4, c=3, p=4))
        t18f = [g1(dict(kind="m18", nt=4, c=3, p=2, pfd=k)) for k in (2, 4)]
        t18m = [guarded(f"m18 mode {m}", lambda m=m: tuner.time_graph(lambda: ext.m18_recon(
                    bank["w13r"], bank["w13r_s"], bank["c13"], flat, x, out, bank["scr13"], bank["ctr13"], flat.numel(), 0,
                    bank["rt_w13"], 3 * sms, m))) for m in (1, 2, 3, 4)]
        _clean(state)
        tlog(f"skeleton bs{bs} GEMM1 {mb:.0f}MB ({ne} exp), us: M14 n2c3 {t14:.0f} (loads {t14l:.0f}) | M18 g2.4c3 P2 {t18:.0f}, "
             f"P4 {t18p4:.0f}, L2-ahead 2/4 {t18f[0]:.0f}/{t18f[1]:.0f} | M18 modes: loads+stages+barrier {t18m[0]:.0f}, "
             f"decode+store {t18m[1]:.0f}, wgmma only {t18m[2]:.0f}, no CTA barrier {t18m[3]:.0f}")
    _clean(state)


def _m11b_recon(tuner, state):
    """E29 (MF31): where M11b's time goes at 1-128 tokens. Per batch size, the class's GEMM1 and GEMM2 shapes (bs128:
    the cold side, T 16) full and in recon modes 1-3 (loads only / decode without MMA / MMA without decode), in us
    and in TB/s of weight bytes read (codes + scales of the experts the launch covers), one routing each."""
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    for bs, (nt1, d1, k1, c1), (tt2, d2, k2, c2), thr in ((1, (1, 4, 4, 8), (1, 4, 4, 4), 0), (8, (1, 4, 2, 8), (2, 4, 4, 8), 0),
                                                         (32, (1, 4, 2, 8), (2, 4, 4, 8), 0), (128, (1, 4, 2, 8), (2, 4, 4, 8), 16)):
        try:
            x, ids, _ = _case(bs, 950 + bs)
            flat = ids.reshape(-1).contiguous()
            cnt = torch.bincount(flat.long(), minlength=256)[:32]
            ne = int(((cnt > 0) & (cnt <= thr if thr else cnt > 0)).sum())
            act = torch.randn((flat.numel(), 4096), device="cuda").to(torch.bfloat16)
            out1 = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")
            out2 = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")
            parts = []
            for name, mb, args in (
                    ("GEMM1", ne * 18.87, (bank["w13"], bank["w13_s"], bank["c13"], flat, x, out1, flat.numel(), 0,
                                           bank["rt_w13"], 0, bank["inter"], 1, 2, nt1, d1, k1, c1 * sms)),
                    ("GEMM2", ne * 9.44, (bank["w2"], bank["w2_s"], bank["c2"], flat, act, out2, flat.numel(), 0,
                                          bank["rt_w2"], 1, bank["hidden"], 0, tt2, 1, d2, k2, c2 * sms))):
                ts = [tuner.time_graph(lambda: ext.m11b_gemm(*args, 0, thr))]
                ts += [tuner.time_graph(lambda m=m: ext.m11b_recon(*args, thr, m)) for m in (1, 2, 3)]
                parts.append(f"{name} {mb:.0f}MB: full {ts[0]:.1f} loads {ts[1]:.1f} no-MMA {ts[2]:.1f} no-decode {ts[3]:.1f} us"
                             f" ({mb / ts[0]:.2f}/{mb / ts[1]:.2f} TB/s)")
            tlog(f"m11b recon bs{bs}{' cold<=16' if thr else ''} ({ne} exp): " + " | ".join(parts))
        except Exception as exc:
            tlog(f"m11b recon bs{bs} FAILED {type(exc).__name__}: {str(exc)[-160:]}")
            _clean(state)


def _m14_stage_search(tuner, state, bss):
    """Best large-batch GEMM1 and GEMM2 stage per batch size (cold-L2 graph time, 2 routings), over the
    compiled M14 and M15 instances x CTAs per SM up to the measured occupancy (M14 with L2 prefetch 2 groups
    ahead, E6/MF9; M15 with cost weight 0 or 4)."""
    variants = []
    for kind in ("m14", "m19", "m18", "m15"):  # M14 first (proven); M16/M17 left the search in E16 (2x slower, MF14/MF15)
        res = state.get(kind + "_resources", [])
        tlog(f"{kind} attrs: " + ", ".join(f"{'g' if g else 'p'}{tt}.{nt}P{pr} "
                                          f"{regs}r{f' {local}B' if local else ''} {(smem + dyn) // 1024}K {blocks}cta"
                                          for g, tt, nt, pr, regs, local, smem, dyn, blocks in res))
        for g, tt, nt, pr, regs, local, smem, dyn, blocks in res:
            pa, db = pr & 15, pr >> 4  # M16/M17 rows carry P | DB << 4
            for c in range(1, blocks + 1):
                for extra in (tuple(dict(beta=b, p=pa, db=db) for b in (0, 4)) if kind in ("m15", "m17")
                              else (dict(pfd=0, p=pa, mb=pr >> 4),) if kind == "m19"  # E19: column 3 = P | MB << 4
                              else (dict(pfd=0, p=pa, db=db),)):
                    st = dict(kind=kind, nt=nt, c=c, **extra)
                    if not g:
                        st["tt"] = tt
                    variants.append((g, st))
    best = {1: {}, 2: {}}
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)] for bs in bss}
    acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in bss}
    rows = []
    for g, st in variants:
        if tuner.left() < 60:
            tlog(f"stage search out of budget after {len(rows)} of {len(variants)} variants")
            break
        stage = 1 if g else 2
        line = {}
        for bs in bss:
            if (st["kind"] == "m14" and bs < 64) or (st["kind"] == "m19" and bs < 16):  # E6: M14 lost below 64 tokens; E20 tries M19 (4 CTAs) from 16
                continue
            if st["kind"] in ("m15", "m17") and st["nt"] <= 2 and bs > 32:  # their small-batch instances
                continue
            ts = []
            for x, ids, w in cases[bs]:
                flat = ids.reshape(-1)
                fn = (lambda x=x, f=flat: m13_gemm1(x, f, state, st)) if g else \
                     (lambda a=acts[bs], f=flat: m13_gemm2(a, f, state, st))
                ts.append(tuner.time_graph(fn))
            t = sum(ts) / len(ts)
            line[bs] = t
            best[stage].setdefault(bs, []).append((t, st))
        _clean(state)
        rows.append((stage, st, line))
    # the log keeps variants within 1.25x of their stage's best at some size (the output tail is 16,000 chars)
    top = {(stage, bs): min(t for t, _ in v) for stage in (1, 2) for bs, v in best[stage].items()}
    shown = 0
    for stage, st, line in rows:  # M16/M17 within 1.5x of the best somewhere (E13 logged them all), others 1.25x
        lim = 1.5 if st["kind"] in ("m16", "m17") else 1.25
        if any(t <= lim * top[(stage, bs)] for bs, t in line.items()):
            shown += 1
            tlog(f"s{stage} {_sname(st)} | " + " ".join(f"{bs}:{t:.0f}" for bs, t in line.items()))
    tlog(f"stage search: {len(rows)} variants timed, {shown} shown")
    for stage in (1, 2):  # keep the 2 fastest per batch size, so one wrong stage can't block the others
        for bs in best[stage]:
            best[stage][bs] = sorted(best[stage][bs], key=lambda v: v[0])[:2]
    return best


def tune(state, weights, deadline):
    tuner = _Tuner(min(TUNE_BUDGET_S, deadline - time.monotonic()))
    ref = _Ref(weights)
    tlog(f"start, budget {tuner.budget:.0f}s; incumbents: M11 E3c up to 32 tokens, v1 above")
    try:
        _bw_recon(tuner, state)
    except Exception as exc:
        tlog(f"bw recon FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    try:
        rows = []  # E22: registers / spill / static smem of the small-batch M11 kernels (incumbent, CR, SR)
        for bit, insts in ((0, M13_M11_INSTANCES), (16, M13_M11_CR_INSTANCES), (32, M13_M11_SR_INSTANCES)):
            for g, tt, nt, d, ks in insts:
                r, loc, shm = state["m11_ext"].m11_attrs(g, tt, nt, d | bit, ks).tolist()
                rows.append(f"{'gpc'[0 if g else 1]}{'' if not bit else 'R' if bit == 16 else 'S'}{tt}.{nt}d{d}k{ks} {r}r"
                            f"{f' {loc}B' if loc else ''} {shm // 1024}K")
        tlog("m11 attrs: " + ", ".join(rows))
    except Exception as exc:
        tlog(f"m11 attrs FAILED {type(exc).__name__}: {str(exc)[-160:]}")
    try:  # E27: registers / spill / static smem / CTAs per SM of every M11b instance (occupancy ones: minb)
        rows = []
        for g, tt, nt, d, ks, mb, e in _m11b_all():  # E30: + early-refill builds (label E)
            r, loc, shm, blk = state["m11_ext"].m11b_attrs(g, tt, nt, d | (mb << 4 if mb > 1 else 0) | (256 if e else 0), ks).tolist()
            rows.append(f"{'gp'[0 if g else 1]}{tt}.{nt}d{d}k{ks}{f'o{mb}' if mb > 1 else ''}{'E' if e else ''} {r}r{f' {loc}B' if loc else ''} "
                        f"{shm // 1024}K {blk}c")
        tlog("m11b attrs: " + ", ".join(rows))
    except Exception as exc:
        tlog(f"m11b attrs FAILED {type(exc).__name__}: {str(exc)[-160:]}")
    _m11b_recon(tuner, state)  # E29
    m14 = {1: {}, 2: {}}
    try:
        m14 = _m14_stage_search(tuner, state, (1, 8, 16, 32, 64, 128, 256))
    except Exception as exc:
        tlog(f"m14 search FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    configs = dict(state["m13_configs"])
    lo = 1
    for hi, bs in TUNE_CLASSES:
        if tuner.left() < 15:
            tlog("out of budget; remaining classes keep their incumbents")
            break
        cands = []
        if bs <= M13_MAX_TOKENS:
            for s1 in TUNE_M13_S1:
                for s2 in TUNE_M13_S2:
                    for pfd in TUNE_PFD:
                        for p1 in TUNE_M13_S1_PF:
                            for p2 in TUNE_M13_S2_PF:
                                for pdl, ppdl in TUNE_PDL:
                                    cands.append(dict(s1=dict(s1, pfd=pfd, **p1),
                                                      s2=dict(s2, pfd=pfd, pdl=pdl, pf=0, **p2),
                                                      post=dict(kind="m13", pdl=ppdl)))
        if bs in m14[1] and bs in m14[2]:  # M15 at every size, M14 from 64 tokens
            for _, s1b in m14[1][bs]:
                for _, s2b in m14[2][bs]:
                    for pfd in (TUNE_PFD if "m14" in (s1b["kind"], s2b["kind"]) else (0,)):
                        for pdl, ppdl in TUNE_PDL:
                            s1c = dict(s1b, pfd=pfd) if s1b["kind"] == "m14" else dict(s1b)
                            s2c = dict(s2b, pfd=pfd) if s2b["kind"] == "m14" else dict(s2b)
                            cands.append(dict(s1=s1c, s2=dict(s2c, pdl=pdl, pf=0), post=dict(kind="m13", pdl=ppdl)))
        if bs >= 128 and bs in m14[1] and bs in m14[2]:  # E18: hot experts in a second launch (MF19)
            c1 = [st for _, st in m14[1][bs] if st["kind"] in ("m14", "m18", "m19")]
            c2 = [st for _, st in m14[2][bs] if st["kind"] in ("m14", "m18", "m19")]
            if c1 and c2:
                for thr in TUNE_SPLIT_T:
                    for h1 in TUNE_HOT1:
                        for h2 in TUNE_HOT2:
                            cands.append(dict(s1=dict(c1[0], thr=thr, hot=dict(h1)),
                                              s2=dict(c2[0], pdl=1, pf=0, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
        if bs >= 128:  # E26 (T34): M11b for the cold experts (<= T pairs), M19/M14/M18 for the hot ones
            c2 = [st for _, st in m14[2].get(bs, ()) if st["kind"] in ("m14", "m18", "m19")][:1]
            for thr in TUNE_COLD11B_T:
                for h1, h2 in TUNE_HOT_PAIRS:
                    for i1, s1x in enumerate(TUNE_COLD11B_S1):
                        s1c = dict(s1x, thr=thr, hot=dict(h1))
                        for i2, s2x in enumerate(TUNE_COLD11B_S2):
                            cands.append(dict(s1=dict(s1c), s2=dict(s2x, pdl=1, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
                        if c2:  # cold GEMM2 on the stage search's best M19/M14/M18 stage (E25's choice)
                            cands.append(dict(s1=dict(s1c), s2=dict(c2[0], pdl=1, pf=0, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
        if bs >= 128:  # E35 (T35b): every split whose hot GEMM1 is M19 also with the late PDL wait
            cands += [dict(c, s1=dict(c["s1"], hot=dict(c["s1"]["hot"], lw=1))) for c in list(cands)
                      if c["s1"].get("thr") and c["s1"]["hot"].get("kind") == "m19"]
            # E36: and GEMM2's hot M19 launch waiting at its end (cold M11b/M19 with PDL signals after its own wait)
            cands += [dict(c, s2=dict(c["s2"], hot=dict(c["s2"]["hot"], lw=1))) for c in list(cands)
                      if c["s2"].get("thr") and c["s2"]["hot"].get("kind") in ("m19", "m18") and c["s2"].get("pdl")
                      and c["s2"]["kind"] in ("m11b", "m19") and c["s1"].get("hot", {}).get("lw")]
        if bs in (128, 256):  # E39: hot GEMM1 on M18 (64- or 32-pair passes) with the late wait, around E37's picks
            m19h1 = dict(kind="m19", nt=4, c=3, p=2, mb=3, lw=1)
            if bs == 128:
                c1 = dict(kind="m11b", nt=1, d=4, ks=2, c=8)
                c2 = dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8, pdl=1)
                h2s = (dict(kind="m18", tt=2, nt=8, c=2, p=2, lw=1), dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4, lw=1))
            else:
                c1 = dict(kind="m19", nt=2, c=4, pfd=0, p=2, mb=4)
                c2 = dict(kind="m19", tt=2, nt=2, c=4, pfd=0, p=4, mb=4, pdl=1, pf=0)
                h2s = (dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4, lw=1), dict(kind="m18", tt=2, nt=8, c=2, p=2, lw=1))
            for h1 in (dict(kind="m18", nt=8, c=2, p=2, lw=1), dict(kind="m18", nt=4, c=3, p=2, lw=1),
                       dict(kind="m18", nt=4, c=2, p=4, lw=1), dict(kind="m19", nt=4, c=4, p=2, mb=4, lw=1), m19h1):
                for h2 in h2s:
                    for thr in (16, 24, 32):
                        cands.append(dict(s1=dict(c1, thr=thr, hot=dict(h1)), s2=dict(c2, thr=thr, hot=dict(h2)),
                                          post=dict(kind="m13", pdl=1)))
        inc11 = m11_config_for(state, bs)
        if inc11 is not None:  # the M11 incumbent with a PDL tail: GEMM2 + post-norm, or the post-norm only (E15)
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"], kind="m11p"), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"]), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
        if inc11 is not None:  # E21: M11 with its ring streaming across units (MF23), at the incumbent's c and at 4
            for c1 in sorted({inc11["s1"]["c"], 4}):
                for c2 in sorted({inc11["s2"]["c"], 4}):
                    s1x, s2x = dict(inc11["s1"], cr=1, c=c1), dict(inc11["s2"], cr=1, c=c2)
                    cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(s1x), s2=dict(s2x), post=dict(kind="m13", pdl=1)))
        if inc11 is not None:  # E24: M11b (M11 on the M13 bank and decode) for GEMM1, GEMM2 or both
            s1i, s2i = inc11["s1"], inc11["s2"]
            b1 = (1, 2, s1i["nt"], s1i["d"], s1i["ks"]) in M11B_INSTANCES
            b2 = (0, s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"]) in M11B_INSTANCES
            if b1:
                s1b = dict(s1i, kind="m11b")
                for s2x in ([dict(s2i, kind="m11b", pdl=1), dict(s2i, kind="m11b", pdl=0)] if b2 else []) + \
                           [dict(s2i, kind="m11p"), dict(s2i)]:
                    cands.append(dict(s1=dict(s1b), s2=s2x, post=dict(kind="m13", pdl=1)))
                for p in (1, 0):  # with G2P for GEMM2 (E23)
                    if (s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"], p) in M11F_INSTANCES:
                        cands.append(dict(s1=dict(s1b), s2=dict(s2i, kind="m11f", pdl=p), post=dict(kind="m11f", pdl=0)))
            if b2:
                cands.append(dict(s1=dict(s1i), s2=dict(s2i, kind="m11b", pdl=1), post=dict(kind="m13", pdl=1)))
        if bs in TUNE_OCC_BASE:  # E27: M11b at 5-6 CTAs per SM (MF29), GEMM1, GEMM2 or both
            b1, b2 = TUNE_OCC_BASE[bs]
            if bs in (128, 256):  # E47: hot GEMM1 first (static M19), the dynamic M11b cold GEMM1 second with a late wait
                hf1 = dict(kind="m11b", nt=1, d=4, ks=2, c=4, dyn=1, hf=1, hot=dict(kind="m19", nt=4, c=3, p=2, mb=3))
                if bs == 128:
                    g2s = (dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=4, dyn=1, pdl=1, hot=dict(kind="m18", tt=2, nt=8, c=2, p=2, lw=1)),
                           dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8, pdl=1, hot=dict(kind="m18", tt=2, nt=8, c=2, p=2, lw=1)))
                else:
                    g2s = (dict(kind="m19", tt=2, nt=2, c=4, pfd=0, p=4, mb=4, pdl=1, pf=0,
                                hot=dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4, lw=1)),)
                for t in (12, 16, 20):
                    for g2 in g2s:
                        cands.append(dict(s1=dict(hf1, thr=t), s2=dict(g2, thr=t), post=dict(kind="m13", pdl=1)))
                # E49: around E48's picks: T 8/10/12, hot GEMM1 M19 at 3 or 4 CTAs per SM, GEMM2 hot M18
                for t in (4, 5, 6, 7, 8):  # E50: T 8 won at both sizes in E49 (the lowest tried): go lower
                    for h1 in (dict(kind="m19", nt=4, c=3, p=2, mb=3),):
                        for h2 in (dict(kind="m18", tt=2, nt=8, c=2, p=2), dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4)):
                            g2 = dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=4, dyn=1, pdl=1, hf=1, thr=t, hot=h2)
                            cands.append(dict(s1=dict(hf1, thr=t, hot=dict(h1)), s2=g2, post=dict(kind="m13", pdl=1)))
                # E48: GEMM2 hot first too (M18/M19 hot waits then signals, dyn M11b cold second with a late wait)
                for t in (12, 16):
                    for h2 in (dict(kind="m18", tt=2, nt=8, c=2, p=2), dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4)):
                        g2 = dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=4, dyn=1, pdl=1, hf=1, thr=t, hot=h2)
                        cands.append(dict(s1=dict(hf1, thr=t), s2=g2, post=dict(kind="m13", pdl=1)))
            if bs in (128, 256):  # E45: finer split thresholds around E43's picks (T 16 won among 8/16/24/32/48)
                if bs == 128:
                    e1 = dict(kind="m11b", nt=1, d=4, ks=2, c=4, dyn=1)
                    e2 = dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8, pdl=1)
                    eh1, eh2 = dict(kind="m19", nt=4, c=3, p=2, mb=3, lw=1), dict(kind="m18", tt=2, nt=8, c=2, p=2, lw=1)
                else:
                    e1 = dict(kind="m19", nt=2, c=4, pfd=0, p=2, mb=4)
                    e2 = dict(kind="m19", tt=2, nt=2, c=4, pfd=0, p=4, mb=4, pdl=1, pf=0)
                    eh1, eh2 = dict(kind="m19", nt=4, c=3, p=2, mb=3, lw=1), dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4, lw=1)
                for t in (10, 12, 14, 16, 18, 20):
                    cands.append(dict(s1=dict(e1, thr=t, hot=dict(eh1)), s2=dict(e2, thr=t, hot=dict(eh2)), post=dict(kind="m13", pdl=1)))
            if bs in (16, 32):  # E44: split again, light experts on the dynamic deal, busy ones (> 8 pairs) n2, both W
                for h1, h2 in ((dict(kind="m11b", nt=2, d=4, ks=2, c=8, lw=1), dict(kind="m11b", tt=2, nt=2, d=4, ks=2, c=8, lw=1)),
                               (dict(kind="m11b", nt=2, d=2, ks=4, c=8, lw=1), dict(kind="m11b", tt=2, nt=2, d=2, ks=4, c=8, lw=1)),
                               (dict(kind="m11b", nt=2, d=2, ks=4, c=4, lw=1), dict(kind="m11b", tt=2, nt=2, d=2, ks=4, c=4, lw=1))):
                    cands.append(dict(s1=dict(b1, dyn=1, c=4, thr=8, hot=dict(h1)),
                                      s2=dict(b2, dyn=1, c=4, thr=8, hot=dict(h2)), post=dict(kind="m13", pdl=1)))
            if bs == 256:  # E43: M11b cold with the dynamic deal at 256 tokens (static M11b cold lost here, MF29), hot W
                b1d = dict(b1, dyn=1, c=4, hot=dict(b1["hot"], lw=1))
                for t in (16, 24):
                    for b2d in (dict(b2, dyn=1, c=4, thr=t, hot=dict(b2["hot"], lw=1)), dict(b2, dyn=1, c=4, thr=t)):
                        cands.append(dict(s1=dict(b1d, thr=t), s2=b2d, post=dict(kind="m13", pdl=1)))
            if bs <= 128:  # E38: finer grid sizes for M11b (the static task deal depends on them, MF23)
                hw = dict(b1.get("hot", {}), lw=1) if b1.get("thr") else None  # bs128: E36's hot W pick
                b1w = dict(b1, hot=hw) if hw else dict(b1)
                for c in (5, 6, 7, 9, 10, 12, 14, 16, 20, 24):  # E39: + 14/20/24 (E38: bs32 best at c16, the largest tried)
                    cands.append(dict(s1=dict(b1w, c=c), s2=dict(b2), post=dict(kind="m13", pdl=1)))
                for c in (3, 5, 6, 10, 12, 16):
                    cands.append(dict(s1=dict(b1w), s2=dict(b2, c=c), post=dict(kind="m13", pdl=1)))
                for c in (4, 8):  # E42: dynamic task deal (atomic counter), persistent (c 4) or two waves (c 8)
                    cands.append(dict(s1=dict(b1w, dyn=1, c=c), s2=dict(b2), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(b1w), s2=dict(b2, dyn=1, c=c), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(b1w, dyn=1, c=c), s2=dict(b2, dyn=1, c=c), post=dict(kind="m13", pdl=1)))
                    if b2.get("thr") and b2["hot"].get("kind") in ("m18", "m19"):  # E43: + the hot GEMM2 late wait (cold lad)
                        b2w = dict(b2, hot=dict(b2["hot"], lw=1))
                        cands.append(dict(s1=dict(b1w, dyn=1, c=c), s2=dict(b2w, dyn=1, c=c), post=dict(kind="m13", pdl=1)))
                        cands.append(dict(s1=dict(b1w, dyn=1, c=c), s2=dict(b2w), post=dict(kind="m13", pdl=1)))
                if b2["tt"] == 2:  # E41: GEMM2 with one 16-row tile per task (twice the tasks, finer deal), bs8-128
                    for c in (4, 8, 12, 16):
                        cands.append(dict(s1=dict(b1w), s2=dict(b2, tt=1, c=c), post=dict(kind="m13", pdl=1)))
                if b1["ks"] == 2 and not b1.get("thr"):  # E40: GEMM1 with K split over 4 warps at larger grids (bs8-32)
                    for c in (8, 12, 16, 20, 24):
                        cands.append(dict(s1=dict(b1, ks=4, c=c), s2=dict(b2), post=dict(kind="m13", pdl=1)))
            v1, v2 = _occ_variants(b1, True), _occ_variants(b2, False)
            post = dict(kind="m13", pdl=1)
            cands.append(dict(s1=dict(b1), s2=dict(b2), post=dict(post)))  # the base itself (E26's pick)
            cands += [dict(s1=x, s2=dict(b2), post=dict(post)) for x in v1]
            cands += [dict(s1=dict(b1), s2=y, post=dict(post)) for y in v2]
            cands += [dict(s1=x, s2=y, post=dict(post)) for x in v1 for y in v2 if x["occ"] == y["occ"]]
            n1 = (1, 2, b1["nt"], b1["d"], b1["ks"]) in M11B_NU_INSTANCES  # E34: outer K loop rolled (MF36)
            n2 = (0, b2["tt"], b2["nt"], b2["d"], b2["ks"]) in M11B_NU_INSTANCES
            for x, y in ((n1, 0), (0, n2), (n1, n2)):
                if x or y:
                    cands.append(dict(s1=dict(b1, nu=1) if x else dict(b1), s2=dict(b2, nu=1) if y else dict(b2),
                                      post=dict(post)))
            if b2.get("pdl") and (0, b2["tt"], b2["nt"], b2["d"], b2["ks"]) in M11B_PF_INSTANCES:  # E33: L2 prefetch
                for x in [dict(b1)] + [v for v in v1 if v["occ"] == 4]:
                    cands.append(dict(s1=x, s2=dict(b2, pf=1), post=dict(post)))
            e1, e2 = _early_variants(b1, True), _early_variants(b2, False)  # E30 (MF32): refill before decode
            cands += [dict(s1=x, s2=dict(b2), post=dict(post)) for x in e1]
            cands += [dict(s1=dict(b1), s2=y, post=dict(post)) for y in e2]
            cands += [dict(s1=x, s2=y, post=dict(post)) for x in e1 for y in e2]
        if bs in (16, 32):  # E26 (T34b): experts above 8 pairs take one 16-pair pass instead of two 8-pair passes
            for s1x, s2x in ((dict(nt=1, d=4, ks=2, c=8), dict(tt=2, nt=1, d=4, ks=4, c=8)),
                             (dict(nt=1, d=4, ks=4, c=8), dict(tt=1, nt=1, d=4, ks=4, c=4))):
                for h1, h2 in ((dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8)),
                               (dict(nt=2, d=2, ks=4, c=8), dict(tt=2, nt=2, d=2, ks=4, c=8)),
                               (dict(nt=2, d=4, ks=2, c=4), dict(tt=2, nt=2, d=4, ks=2, c=4))):
                    for l1, l2 in ((0, 0), (1, 0), (1, 1)):  # E36: hot GEMM1 / GEMM2 waiting at their end
                        cands.append(dict(s1=dict(s1x, kind="m11b", thr=8, hot=dict(h1, kind="m11b", lw=l1)),
                                          s2=dict(s2x, kind="m11b", pdl=1, thr=8, hot=dict(h2, kind="m11b", lw=l2)),
                                          post=dict(kind="m13", pdl=1)))
        if inc11 is not None and bs == 32:  # E25: M11b at 17-32 tokens (M19 holds the class since E20)
            for s1x, s2x in ((dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8)),
                             (dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=4)),
                             (dict(nt=2, d=2, ks=4, c=8), dict(tt=2, nt=2, d=2, ks=4, c=8)),
                             (dict(nt=1, d=4, ks=2, c=8), dict(tt=2, nt=1, d=4, ks=4, c=8))):
                for p in (1, 0):
                    cands.append(dict(s1=dict(s1x, kind="m11b"), s2=dict(s2x, kind="m11b", pdl=p), post=dict(kind="m13", pdl=1)))
        if inc11 is not None:  # E23: G2P = M11 GEMM2 + post-norm in one kernel (prefill's shared/glue, M11F_INSTANCES)
            s2i = inc11["s2"]
            for p in (1, 0):
                if (s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"], p) in M11F_INSTANCES:
                    cands.append(dict(s1=dict(inc11["s1"]), s2=dict(s2i, kind="m11f", pdl=p), post=dict(kind="m11f", pdl=0)))
        if inc11 is not None:  # E22: M11 with an 8-deep shared-memory weight ring (MF24), c 4/5/8
            for c1 in sorted({inc11["s1"]["c"], 4, 5}):
                for c2 in sorted({inc11["s2"]["c"], 4}):
                    s1x, s2x = dict(inc11["s1"], sr=1, c=c1), dict(inc11["s2"], sr=1, c=c2)
                    cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(s1x), s2=dict(s2x), post=dict(kind="m13", pdl=1)))
        if inc11 is not None and bs == 32:  # decode 32 (E17): M11 configs with 4 K chunks in flight per warp (E3c: d2)
            for s1x, s2x in ((dict(nt=1, d=4, ks=2, c=8), dict(inc11["s2"])),
                             (dict(nt=2, d=4, ks=2, c=8), dict(inc11["s2"])),
                             (dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8))):
                cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1), margin=0.97))
        if TUNE_PRUNE:  # E46
            n0 = len(cands)
            cands = [c for c in cands if not _pruned(c, bs)]
            tlog(f"class [{lo},{hi}) bs{bs}: {len(cands)} of {n0} candidates after pruning")
        runs = [_case(bs, 1700 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)]
        try:
            t_inc = sum(tuner.time_graph(lambda: _run(state, None, x, ids, w)) for x, ids, w in runs) / TUNE_ROUTINGS
            timed = []
            nfail = 0
            for cfg in cands:
                if tuner.left() < 15:
                    break
                try:  # E22b: a refused or faulty candidate is skipped, not the whole class (E22 lost every small class)
                    timed.append((sum(tuner.time_graph(lambda: _run(state, cfg, x, ids, w)) for x, ids, w in runs)
                                  / TUNE_ROUTINGS, cfg))
                except Exception as exc:
                    if nfail == 0:
                        tlog(f"class [{lo},{hi}) candidate {_name(cfg)} FAILED {type(exc).__name__}: {str(exc)[-140:]}")
                    nfail += 1
                _clean(state)
            if nfail:
                tlog(f"class [{lo},{hi}) {nfail} candidate(s) failed and were skipped")
            pick, why = None, "no candidate faster"
            for t, cfg in sorted(timed, key=lambda v: v[0]):
                if t >= cfg.get("margin", TUNE_MARGIN) * t_inc:
                    continue
                worst = _valid(state, ref, cfg, lo, hi)
                if worst is not None and worst[0] < TUNE_MAX_ERR and worst[1] < TUNE_MAX_ROW_ERR:
                    pick, why = (t, cfg, worst), "ok"
                    break
                tlog(f"class [{lo},{hi}) rejected {_name(cfg)} {t:.1f}us: "
                     + ("dirty scratch" if worst is None else f"err {worst[0]:.1e} row {worst[1]:.1e}"))
            grid = [(c["s1"].get("npre", 0), c["s1"].get("nxq", 0), c["s2"].get("npre", 0), c["s2"].get("pdl", 0), t)
                    for t, c in timed if c["s1"].get("kind") == "m13"]
            if grid:  # every start-up prefetch combination: s1 npre/nxq, then s2 npre 0/255 x pdl 0/1
                cells = {}
                for n1, q, n2, pdl, t in grid:
                    cells.setdefault((n1, q), {})[(n2, pdl)] = t
                tlog(f"class [{lo},{hi}) m13 s1(npre,nxq): s2 npre0 pdl0/1, npre255 pdl0/1 | " + " ".join(
                    f"{n1},{q}:" + "/".join(f"{v[k]:.1f}" if k in v else "-" for k in ((0, 0), (0, 1), (255, 0), (255, 1)))
                    for (n1, q), v in sorted(cells.items())))
            top = " ".join(f"{_name(c)}={t:.1f}" for t, c in sorted(timed, key=lambda v: v[0])[:4])
            if pick:
                configs[hi] = pick[1]
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us -> {_name(pick[1])} {pick[0]:.1f}us "
                     f"({t_inc / pick[0]:.2f}x, err {pick[2][0]:.1e}) | best: {top}")
            else:
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us kept ({why}) | best: {top}")
        except Exception as exc:
            tlog(f"class [{lo},{hi}) FAILED {type(exc).__name__}: {str(exc)[-200:]}; incumbent kept")
            _clean(state)
        lo = hi
    state["m13_configs"] = tuple(sorted(configs.items()))
    ref.cache.clear()
    tuner.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    tlog(f"done in {time.monotonic() - tuner.t0:.0f}s; M13_CONFIGS = {state['m13_configs']!r}"[:1500])


def prepare(weights):
    t0 = time.monotonic()
    state = _m13_prepare(weights)
    tlog(f"base prepare (v1, nvcc, banks, warm-up) took {time.monotonic() - t0:.1f}s")
    if "m13" in state:
        try:
            tune(state, weights, t0 + TUNE_PREPARE_LIMIT_S)
        except Exception as exc:  # never fail prepare: the incumbents stay
            tlog(f"tune FAILED {type(exc).__name__}: {str(exc)[-300:]}")
            state["m13_configs"] = M13_CONFIGS
        for _, cfg in state["m13_configs"]:
            if cfg is not None:
                m13_warm(state, cfg)
        # (E14's verbose ptxas probe build is gone: its output filled the 16,000-char log tail; result in MF15)
        tlog(f"prepare done in {time.monotonic() - t0:.0f}s")
    return state

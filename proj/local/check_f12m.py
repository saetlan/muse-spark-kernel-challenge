"""Exactness of F12M (12-bit weights in mma.sync A-fragment order, m = 2..32), checked on the CUDA text:
f12m_decode, the scale helpers and every index helper are parsed out of proj/f12m/f12m.py and executed
(check_f12v's translator). The emulation follows the PTX ISA 9.4 m16n8k16 fragment layouts (sec.
9.7.16.5.8): A register r of lane (g, t) = A[g + 8 (r & 1)][2 t + 8 (r >> 1) + p] (half p), B b0/b1 =
B[2 t + p (+ 8)][g], C c[r] = C[g + 8 (r >> 1)][2 t + (r & 1)]. Checks:
  1. decode: every (sign, code, mantissa) at every element gives BF16 s << 15 | (112 + c) << 7 | m in the
     right half of the right register, no cross-talk;
  2. encode -> decode -> unscale -> escapes rebuilds W bit for bit (all 65,536 BF16 patterns, Gaussian
     weights at the real shapes);
  3. kernel emulation for every lab config (F12 and the BF16 twin): weight slots read exactly once, the
     ring hands each step its own data, every (row, token) output written once, and the result equals
     FP64 x @ W.T (to fp64 rounding) on normal inputs and exactly on probe inputs (escape columns in).

Run: .venv/bin/python proj/local/check_f12m.py
"""

import math
import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
PROJ = LOCAL.parent
sys.path.insert(0, str(LOCAL))
for sub in ("f12v", "f12m", "p9"):
    sys.path.insert(0, str(PROJ / sub))
import check_f12v as cv  # noqa: E402
from f12m_layout import f12m_device_planes, f12m_encode  # noqa: E402
from f12v_layout import f12v_words  # noqa: E402
from p9_layout import k_perm, p9_pack  # noqa: E402

M32 = 0xFFFFFFFF
TEXT = [
    "for (int s = 0; s < D; ++s) {", "loadw(s, s);", "if (s < XD) loadx(s, s);", "for (int s0 = 0; s0 < NS; s0 += D) {",
    "for (int slot = 0; slot < D; ++slot) {", "const int s = s0 + slot;", "const int xs = slot % XD;",
    "if (s + D < NS) loadw(slot, s + D);", "if (s + XD < NS) loadx(xs, s + XD);",
    "uint4 xv[XD][NT][2];", "static_assert(XD >= 1 && XD <= D && D % XD == 0, \"f12m x ring depth\");",
    "const int rg = f12m_wrg(warp, TG, KSW), tg = f12m_wtg(warp, TG, KSW), ks = warp % KSW;",
    "const int tb = MT * tg;               // this warp's first token", "const int g = lane >> 2, t = lane & 3;",
    "for (int n = 0; n < NT; ++n) tok_ok[n] = tb + 8 * n + g < m;", "constexpr int MT = 8 * NT;            // token slots per warp",
    "constexpr int MTC = MT * TG;          // token slots per CTA", "constexpr int NTH = RG * TG * KSW * 32;",
    "__shared__ float part[KSW][RB][MTC];", "const int mo = m < MTC ? m : MTC;",
    "ld_w_l1(wp + f12m_widx(tile, CHS, c, j, lane), wv[slot][i][j]);",
    "const int c = f12m_chunk(static_cast<int>(rank), ks, s, NS, KSW);", "const long long tile = f12m_tile(bx, rg, i, RT, RTW);",
    "const uint32_t rank = cluster_rank<CS>();", "const long long bx = blockIdx.x / CS;",
    "constexpr int CHS = NS * KSW * CS;    // 64-wide chunks per row",
    "ld_w(wp + f12m_widx(tile, CHS, c, j, lane), wv[slot][i][j]);",
    "ld_w(wp + f12m_bidx(tile, KSS, c, j, lane), wv[slot][i][j]);",
    "ld_x(x + f12m_xidx(tb + 8 * n + g, K8, c, t, 0), xv[xslot][n][0]);",
    "ld_x(x + f12m_xidx(tb + 8 * n + g, K8, c, t, 1), xv[xslot][n][1]);",
    "ld_x(rec + f12m_ridx(bx * RB + tid, 0), rc0);", "ld_x(rec + f12m_ridx(bx * RB + tid, 1), rc1);",
    "const uint4& lv = wv[slot][i][f12m_lslot(j)];",
    "a[i] = f12m_decode(word(wv[slot][i][0], j), word(lv, f12m_lword(j, 0)), word(lv, f12m_lword(j, 1)));",
    "a[i] = wv[slot][i][j];",
    "const uint32_t b0 = word(xv[xs][n][j >> 1], (2 * j) & 3);",
    "const uint32_t b1 = word(xv[xs][n][j >> 1], (2 * j + 1) & 3);",
    "for (int i = 0; i < RTW; ++i) mma(acc[i][n], a[i], b0, b1);",
    "for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][tb + f12m_ctok(n, t, r)] = acc[i][n][r];",
    "recs[2 * tid] = rc0;", "recs[2 * tid + 1] = rc1;",
    "for (int idx = tid; idx < RB * mo; idx += NTH) {", "const int tok = idx / RB, row = idx % RB;",
    "for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];",
    "for (int p = 0; p < KSW; ++p) own += part[p][row][tok];", "part[0][row][tok] = own;", "cluster_sync_all();",
    "for (int idx = tid * CS + static_cast<int>(rank); idx < RB * mo; idx += NTH * CS) {",
    "const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&part[0][row][tok]));",
    "for (int p = 0; p < CS; ++p) sum += ld_peer_f32(a, static_cast<uint32_t>(p));",
    "const long long n_rows = static_cast<long long>(gridDim.x / CS) * RB;",
    "if constexpr (CS > 1) cluster_sync_all();   // peers may still read this CTA's sums",
    "const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];",
    "const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;", "sum *= f12m_scale(base);",
    "const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};",
    "const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);",
    "sum = fmaf(-xk, wdec, sum);", "sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);",
    "out[static_cast<long long>(tok) * n_rows + bx * RB + row] = __float2bfloat16_rn(sum);",
]


def helpers():
    import f12m
    cv.SRC = f12m.F12M_SRC
    for stmt in TEXT:
        assert cv.SRC.count(stmt) >= 1, f"kernel text changed, update the emulation: {stmt}"
    names = ("tile", "chunk", "widx", "bidx", "xidx", "lslot", "lword", "crow", "ctok", "ridx", "decode", "scale", "wdec",
             "wrg", "wtg")
    return {n: cv.parse("f12m_" + n) for n in names}


def u32(t):
    return t.to(torch.int64) & M32


def bf16_bits_to_f64(b):
    return cv.bits_to_f64(u32(b) << 16)


def check_decode(dec):
    g = torch.Generator().manual_seed(9)
    n = 2 * 16 * 128
    combo = torch.arange(n)
    for pos in range(8):
        s = torch.randint(0, 2, (n, 8), generator=g)
        c = torch.randint(0, 16, (n, 8), generator=g)
        m = torch.randint(0, 128, (n, 8), generator=g)
        s[:, pos], c[:, pos], m[:, pos] = combo >> 11, (combo >> 7) & 15, combo & 127
        q, l0, l1 = f12v_words(s, c, m)
        regs = [u32(r) for r in dec(q, l0, l1)]
        for e in range(8):
            got = (regs[e >> 1] >> (16 * (e & 1))) & 0xFFFF
            want = (s[:, e] << 15) | ((112 + c[:, e]) << 7) | m[:, e]
            assert torch.equal(got, want), f"decode wrong at element {e} (varying {pos})"
    print("ok: CUDA f12m_decode (parsed) exact for every sign/code/mantissa at every element, no cross-talk")


def regs_of(p, dec):
    """Decoded A registers [RT, KS, 32, 4] (uint32) from planes P [RT, ch, 3, 32, 4] via the parsed decode."""
    rt, ch = p.shape[0], p.shape[1]
    out = torch.zeros(rt, ch, 4, 32, 4, dtype=torch.int64)
    for j in range(4):
        q = p[:, :, 0, :, j]
        lv = p[:, :, 1 + (j >> 1)]
        r = dec(q, lv[..., 2 * (j & 1)], lv[..., 2 * (j & 1) + 1])
        for i in range(4):
            out[:, :, j, :, i] = u32(r[i])
    return out.view(rt, ch * 4, 32, 4)


def decode_weights(planes, k, dec):
    """FP64 W [N, K] from F12M planes (parsed decode, inverse fragment order, per-row unscale, escapes
    from the records): the dry runs' mock kernel."""
    p = u32(planes["p"].to(torch.int64))
    rt = p.shape[0]
    n = 16 * rt
    regs = regs_of(p, dec)
    el = torch.stack([(regs[..., e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    v = el.view(rt, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
    wb = torch.empty_like(v)
    wb[:, k_perm(k)] = v
    rec = u32(planes["rec"].to(torch.int64)).view(n, 8)
    base = rec[:, 0] & 0xFF
    w = bf16_bits_to_f64(wb) * torch.pow(2.0, (base - 112).double()).view(-1, 1)
    count = (rec[:, 0] >> 8) & 0xFF
    for j in range(7):
        rows = (count > j).nonzero().flatten()
        e = rec[rows, 1 + j]
        w[rows, e & 0xFFFF] = bf16_bits_to_f64(e >> 16)
    return w


def check_roundtrip(dec):
    def recon(w):
        n, k = w.shape
        enc = f12m_encode(w)
        regs = regs_of(enc["p"], dec)                      # [RT, KS, 32, 4]
        el = torch.stack([(regs[..., e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
        # undo the fragment order and k_perm (inverse of p9_pack)
        v = el.view(n // 16, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
        wb = torch.empty_like(v)
        wb[:, k_perm(k)] = v
        e = ((wb >> 7) & 0xFF) - 112 + enc["base"].view(-1, 1)
        bits = ((wb >> 15) << 15) | (e << 7) | (wb & 0x7F)
        bits[enc["esc_row"], enc["esc_k"]] = enc["esc_bits"]
        return bits, enc
    allbits = torch.arange(65536, dtype=torch.int64).to(torch.int16)
    perm = torch.randperm(65536, generator=torch.Generator().manual_seed(0))
    w = allbits[perm].view(torch.bfloat16).view(16, 4096)
    bits, enc = recon(w)
    assert torch.equal(bits, w.view(torch.int16).to(torch.int64) & 0xFFFF), "all-patterns reconstruction differs"
    print(f"ok: all 65,536 BF16 bit patterns round-trip exactly through the fragment layout (escape rate {enc['n_esc'] / enc['n']:.3f})")
    torch.manual_seed(1)
    for name, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "unpooler": (8192, 4096)}.items():
        w = (torch.randn(n, k) / math.sqrt(k)).to(torch.bfloat16)
        bits, enc = recon(w)
        assert torch.equal(bits, w.view(torch.int16).to(torch.int64) & 0xFFFF), name
        print(f"ok: {name:8s} Gaussian round-trip exact; escapes {enc['n_esc'] / enc['n'] * 100:.4f}%")


def emulate(inst, planes, frag, x, n_rows, h, ctas=None):
    """The kernel per its text in float64 for the CTAs in `ctas` (all by default). planes: F12 planes;
    frag: P9 BF16 fragments [RT, KS, 32, 8]. Returns {(row, tok): value} as a dense [n_rows, m] tensor
    (NaN where not computed)."""
    inst = tuple(inst) + ((1,) if len(inst) == 7 else ())
    inst = inst + ((inst[5],) if len(inst) == 8 else ())
    nt, rtw, rg_n, ksw, ns, d, f12, cs, xd, tgn = inst + ((1,) if len(inst) == 9 else ())
    mtc = 8 * nt * tgn                                    # token slots per CTA
    rt_n, rb, chs = rg_n * rtw, 16 * rg_n * rtw, ns * ksw * cs
    k, kss, k8 = 64 * chs, 4 * chs, 8 * chs
    m = x.shape[0]
    grid = n_rows // rb                                   # clusters (= CTAs when CS = 1)
    assert n_rows % rb == 0 and ns % d == 0 and m <= mtc
    assert 1 <= xd <= d and d % xd == 0, inst
    # warp -> (row group, token group, k part) is a bijection (parsed f12m_wrg / f12m_wtg, ks = warp % KSW)
    nwarps = rg_n * tgn * ksw
    trip = {(int(h["wrg"](w_, tgn, ksw)), int(h["wtg"](w_, tgn, ksw)), w_ % ksw) for w_ in range(nwarps)}
    assert trip == {(a, b_, c_) for a in range(rg_n) for b_ in range(tgn) for c_ in range(ksw)}, (inst, "warp map")
    ring = {s: s for s in range(d)}
    xring = {s: s for s in range(d) if s < xd}               # prologue: loadw(s, s); if (s < XD) loadx(s, s)
    for s0 in range(0, ns, d):
        for slot in range(d):
            s = s0 + slot
            xs = slot % xd
            assert ring[slot] == s, (inst, s)
            assert xring[xs] == s, (inst, s, "x ring hands a step another chunk's x")
            if s + d < ns:
                ring[slot] = s + d
            if s + xd < ns:
                xring[xs] = s + xd
    ar = torch.arange
    # index coverage over the whole grid (cheap)
    bx = ar(grid).view(-1, 1, 1, 1, 1, 1, 1)
    rk = ar(cs).view(1, -1, 1, 1, 1, 1, 1)
    rg = ar(rg_n).view(1, 1, -1, 1, 1, 1, 1)
    ii = ar(rtw).view(1, 1, 1, -1, 1, 1, 1)
    ks = ar(ksw).view(1, 1, 1, 1, -1, 1, 1)
    ss = ar(ns).view(1, 1, 1, 1, 1, -1, 1)
    lane = ar(32).view(1, 1, 1, 1, 1, 1, -1)
    tile = h["tile"](bx, rg, ii, rt_n, rtw).expand(grid, cs, rg_n, rtw, ksw, ns, 32)
    cc = h["chunk"](rk, ks, ss, ns, ksw).expand_as(tile)
    lanes = lane.expand_as(tile)
    if f12:
        reads = torch.cat([h["widx"](tile, chs, cc, j, lanes).flatten() for j in range(3)])
        total = n_rows // 16 * chs * 3 * 32
    else:
        reads = torch.cat([h["bidx"](tile, kss, cc, j, lanes).flatten() for j in range(4)])
        total = n_rows // 16 * kss * 32
    assert torch.equal(torch.sort(reads).values, ar(total)), (inst, "weight slots not read exactly once")
    idx_all = ar(rb * min(m, mtc))
    assert torch.equal(torch.sort((idx_all // rb) * rb + idx_all % rb).values, idx_all)
    # final items: rank r takes idx = tid * CS + r, stepping threads * CS (the kernel's loop): each once
    nth = 32 * rg_n * ksw * tgn
    owned = [idx for r in range(cs) for tid in range(nth)
             for idx in range(tid * cs + r, rb * min(m, mtc), nth * cs)]
    assert sorted(owned) == list(range(rb * min(m, mtc))), (inst, "final items not covered exactly once")
    # math for the chosen CTAs
    ctas = list(range(grid)) if ctas is None else sorted(set(c for c in ctas if 0 <= c < grid))
    p4 = u32(planes["p"].view(-1, 4)) if f12 else None
    fr = u32(frag.contiguous().view(torch.int32).view(-1, 4)) if not f12 else None
    xw = u32(x.contiguous().view(torch.int32)).view(-1, 4)     # x uint4 words
    out = torch.full((n_rows, m), float("nan"), dtype=torch.float64)
    gl, tl_ = ar(32) >> 2, ar(32) & 3
    for b in ctas:
      owns = []
      for rank in range(cs):
        part = torch.full((ksw, rb, mtc), float("nan"), dtype=torch.float64)
        for rgv in range(rg_n):
          for tgv in range(tgn):
            tb = 8 * nt * tgv                             # this warp's first token
            for ksv in range(ksw):
                for i in range(rtw):
                    tl = int(h["tile"](b, rgv, i, rt_n, rtw))
                    acc = torch.zeros(nt, 16, 8, dtype=torch.float64)       # C tiles per token tile
                    for s in range(ns):
                        c = int(h["chunk"](rank, ksv, s, ns, ksw))
                        for j in range(4):
                            if f12:
                                q = p4[h["widx"](tl, chs, c, 0, ar(32))][:, j]
                                lv = p4[h["widx"](tl, chs, c, h["lslot"](j), ar(32))]
                                regs = [u32(r) for r in h["decode"](q, lv[:, h["lword"](j, 0)], lv[:, h["lword"](j, 1)])]
                            else:
                                a4 = fr[h["bidx"](tl, kss, c, j, ar(32))]
                                regs = [a4[:, r] for r in range(4)]
                            a = torch.zeros(16, 16, dtype=torch.float64)
                            for r in range(4):
                                for pp in range(2):
                                    a[gl + 8 * (r & 1), 2 * tl_ + 8 * (r >> 1) + pp] = bf16_bits_to_f64(regs[r] >> (16 * pp))
                            for n in range(nt):
                                tok = tb + 8 * n + gl
                                ok = tok < m
                                bmat = torch.zeros(16, 8, dtype=torch.float64)
                                for hh, (wsel, roff) in enumerate((((2 * j) & 3, 0), ((2 * j + 1) & 3, 8))):
                                    xv = torch.zeros(32, dtype=torch.int64)
                                    src = h["xidx"](tok.clamp(max=m - 1), k8, c, tl_, j >> 1)
                                    xv = torch.where(ok, xw[src][ar(32), wsel], xv)
                                    for pp in range(2):
                                        bmat[2 * tl_ + pp + roff, gl] = bf16_bits_to_f64(xv >> (16 * pp))
                                acc[n] += a @ bmat
                    for n in range(nt):
                        for r in range(4):
                            crow = h["crow"](rgv, rtw, i, gl, r)
                            ctok = tb + h["ctok"](n, tl_, r)
                            assert torch.isnan(part[ksv, crow, ctok]).all(), (inst, "part slot written twice")
                            part[ksv, crow, ctok] = acc[n][gl + 8 * (r >> 1), 2 * tl_ + (r & 1)]
        assert not torch.isnan(part).any(), (inst, "part slot never written")
        owns.append(part)
      if True:
        rec4 = u32(planes["rec"].view(-1, 4)) if f12 else None
        xbits = u32(x.contiguous().view(torch.int16).to(torch.int64)).view(m, k)
        for idx in range(rb * min(m, mtc)):
            tok, row = idx // rb, idx % rb
            if cs == 1:
                v = float(owns[0][:, row, tok].sum())
            else:   # each rank's KSW parts summed in place, then the ranks in order
                v = 0.0
                for rank in range(cs):
                    v += float(owns[rank][:, row, tok].sum())
            grow = b * rb + row
            if f12:
                q0, q1 = rec4[h["ridx"](b * rb + row, 0)], rec4[h["ridx"](b * rb + row, 1)]
                base, count = int(q0[0]) & 0xFF, (int(q0[0]) >> 8) & 0xFF
                v *= cv.struct_f32(h["scale"](base))
                wdec = cv.struct_f32(h["wdec"](base))
                esc = [q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]]
                for jj in range(7):
                    if jj < count:
                        kk = int(esc[jj]) & 0xFFFF
                        xk = float(bf16_bits_to_f64(xbits[tok, kk]))
                        v = v - xk * wdec + xk * cv.struct_f32(int(esc[jj]) & 0xFFFF0000)
            out[grow, tok] = v
    return out


def check_kernel():
    import f12m_configs
    h = helpers()
    torch.manual_seed(4)
    for layer, (n_all, k) in f12m_configs.SHAPES.items():
        w = (torch.randn(n_all, k) / math.sqrt(k)).to(torch.bfloat16)
        w[0, :5] = 0
        w[1, 3] = 1e4
        w[2, 77] = 1e-30
        w[17, 77] = 0
        w[n_all // 2 + 1, k - 1] = -3e3
        planes = f12m_device_planes(w)
        assert planes is not None, layer
        frag = p9_pack(w)
        runs = [(m, f12m_configs.candidates(layer, m), "F12 and BF16") for m in f12m_configs.MS]
        runs += [(m, cands, "split-K CS") for (lay, m), cands in f12m_configs.CS_CANDIDATES.items() if lay == layer]
        runs += [(m, cands, "x ring XD") for (lay, m), cands in f12m_configs.XD_CANDIDATES.items() if lay == layer]
        runs += [(m, cands, "token groups TG") for (lay, m), cands in f12m_configs.TG_CANDIDATES.items() if lay == layer]
        for m, cands, label in runs:
            n_rows = f12m_configs.n_rows(layer, m)
            wd = w[:n_rows].double()
            gen = torch.Generator().manual_seed(m)
            mm = m - 3 if m > 8 else m                       # also a partial last token tile
            x_norm = torch.randn(mm, k, generator=gen).to(torch.bfloat16)
            x_probe = torch.zeros(mm, k, dtype=torch.bfloat16)
            x_probe[0, 77] = -32.0
            x_probe[mm - 1, 3] = 0.5
            insts = sorted({f12m_configs.instance(c, layer, m) for c in cands}, key=str)
            for inst in insts:
                rb = 16 * inst[1] * inst[2]
                grid = n_rows // rb
                ctas = [0, 17 // rb, (n_all // 2 + 1) // rb, grid - 1]   # the planted rows and the last tile
                for x, exact in ((x_norm, False), (x_probe, True)):
                    got = emulate(inst, planes, frag, x, n_rows, h, ctas)
                    ref = (x.double() @ wd.T).T                 # [n_rows, m]
                    sel = ~torch.isnan(got)
                    assert int(sel.sum()) == len({c for c in ctas if c < grid}) * rb * mm, (layer, m, inst)
                    if exact:
                        assert torch.equal(got[sel], ref[sel]), (layer, m, inst, "probe not exact")
                    else:
                        scale = (wd.abs() @ x.double().abs().T)
                        err = float(((got - ref).abs() / scale.clamp(min=1e-30))[sel].max())
                        assert err < 1e-12, (layer, m, inst, err)
            what = label
            if label == "split-K CS":
                what += " " + "/".join(sorted({str(i[7]) for i in insts}))
            elif label == "x ring XD":
                what += " " + ", ".join(f"D{i[5]}/XD{i[8]}" for i in insts)
            elif label == "token groups TG":
                what += " " + ", ".join(f"NT{i[0]}/KSW{i[3]}/D{i[5]}/TG{i[9]}" for i in insts)
            print(f"ok: {layer:8s} m{m:<2d} kernel emulation exact for {len(insts)} configs ({what}; probe rows bit exact)")


def main():
    h = helpers()
    check_decode(h["decode"])
    check_roundtrip(h["decode"])
    check_kernel()


if __name__ == "__main__":
    main()

"""Exactness of SX (proj/sx/sx.py, m = 32/128/256 mma.sync GEMM, 12-bit or BF16 weights), on the CUDA text:
every index helper and the 12-bit decode are parsed and executed (check_f12v's translator), and the kernel is
emulated with the PTX ISA 9.4 layouts:
  - cp.async x stages: each copy_x writes its chunk's [KSW][MT][128 B] image (sx_xoff swizzle, zero fill for
    tokens >= m) into its slot when its commit group completes (cp.async.wait_group N completes all but the
    newest N groups); a read finds the expected chunk in its slot and no pending copy into that slot;
  - ldmatrix (m8n8 .b16): threads 8i..8i+7 give the row addresses of matrix i, thread l receives row l / 4,
    elements 2 (l % 4) and +1 (x2: threads 0..15);
  - mma.m16n8k16: A register j of lane (g, t) = A[g + 8 (j & 1)][2 t + 8 (j >> 1) + p]; b0/b1 = B[2 t + p
    (+ 8)][g]; c[q] = C[g + 8 (q >> 1)][2 t + (q & 1)];
  - escapes only where the (tile, chunk) flag is set, records from the CTA's shared copy, x from the stage;
  - the [ks][token][PROW] partial tiles and the cluster reduction by the parsed item map, row scale (F12).
Checks per config: weight slots and flags read exactly once (token group 0; other groups read the same),
stages fully written, partial tile entries written once, outputs reduced and written exactly once, result =
FP64 x @ W.T (fp64 rounding) and bit exact on probe inputs; sx_flags marks exactly the escaped chunks.

Run: .venv/bin/python proj/local/check_sx.py
"""

import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
for sub in ("sx", "f12m", "f12v", "p9"):
    sys.path.insert(0, str(LOCAL.parent / sub))
sys.path.insert(0, str(LOCAL))
import check_f12v as cv  # noqa: E402
import sx  # noqa: E402
from f12m_layout import f12m_device_planes  # noqa: E402
from sx_layout import sx_bf16_frag, sx_flags  # noqa: E402

TEXT = [
    "const int rg = sx_wrg(warp, TG, KSW), tgi = sx_wtg(warp, TG, KSW), ks = sx_wks(warp, KSW);",
    "const long long cl = blockIdx.x / CS;", "const int tb = tgi * 8 * NT8;",
    "for (int r = tid; r < 2 * ROWS; r += NTH) recs[r] = rec[2 * cl * ROWS + r];",
    "const int ksp = p / (MT * 8), rem = p % (MT * 8), tok = rem >> 3, j = rem & 7;",
    "for (int p = tid; p < KSW * MT * 8; p += NTH) {",
    "const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ksp, i);",
    "const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;",
    "cp16(sbase + static_cast<uint32_t>(s * STAGE + ksp * MT * 128 + sx_xoff(tok, j)), src, ok ? 16u : 0u);",
    "const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ks, i);",
    "const long long tile = sx_tile(cl, rg, RTW, RGN, r);",
    "if constexpr (F12) fl[sl][r] = ld_flag(flags + sx_fidx(tile, CHS, c));",
    "ld_w(wp + sx_widx(tile, CHS, c, j, lane), wv[sl][r][j]);", "ld_w(wp + sx_bidx(tile, KSS, c, j, lane), wv[sl][r][j]);",
    "if (sl < NS) load_w(sl, sl);",
    "copy_x(0, 0);\n    cp_commit();\n    for (int s = 1; s < NS; ++s) copy_x(s, s);\n    cp_commit();",
    "for (int s = 0; s < XS - 1; ++s) {\n      if (s < NS) copy_x(s, s);\n      cp_commit();",
    "if (i == 0) {\n            cp_wait<1>();\n            __syncthreads();",
    "} else if (i == 1) {\n            cp_wait<0>();\n            __syncthreads();",
    "cp_wait<XS - 2>();\n          __syncthreads();",
    "if (i + XS - 1 < NS) copy_x(i + XS - 1, (i + XS - 1) % XS);\n          cp_commit();",
    "const uint32_t st = sbase + static_cast<uint32_t>((i % XS) * STAGE + ks * MT * 128);",
    "const uint4& lv = wv[sl][r][sx_lslot(j)];",
    "a[r] = sx_decode(word(wv[sl][r][0], j), word(lv, sx_lword(j, 0)), word(lv, sx_lword(j, 1)));",
    "a[r] = wv[sl][r][j];",
    "ldm_x2(st + static_cast<uint32_t>(sx_xoff(sx_ltok(tb, 0, lane & 15), sx_lkch(j, lane & 15))), b0, b1);",
    "for (int r = 0; r < RTW; ++r) mma(acc[r][0], a[r], b0, b1);",
    "ldm_x4(st + static_cast<uint32_t>(sx_xoff(sx_ltok(tb, p, lane), sx_lkch(j, lane))), b[0], b[1], b[2], b[3]);",
    "mma(acc[r][2 * p], a[r], b[0], b[1]);\n                mma(acc[r][2 * p + 1], a[r], b[2], b[3]);",
    "const unsigned short* xs = reinterpret_cast<const unsigned short*>(smem + (i % XS) * STAGE + ks * MT * 128);",
    "if (fl[sl][r] == 0u) continue;",
    "const int lrow = sx_crow(rg, RTW, r, g, 2 * h);",
    "const uint32_t* rw = reinterpret_cast<const uint32_t*>(recs);",
    "const uint32_t w0 = rw[8 * lrow];",
    "const int base = static_cast<int>(w0 & 0xFFu), count = static_cast<int>((w0 >> 8) & 0xFFu);",
    "for (int jj = 0; jj < count; ++jj) {", "const uint32_t ew = rw[8 * lrow + 1 + jj];",
    "const int kk = static_cast<int>(ew & 0xFFFFu) - 64 * c;", "if (kk >= 0 && kk < 64) {",
    "const float wt = ldexpf(__uint_as_float(ew & 0xFFFF0000u), 112 - base);",
    "const int tok = sx_ctok(tb, n, t, 2 * h + e);",
    "const float xk = __uint_as_float(static_cast<uint32_t>(xs[(sx_xoff(tok, kk >> 3) >> 1) + (kk & 7)]) << 16);",
    "acc[r][n][2 * h + e] = fmaf(-xk, 3.0517578125e-05f, acc[r][n][2 * h + e]);   // 2^-15",
    "acc[r][n][2 * h + e] = fmaf(xk, wt, acc[r][n][2 * h + e]);",
    "if (i + D < NS) load_w(sl, i + D);",
    "part[(ks * MT + sx_ctok(tb, n, t, q)) * PROW + sx_crow(rg, RTW, r, g, q)] = acc[r][n][q];",
    "constexpr int SHARE = MT / CS, GROUPS = ROWS / 8;",
    "for (int idx = tid; idx < SHARE * GROUPS; idx += NTH) {",
    "const int tok = sx_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = sx_erow(idx, GROUPS);",
    "const int off = (kp * MT + tok) * PROW + row;",
    "lo = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(pr));",
    "hi = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off + 4), static_cast<uint32_t>(pr));",
    "for (int ii = 0; ii < 8; ++ii) v[ii] *= __uint_as_float(((recs[2 * (row + ii)].x & 0xFFu) + 15u) << 23);",
    "*reinterpret_cast<uint4*>(out + static_cast<long long>(tok) * n_rows + cl * ROWS + row) =",
    "constexpr bool RES = XS >= NS;", "constexpr int PROW = ROWS + 4;",
    "constexpr int MT = TG * 8 * NT8;", "constexpr int ROWS = 16 * RTW * RGN;", "constexpr int CHS = NS * KSW * CS;",
    "constexpr int STAGE = KSW * MT * 128;",
]
NAMES = ("wrg", "wtg", "wks", "chunk", "tile", "xoff", "widx", "bidx", "fidx", "lslot", "lword", "ltok", "lkch",
         "crow", "ctok", "etok", "erow", "decode")
LANES = torch.arange(32)
GL, TL = LANES >> 2, LANES & 3


def helpers():
    cv.SRC = sx.sx_source(sx.SX_CHECK_INSTANCES, sx.SXW_CHECK_INSTANCES)
    for stmt in TEXT + TEXT_W:
        assert cv.SRC.count(stmt) >= 1, f"kernel text changed, update check_sx.py: {stmt!r}"
    h = {n: cv.parse("sx_" + n) for n in NAMES}
    h.update({"w" + n: cv.parse("sxw_" + n) for n in W_NAMES})
    return h


# SXW (wgmma variant): statements the emulation mirrors
TEXT_W = [
    "constexpr int XSL = P + 2, WSL = P + 1;", "constexpr int STAGE = KSW * N * 128;",
    "constexpr int XAREA = XSL * STAGE, WAREA = (NTH / 32) * WSL * 1536;", "constexpr int N = 128 * NB;",
    "constexpr int ROWS = 64 * WGS;", "constexpr int NTH = 128 * WGS * KSW;",
    "const int wgi = warp >> 2, q = warp & 3;", "const int wrg = sxw_wrg(wgi, KSW), wks = sxw_wks(wgi, KSW);",
    "const long long tile = sxw_tile(cl, wrg, q, WGS);",
    "const int s = i % XSL;", "for (int p = tid; p < KSW * N * 8; p += NTH) {",
    "const int ksp = p / (N * 8), rem = p % (N * 8), tok = rem >> 3, j = rem & 7;",
    "cp16(sbase + static_cast<uint32_t>(s * STAGE + ksp * N * 128 + sx_xoff(tok, j)), src, ok ? 16u : 0u);",
    "const int c = sx_chunk(static_cast<int>(rank), NS, KSW, wks, i);",
    "cp16(sbase + static_cast<uint32_t>(XAREA + sxw_wring(warp, i % WSL, WSL) + 512 * j + 16 * lane),\n           wp + sx_widx(tile, CHS, c, j, lane), 16u);",
    "for (int s = 0; s < P; ++s) {\n    if (s < NS) copy(s);\n    cp_commit();",
    "cp_wait<P - 1>();\n        __syncthreads();", "if (i + P < NS) copy(i + P);\n        cp_commit();",
    "const uint4* wr = reinterpret_cast<const uint4*>(smem + XAREA + sxw_wring(warp, i % WSL, WSL));",
    "const uint4 w0 = wr[lane], w1 = wr[32 + lane], w2 = wr[64 + lane];",
    "const uint4& lv = j < 2 ? w1 : w2;",
    "a[b][j] = sx_decode(word(w0, j), word(lv, sx_lword(j, 0)), word(lv, sx_lword(j, 1)));",
    "const uint32_t xs = sbase + static_cast<uint32_t>((i % XSL) * STAGE);",
    "for (int nb = 0; nb < NB; ++nb) wgmma_rs_128(acc[nb], a[b][j], sxw_desc(xs + sxw_boff(wks, nb, j, N)), 1u);",
    "wgmma_commit();\n        wgmma_wait<1>();",
    "const int k0 = 64 * sx_chunk(static_cast<int>(rank), NS, KSW, wks, 0);",
    "const int lrow = sxw_row(wrg, q, g, 2 * h);", "if (kk >= k0 && kk < k0 + 64 * NS) {",
    "if (((e >> 1) & 1) == h) {", "const int tok = sxw_tok(nb, t, e);",
    "part[(wks * N + sxw_tok(nb, t, e)) * PROW + sxw_row(wrg, q, g, e)] = acc[nb][e];",
    "constexpr int SHARE = N / CS, GROUPS = ROWS / 8;", "const int off = (kp * N + tok) * PROW + row;",
    "for (int i0 = 0; i0 < NS; i0 += 2) {",
]
W_NAMES = ("wrg", "wks", "tile", "row", "tok", "wring", "boff")


def check_w_desc():
    """sxw_desc must be WG's wg_desc (K-major 128B swizzle; run on the judge's H200 in PE11-PE13), and the
    wgmma asm the hardware-proven one."""
    sys.path.insert(0, str(LOCAL.parent / "wg"))
    import wg
    body_w = cv.body_of("sxw_desc")[1]
    src_wg = wg.WG_SRC
    i = src_wg.index(" wg_desc(")
    j = src_wg.index("{", i)
    body_g = src_wg[j + 1:src_wg.index("}", j)]
    assert body_w == body_g, "sxw_desc differs from WG's descriptor"
    assert sx.sx_wgmma_rs_asm(128) == wg.wgmma_rs_asm(128)


def wgmma_b(img, start, rows):
    """The rows x 16 K-major operand a descriptor (start, SBO 1024, 128B swizzle) selects (vectorized)."""
    r = torch.arange(rows).view(-1, 1)
    kk = torch.arange(16).view(1, -1)
    a = start + (r % 8) * 128 + (r // 8) * 1024 + 2 * kk
    a = a ^ (((a >> 7) & 7) << 4)
    return img[a // 2]


class Stages:
    """cp.async stage slots: copies land when their commit group completes."""

    def __init__(self, nslots, size):
        self.img = [torch.full((size,), float("nan"), dtype=torch.float64) for _ in range(nslots)]
        self.chunk = [None] * nslots
        self.open, self.groups = [], []

    def copy(self, slot, i, writes):
        self.open.append((slot, i, writes))

    def commit(self):
        self.groups.append(self.open)
        self.open = []

    def wait(self, n):
        assert not self.open, "copies issued after the last commit"
        while len(self.groups) > n:
            for slot, i, writes in self.groups.pop(0):
                img = torch.full_like(self.img[slot], float("nan"))
                for off, vals in writes:
                    assert torch.isnan(img[off:off + 8]).all(), "stage piece written twice"
                    img[off:off + 8] = vals
                assert not torch.isnan(img).any(), "x stage not fully written"
                self.img[slot], self.chunk[slot] = img, i

    def read(self, slot, i):
        assert self.chunk[slot] == i, f"slot {slot} holds chunk {self.chunk[slot]}, expected {i}"
        for grp in self.groups + [self.open]:
            assert all(s != slot for s, _, _ in grp), "read while a copy into the slot is pending"
        return self.img[slot]


def emulate(cfg, wt, x, n, h, counters):
    """out [m, n] float64 per the kernel text. wt: F12 {"p4", "rec", "flags"} or BF16 {"frag"}."""
    nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = cfg
    mt, rows, nth = tg * 8 * nt8, 16 * rtw * rgn, 32 * rgn * tg * ksw
    chs = ns * ksw * cs
    k, kss = 64 * chs, 4 * chs
    stage_sz = ksw * mt * 64                      # bf16 values per stage
    res = xs >= ns
    prow = rows + 4
    m = x.shape[0]
    assert x.shape[1] == k and m <= mt and n % rows == 0 and mt % cs == 0
    out = torch.full((m, n), float("nan"), dtype=torch.float64)
    seen_out = torch.zeros(m, n, dtype=torch.int64)
    xd = x.double()
    nwarps = nth // 32
    for cl in range(n // rows):
        parts, recs_cta = [], None
        if f12:
            ru4 = wt["rec"].view(-1, 4)
            recs_cta = torch.stack([ru4[2 * cl * rows + r] for r in range(0, 2 * rows)])      # recs[r] = rec[2 cl ROWS + r]
        for rank in range(cs):
            st = Stages(ns if res else xs, stage_sz)

            def copy_x(i, s):
                writes = []
                for p in range(ksw * mt * 8):
                    ksp, rem = p // (mt * 8), p % (mt * 8)
                    tok, j = rem >> 3, rem & 7
                    c = h["chunk"](rank, ns, ksw, ksp, i)
                    off = ksp * mt * 128 + h["xoff"](tok, j)
                    assert off % 16 == 0 and 0 <= off < ksw * mt * 128
                    vals = xd[tok, 64 * c + 8 * j:64 * c + 8 * j + 8] if tok < m else torch.zeros(8, dtype=torch.float64)
                    writes.append((off // 2, vals))
                st.copy(s, i, writes)

            # prologue (weights are register loads; only the x protocol is simulated)
            if res:
                copy_x(0, 0)
                st.commit()
                for s in range(1, ns):
                    copy_x(s, s)
                st.commit()
            else:
                for s in range(xs - 1):
                    if s < ns:
                        copy_x(s, s)
                    st.commit()
            acc = torch.zeros(nwarps, rtw, nt8, 32, 4, dtype=torch.float64)
            ring = {}
            for warp in range(nwarps):
                for sl in range(d):
                    if sl < ns:
                        ring[(warp, sl)] = sl
            for i in range(ns):
                if res:
                    if i == 0:
                        st.wait(1)
                    elif i == 1:
                        st.wait(0)
                else:
                    st.wait(xs - 2)
                    if i + xs - 1 < ns:
                        copy_x(i + xs - 1, (i + xs - 1) % xs)
                    st.commit()
                img = st.read(i % xs, i)
                for warp in range(nwarps):
                    rg, tgi, ks = h["wrg"](warp, tg, ksw), h["wtg"](warp, tg, ksw), h["wks"](warp, ksw)
                    tb = tgi * 8 * nt8
                    sl = i % d
                    assert ring[(warp, sl)] == i, "weight ring slot holds another chunk"
                    c = h["chunk"](rank, ns, ksw, ks, i)
                    sbase = ks * mt * 64                      # bf16 offset of the slice in the stage
                    for r in range(rtw):
                        tile = h["tile"](cl, rg, rtw, rgn, r)
                        # weights of this (tile, chunk) as the kernel's loads give them
                        if f12:
                            fi = h["fidx"](tile, chs, c)
                            flag = int(wt["flags"][fi])
                            if tgi == 0:
                                counters["flag"][fi] += 1
                            words = []
                            for j in range(3):
                                idx = h["widx"](tile, chs, c, j, LANES)
                                words.append(wt["p4"][idx])
                                if tgi == 0:
                                    counters["w"][idx] += 1
                        else:
                            words = []
                            for j in range(4):
                                idx = h["bidx"](tile, kss, c, j, LANES)
                                words.append(wt["frag"][idx])
                                if tgi == 0:
                                    counters["w"][idx] += 1
                        for j in range(4):
                            a = torch.zeros(16, 16, dtype=torch.float64)
                            if f12:
                                lv = words[h["lslot"](j)]
                                regs = [cv.u32(q) for q in h["decode"](words[0][:, j], lv[:, h["lword"](j, 0)], lv[:, h["lword"](j, 1)])]
                                for q in range(4):
                                    for pp in range(2):
                                        a[GL + 8 * (q & 1), 2 * TL + 8 * (q >> 1) + pp] = cv.bits_to_f64(((regs[q] >> (16 * pp)) & 0xFFFF) << 16)
                            else:
                                fr = words[j]                                # [32 lanes, 8] bf16 as float64
                                for q in range(4):
                                    for pp in range(2):
                                        a[GL + 8 * (q & 1), 2 * TL + 8 * (q >> 1) + pp] = fr[:, 2 * q + pp]
                            pairs = 1 if nt8 == 1 else nt8 // 2
                            for p in range(pairs):
                                la = LANES & 15 if nt8 == 1 else LANES
                                addrs = h["xoff"](h["ltok"](tb, p, la), h["lkch"](j, la))
                                rowdata = torch.stack([img[sbase + int(ad) // 2:sbase + int(ad) // 2 + 8] for ad in addrs])
                                nmat = 2 if nt8 == 1 else 4
                                for half in range(nmat // 2):
                                    b0 = torch.stack([rowdata[8 * (2 * half) + l // 4, 2 * (l % 4):2 * (l % 4) + 2] for l in range(32)])
                                    b1 = torch.stack([rowdata[8 * (2 * half + 1) + l // 4, 2 * (l % 4):2 * (l % 4) + 2] for l in range(32)])
                                    bmat = torch.zeros(16, 8, dtype=torch.float64)
                                    for pp in range(2):
                                        bmat[2 * TL + pp, GL] = b0[:, pp]
                                        bmat[2 * TL + 8 + pp, GL] = b1[:, pp]
                                    cm = a @ bmat
                                    nn = 2 * p + half
                                    for q in range(4):
                                        acc[warp, r, nn, :, q] += cm[GL + 8 * (q >> 1), 2 * TL + (q & 1)]
                        if f12 and flag:
                            for lane in range(32):
                                g, t = lane >> 2, lane & 3
                                for hh in range(2):
                                    lrow = h["crow"](rg, rtw, r, g, 2 * hh)
                                    rw = recs_cta.reshape(-1)                     # uint32 record words [ROWS * 8]
                                    w0 = int(rw[8 * lrow])
                                    base, count = w0 & 0xFF, (w0 >> 8) & 0xFF
                                    assert count <= 7
                                    for jj in range(count):
                                        ew = int(rw[8 * lrow + 1 + jj])
                                        kk = (ew & 0xFFFF) - 64 * c
                                        if 0 <= kk < 64:
                                            wtrue = cv.struct_f32(ew & 0xFFFF0000) * 2.0 ** (112 - base)
                                            for nn in range(nt8):
                                                for e in range(2):
                                                    tok = h["ctok"](tb, nn, t, 2 * hh + e)
                                                    xk = float(img[sbase + (h["xoff"](tok, kk >> 3) >> 1) + (kk & 7)])
                                                    acc[warp, r, nn, lane, 2 * hh + e] += -xk * 2.0 ** -15
                                                    acc[warp, r, nn, lane, 2 * hh + e] += xk * wtrue
                    if i + d < ns:
                        ring[(warp, sl)] = i + d
            st.wait(0)
            part = torch.full((ksw * mt * prow,), float("nan"), dtype=torch.float64)
            for warp in range(nwarps):
                rg, tgi, ks = h["wrg"](warp, tg, ksw), h["wtg"](warp, tg, ksw), h["wks"](warp, ksw)
                tb = tgi * 8 * nt8
                for r in range(rtw):
                    for nn in range(nt8):
                        for q in range(4):
                            idx = (ks * mt + h["ctok"](tb, nn, TL, q)) * prow + h["crow"](rg, rtw, r, GL, q)
                            assert torch.isnan(part[idx]).all(), "partial entry written twice"
                            part[idx] = acc[warp, r, nn, :, q]
            pv = part.view(ksw, mt, prow)
            assert not torch.isnan(pv[:, :, :rows]).any(), "partial tile not fully written"
            assert torch.isnan(pv[:, :, rows:]).all(), "padding written"
            parts.append(part)
        share, groups = mt // cs, rows // 8
        seen = torch.zeros(mt, rows, dtype=torch.int64)
        for rank in range(cs):
            for idx in range(share * groups):
                tok, row = h["etok"](rank, share, idx, groups), h["erow"](idx, groups)
                seen[tok, row:row + 8] += 1
                v = torch.zeros(8, dtype=torch.float64)
                for pr in range(cs):
                    for kp in range(ksw):
                        off = (kp * mt + tok) * prow + row
                        assert off % 4 == 0
                        v += parts[pr][off:off + 8]
                if f12:
                    v = v * torch.tensor([2.0 ** ((int(recs_cta[2 * (row + ii)][0]) & 0xFF) - 112) for ii in range(8)],
                                         dtype=torch.float64)
                if tok < m:
                    out[tok, cl * rows + row:cl * rows + row + 8] = v
                    seen_out[tok, cl * rows + row:cl * rows + row + 8] += 1
        assert torch.equal(seen, torch.ones_like(seen)), "outputs not reduced exactly once"
    assert torch.equal(seen_out, torch.ones_like(seen_out)), "outputs not written exactly once"
    return out


def make_weights(n, k, seed):
    g_ = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g_) / 22).to(torch.bfloat16)
    w[0, 3] = 0
    w[5, 63] = 1e4                    # escape at a chunk's last k
    w[5, 64] = -2e3                   # and the next chunk's first
    w[9, k - 1] = 1e-30               # last k of a row
    w[n - 1, k - 1] = -3e3            # last row, last k
    for j, kk in enumerate(esc_ks(k)):  # 7 escapes in one row (the record's limit)
        w[17, kk] = 5e3 * (j + 1)
    w[17, 1] = 1e-25
    w[33 % n, k // 2] = 7e3           # escapes at likely K-range boundaries (CS / KSW splits)
    w[40 % n, k // 4] = -6e3
    return w


def esc_ks(k):
    ks = [(70 + (k // 8) * j) % k for j in range(6)]
    assert len(set(ks + [1])) == 7
    return ks


def run(cfg, h, m, n, seed):
    nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = cfg
    k = 64 * ns * ksw * cs
    w = make_weights(n, k, seed)
    counters = {}
    if f12:
        pl = f12m_device_planes(w, natural=True)
        assert pl is not None and pl["max_row"] == 7, pl and pl["max_row"]
        rec = pl["rec"].to(torch.int64) & 0xFFFFFFFF
        flags = sx_flags(pl["rec"], n, k)
        # sx_flags marks exactly the (tile, chunk) pairs with an escape
        expect = torch.zeros(n // 16, k // 64, dtype=torch.uint8)
        r_ = rec.view(n, 8)
        for row in range(n):
            cnt = (int(r_[row, 0]) >> 8) & 0xFF
            for j in range(cnt):
                expect[row // 16, (int(r_[row, 1 + j]) & 0xFFFF) // 64] = 1
        assert torch.equal(flags.view(n // 16, k // 64), expect), "sx_flags differ from the records"
        assert int(flags.sum()) >= 4
        wt = {"p4": pl["p"].to(torch.int64).view(-1, 4) & 0xFFFFFFFF, "rec": rec, "flags": flags}
        counters["w"] = torch.zeros(wt["p4"].shape[0], dtype=torch.int64)
        counters["flag"] = torch.zeros(flags.numel(), dtype=torch.int64)
    else:
        wt = {"frag": sx_bf16_frag(w).view(-1, 8).double()}
        counters["w"] = torch.zeros(wt["frag"].shape[0], dtype=torch.int64)
    g_ = torch.Generator().manual_seed(seed + 1)
    xn = torch.randn(m, k, generator=g_).to(torch.bfloat16)
    xp = torch.zeros(m, k, dtype=torch.bfloat16)
    xp[0, 63] = 4.0
    xp[m - 1, k - 1] = -0.25
    xp[m // 2, esc_ks(k)[2]] = 2.0                 # an escape of row 17
    xp[1, 3] = 8.0
    ref_w = w.double()
    for x, exact in ((xn, False), (xp, True)):
        for key in counters:
            counters[key].zero_()
        got = emulate(cfg, wt, x, n, h, counters)
        ref = x.double() @ ref_w.T
        if exact:
            assert torch.equal(got, ref), (cfg, "probe not exact", float((got - ref).abs().max()))
        else:
            sc = x.double().abs() @ ref_w.abs().T
            err = float(((got - ref).abs() / sc).max())
            assert err < 1e-12, (cfg, err)
        assert int(counters["w"].min()) == 1 and int(counters["w"].max()) == 1, "weight slots not read exactly once"
        if f12:
            assert int(counters["flag"].min()) == 1 and int(counters["flag"].max()) == 1, "flags not read exactly once"
    return k


def emulate_w(cfg, wt, x, n, h, counters):
    """out [m, n] float64 per the SXW kernel text: cp.async groups (x slots + per-warp weight rings), async
    wgmma hazards (a copy never targets a slot an unfinished wgmma reads; a ring slot is decoded only once its
    chunk landed), RS A = decoded fragments, B through the descriptor, accumulators in the PTX m64nNk16 layout,
    escapes after the loop from x, then SX's partial tile / reduction / scale."""
    nb_, wgs, ksw, ns, pp, cs = cfg
    nn, rows, nth = 128 * nb_, 64 * wgs, 128 * wgs * ksw
    chs = ns * ksw * cs
    k = 64 * chs
    xsl, wsl = pp + 2, pp + 1
    prow = rows + 4
    m = x.shape[0]
    assert x.shape[1] == k and m <= nn and n % rows == 0 and nn % cs == 0
    xd = x.double()
    nwarps, nwg = nth // 32, nth // 128
    out = torch.full((m, n), float("nan"), dtype=torch.float64)
    seen_out = torch.zeros(m, n, dtype=torch.int64)
    ru4 = wt["rec"].view(-1, 4)
    for cl in range(n // rows):
        recs_cta = torch.stack([ru4[2 * cl * rows + r] for r in range(2 * rows)])
        rw = recs_cta.reshape(-1)
        parts = []
        for rank in range(cs):
            ximg = [torch.full((ksw * nn * 64,), float("nan"), dtype=torch.float64) for _ in range(xsl)]
            xchunk = [None] * xsl
            ring = {}                                      # (warp, byte offset) -> (chunk, words [4])
            groups, open_ = [], []
            inflight = [[] for _ in range(nwg)]            # chunks whose wgmmas may still run, per warpgroup

            def copy(i, cur=None):
                s_ = i % xsl
                busy = {cc for wq in range(nwg) for cc in inflight[wq]} | ({cur} if cur is not None else set())
                assert all(cc % xsl != s_ for cc in busy), "copy into an x slot a wgmma may still read"
                writes = []
                for pidx in range(ksw * nn * 8):
                    ksp, rem = pidx // (nn * 8), pidx % (nn * 8)
                    tok, j = rem >> 3, rem & 7
                    c = h["chunk"](rank, ns, ksw, ksp, i)
                    off = ksp * nn * 128 + h["xoff"](tok, j)
                    vals = xd[tok, 64 * c + 8 * j:64 * c + 8 * j + 8] if tok < m else torch.zeros(8, dtype=torch.float64)
                    writes.append(("x", s_, off // 2, vals))
                for warp in range(nwarps):
                    wgi, q = warp >> 2, warp & 3
                    wks = h["wwks"](wgi, ksw)
                    tile = h["wtile"](cl, h["wwrg"](wgi, ksw), q, wgs)
                    c = h["chunk"](rank, ns, ksw, wks, i)
                    for j in range(3):
                        idx = h["widx"](tile, chs, c, j, LANES)
                        counters["w"][idx] += 1
                        base = h["wwring"](warp, i % wsl, wsl) + 512 * j
                        for lane in range(32):
                            writes.append(("w", warp, base + 16 * lane, (i, wt["p4"][int(idx[lane])])))
                open_.append(writes)

            def commit():
                groups.append([w for ws in open_ for w in ws])
                open_.clear()

            def wait(nk):
                while len(groups) > nk:
                    for kind, a1, a2, val in groups.pop(0):
                        if kind == "x":
                            ximg[a1][a2:a2 + 8] = val
                        else:
                            ring[(a1, a2)] = val
            for s_ in range(pp):
                if s_ < ns:
                    copy(s_)
                commit()
            dmat = torch.zeros(nwg, 64, nn, dtype=torch.float64)    # D of each warpgroup (PTX layout applied at the end)
            for i in range(ns):
                wait(pp - 1)
                # barrier: every warpgroup did wait_group 1 after chunk i - 1
                for wq in range(nwg):
                    assert all(cc >= i - 1 for cc in inflight[wq])
                # x slot of chunk i landed completely
                s_ = i % xsl
                if i + pp < ns:
                    copy(i + pp, cur=i)
                commit()
                # chunk i's x slot: every piece written by chunk i's copies (it landed: groups <= i completed)
                assert not torch.isnan(ximg[s_]).any(), "x slot not fully written"
                for wgi in range(nwg):
                    wks = h["wwks"](wgi, ksw)
                    c = h["chunk"](rank, ns, ksw, wks, i)
                    a = torch.zeros(64, 16 * 4, dtype=torch.float64)       # A of the 4 k16 steps
                    for q in range(4):
                        warp = 4 * wgi + q
                        base = h["wwring"](warp, i % wsl, wsl)
                        pending = {(w_[1], w_[2]) for grp in groups + [[w for ws in open_ for w in ws]] for w_ in grp if w_[0] == "w"}
                        assert all((warp, base + 512 * j + 16 * lane) not in pending for j in range(3) for lane in range(32)), \
                            "decode of a ring slot with a copy pending into it"
                        words = []
                        for j in range(3):
                            got = [ring.get((warp, base + 512 * j + 16 * lane)) for lane in range(32)]
                            assert all(g_ is not None and g_[0] == i for g_ in got), "weight ring slot not holding chunk i"
                            words.append(torch.stack([g_[1] for g_ in got]))
                        for j in range(4):
                            lv = words[1] if j < 2 else words[2]
                            regs = [cv.u32(r_) for r_ in h["decode"](words[0][:, j], lv[:, h["lword"](j, 0)], lv[:, h["lword"](j, 1)])]
                            for qq in range(4):
                                for p_ in range(2):
                                    a[16 * q + GL + 8 * (qq & 1), 16 * j + 2 * TL + 8 * (qq >> 1) + p_] = \
                                        cv.bits_to_f64(((regs[qq] >> (16 * p_)) & 0xFFFF) << 16)
                    for j in range(4):
                        for nb in range(nb_):
                            start = h["wboff"](wks, nb, j, nn)                  # offset in the slot (1024-B aligned)
                            b = wgmma_b(ximg[s_], start, 128)                   # [128 tokens, 16]
                            dmat[wgi, :, 128 * nb:128 * nb + 128] += a[:, 16 * j:16 * j + 16] @ b.T
                    inflight[wgi] = [cc for cc in inflight[wgi] + [i] if cc >= i]     # wait_group 1: only chunk i left
            wait(0)
            # thread values in the PTX m64nNk16 accumulator layout, escapes, partial tile
            part = torch.full((ksw * nn * prow,), float("nan"), dtype=torch.float64)
            for warp in range(nwarps):
                wgi, q = warp >> 2, warp & 3
                wrg, wks = h["wwrg"](wgi, ksw), h["wwks"](wgi, ksw)
                k0 = 64 * h["chunk"](rank, ns, ksw, wks, 0)
                for nb in range(nb_):
                    vals = torch.zeros(32, 64, dtype=torch.float64)
                    for e in range(64):
                        vals[:, e] = dmat[wgi, 16 * q + GL + 8 * ((e >> 1) & 1), 128 * nb + 8 * (e >> 2) + 2 * TL + (e & 1)]
                    for lane in range(32):
                        g, t = lane >> 2, lane & 3
                        for hh in range(2):
                            lrow = h["wrow"](wrg, q, g, 2 * hh)
                            w0 = int(rw[8 * lrow])
                            base, count = w0 & 0xFF, (w0 >> 8) & 0xFF
                            for jj in range(count):
                                ew = int(rw[8 * lrow + 1 + jj])
                                kk = ew & 0xFFFF
                                if k0 <= kk < k0 + 64 * ns:
                                    wtrue = cv.struct_f32(ew & 0xFFFF0000) * 2.0 ** (112 - base)
                                    for e in range(64):
                                        if ((e >> 1) & 1) == hh:
                                            tok = h["wtok"](nb, t, e)
                                            xk = float(xd[tok, kk]) if tok < m else 0.0
                                            vals[lane, e] += -xk * 2.0 ** -15
                                            vals[lane, e] += xk * wtrue
                    for e in range(64):
                        idx = (wks * nn + h["wtok"](nb, TL, e)) * prow + h["wrow"](wrg, q, GL, e)
                        assert torch.isnan(part[idx]).all(), "partial entry written twice"
                        part[idx] = vals[:, e]
            pv = part.view(ksw, nn, prow)
            assert not torch.isnan(pv[:, :, :rows]).any(), "partial tile not fully written"
            parts.append(part)
        share, groups_ = nn // cs, rows // 8
        seen = torch.zeros(nn, rows, dtype=torch.int64)
        for rank in range(cs):
            for idx in range(share * groups_):
                tok, row = h["etok"](rank, share, idx, groups_), h["erow"](idx, groups_)
                seen[tok, row:row + 8] += 1
                v = torch.zeros(8, dtype=torch.float64)
                for pr in range(cs):
                    for kp in range(ksw):
                        off = (kp * nn + tok) * prow + row
                        v += parts[pr][off:off + 8]
                v = v * torch.tensor([2.0 ** ((int(recs_cta[2 * (row + ii)][0]) & 0xFF) - 112) for ii in range(8)],
                                     dtype=torch.float64)
                if tok < m:
                    out[tok, cl * rows + row:cl * rows + row + 8] = v
                    seen_out[tok, cl * rows + row:cl * rows + row + 8] += 1
        assert torch.equal(seen, torch.ones_like(seen)), "outputs not reduced exactly once"
    assert torch.equal(seen_out, torch.ones_like(seen_out)), "outputs not written exactly once"
    return out


def run_w(cfg, h, m, n, seed):
    nb_, wgs, ksw, ns, pp, cs = cfg
    k = 64 * ns * ksw * cs
    w = make_weights(n, k, seed)
    pl = f12m_device_planes(w, natural=True)
    assert pl is not None and pl["max_row"] == 7
    wt = {"p4": pl["p"].to(torch.int64).view(-1, 4) & 0xFFFFFFFF, "rec": pl["rec"].to(torch.int64) & 0xFFFFFFFF}
    counters = {"w": torch.zeros(wt["p4"].shape[0], dtype=torch.int64)}
    g_ = torch.Generator().manual_seed(seed + 1)
    xn = torch.randn(m, k, generator=g_).to(torch.bfloat16)
    xp = torch.zeros(m, k, dtype=torch.bfloat16)
    xp[0, 63] = 4.0
    xp[m - 1, k - 1] = -0.25
    xp[m // 2, esc_ks(k)[2]] = 2.0
    xp[1, 3] = 8.0
    for x, exact in ((xn, False), (xp, True)):
        counters["w"].zero_()
        got = emulate_w(cfg, wt, x, n, h, counters)
        ref = x.double() @ w.double().T
        if exact:
            assert torch.equal(got, ref), (cfg, "SXW probe not exact", float((got - ref).abs().max()))
        else:
            sc = x.double().abs() @ w.double().abs().T
            err = float(((got - ref).abs() / sc).max())
            assert err < 1e-12, (cfg, err)
        assert int(counters["w"].min()) == 1 and int(counters["w"].max()) == 1, "SXW weight slots not read exactly once"
    return k


CONFIGS_W = [
    # (NB, WGS, KSW, NS, P, CS), m, n
    ((1, 2, 1, 4, 2, 2), 100, 256),          # partial m, two clusters, P 2
    ((2, 1, 2, 3, 1, 2), 256, 128),          # N = 256 (two wgmmas per step), KSW 2, P 1 (3 x slots, 2 ring slots)
    ((1, 1, 1, 6, 3, 1), 128, 64),           # P 3, no cluster
    ((1, 2, 1, 32, 4, 4), 128, 128),         # the lab's m = 128 class at K = 8192
]


CONFIGS = [
    # (NT8, TG, RTW, RGN, KSW, NS, D, XS, F12, CS), m, n
    ((4, 1, 2, 2, 2, 4, 2, 8, True, 4), 32, 128),        # m = 32 resident x, cluster 4
    ((8, 2, 2, 1, 2, 4, 2, 3, True, 2), 100, 64),        # pipelined, 2 token groups, partial m
    ((1, 4, 1, 4, 1, 5, 2, 2, False, 2), 29, 128),       # x2 path, XS 2, NS odd, BF16
    ((8, 4, 1, 1, 1, 5, 2, 3, False, 1), 256, 32),       # m = 256, BF16, no cluster
    ((4, 1, 1, 2, 2, 1, 1, 2, True, 2), 32, 64),         # resident with NS = 1
    ((16, 1, 1, 2, 4, 2, 2, 4, True, 2), 128, 64),       # m = 128, one token group of 16 n8 tiles
    ((4, 1, 2, 4, 4, 8, 2, 8, True, 4), 32, 128),        # the lab's m = 32 shape class at K = 8192
]


def main():
    h = helpers()
    for cfg, m, n in CONFIGS:
        k = run(cfg, h, m, n, seed=sum(cfg[:8]) + m)
        print(f"ok: SX {cfg} m{m} n{n} k{k}: exact (probe bit exact), slots/flags read once, stages/partials/outputs once")
    print("ok: SX emulation (cp.async stage protocol, ldmatrix x2/x4, mma layouts, 12-bit decode, escape flags, "
          "cluster reduction, row scale) matches FP64 x @ W.T")
    check_w_desc()
    for cfg, m, n in CONFIGS_W:
        k = run_w(cfg, h, m, n, seed=sum(cfg) + m)
        print(f"ok: SXW {cfg} m{m} n{n} k{k}: exact (probe bit exact), weight slots read once, no async hazard")
    print("ok: SXW emulation (cp.async x slots + per-warp weight rings, async wgmma slot hazards, RS A fragments, "
          "descriptor B (= WG's), m64nNk16 accumulator layout, tail escapes from x, reduction, row scale) matches FP64")


if __name__ == "__main__":
    main()

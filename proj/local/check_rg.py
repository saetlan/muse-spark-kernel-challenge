"""Exactness of RG (proj/rg/rg.py, router hi + lo mma.sync kernel), on the CUDA text: every index helper is
parsed and executed (check_f12v's translator), and the kernel is emulated with the PTX ISA 9.4 layouts:
  - the x stage of each chunk is written exactly as the kernel's cp.async loop does (rg_xoff swizzle, zero
    fill for tokens >= m);
  - ldmatrix (m8n8 .b16, sec. 9.7.16.5.15): threads 8i..8i+7 give the row addresses of matrix i, thread l
    receives row l / 4, elements 2 (l % 4) and +1 of each matrix;
  - mma.m16n8k16 (sec. 9.7.16.5.8): A register j of lane (g, t) = A[g + 8 (j & 1)][2 t + 8 (j >> 1) + p];
    b0/b1 = B[2 t + p (+ 8)][g]; c[r] = C[g + 8 (r >> 1)][2 t + (r & 1)];
  - the [token][row] partial tiles and the cluster reduction by the parsed item map.
Checks per lab config and m: weight fragments read for every (tile, step, hi/lo) of each rank's chunks, x
stages fully written, outputs written exactly once, result = FP64 x @ (hi + lo).T (fp64 rounding) and bit
exact on probe rows; rg_pack's hi + lo equals SGLang's split.

Run: .venv/bin/python proj/local/check_rg.py
"""

import math
import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))
sys.path.insert(0, str(LOCAL.parent / "rg"))
import check_f12v as cv  # noqa: E402
import rg  # noqa: E402
from rg_layout import rg_pack  # noqa: E402

TEXT = [
    "const int rt = rg_wrt(warp, TG), tgi = rg_wtg(warp, TG);", "const long long tile = cl * RTC + rt;",
    "const long long cl = blockIdx.x / CS;",
    "const int c0 = rg_c0(static_cast<int>(rank), CHUNKS, CS), c1 = rg_c0(static_cast<int>(rank) + 1, CHUNKS, CS);",
    "const int tok = p >> 3, j = p & 7;", "for (int p = tid; p < MT * 8; p += 128) {",
    "const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;",
    "cp16(sbase + static_cast<uint32_t>(s * MT * 128 + rg_xoff(tok, j)), src, ok ? 16u : 0u);",
    "for (int h = 0; h < 2; ++h) ld_a(wp + rg_aidx(tile, KSS, 4 * c + j, h, lane), wa[sl][j][h]);",
    "const int tb = tgi * 8 * NTW;",
    "ldm_x2(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, 0, lane & 15), rg_lkch(j, lane & 15))), b0, b1);",
    "ldm_x4(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, p, lane), rg_lkch(j, lane))), b[0], b[1], b[2], b[3]);",
    "mma(acc[2 * p], wa[sl][j][0], b[0], b[1]);", "mma(acc[2 * p], wa[sl][j][1], b[0], b[1]);",
    "mma(acc[2 * p + 1], wa[sl][j][0], b[2], b[3]);", "mma(acc[2 * p + 1], wa[sl][j][1], b[2], b[3]);",
    "mma(acc[0], wa[sl][j][0], b0, b1);", "mma(acc[0], wa[sl][j][1], b0, b1);",
    "const uint32_t st = sbase + static_cast<uint32_t>((i % XS) * MT * 128);",
    "for (int r = 0; r < 4; ++r) part[rg_ctok(tb, n, t, r) * ROWS + rg_crow(rt, g, r)] = acc[n][r];",
    "const int tok = rg_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = rg_erow(idx, GROUPS);",
    "v[pr] = ld_peer_v4(paddr + 4u * static_cast<uint32_t>(tok * ROWS + row), static_cast<uint32_t>(pr));",
    "if (tok < m) *reinterpret_cast<float4*>(out + static_cast<long long>(tok) * 256 + cl * ROWS + row) = s;",
    "for (int idx = tid; idx < SHARE * GROUPS; idx += 128) {", "constexpr int SHARE = MT / CS, GROUPS = ROWS / 4;",
    "cp_wait<XS - 2>();", "if (i + XS - 1 < nch) copy_x(c0 + i + XS - 1, (i + XS - 1) % XS);",
    "if (i + D < nch) load_a(sl, c0 + i + D);",
]
NAMES = ("c0", "xoff", "aidx", "wrt", "wtg", "ltok", "lkch", "crow", "ctok", "etok", "erow")


def helpers():
    cv.SRC = rg.rg_source(rg.RG_CHECK_INSTANCES)
    for stmt in TEXT:
        assert cv.SRC.count(stmt) >= 1, f"kernel text changed, update check_rg.py: {stmt}"
    return {n: cv.parse("rg_" + n) for n in NAMES}


def bf(v):
    return v.to(torch.bfloat16).double()


def emulate(inst, wp, x, h):
    """out [m, 256] (float64) per the kernel text; wp: rg_pack planes [16, 512, 2, 32, 8] bf16."""
    rtc, ntw, cs, xs, dd = inst
    tg = 4 // rtc
    mt = tg * 8 * ntw
    m, k = x.shape
    chunks, kss, rows = k // 64, k // 16, 16 * rtc
    assert k == 8192 and m <= mt and mt % cs == 0
    planes = wp.view(torch.int16).view(-1, 8)                    # uint4 = 8 bf16 per lane
    lanes = torch.arange(32)
    g, t = lanes >> 2, lanes & 3
    out = torch.full((m, 256), float("nan"), dtype=torch.float64)
    a_reads = torch.zeros(16, kss, 2, dtype=torch.int64)
    for cl in range(16 // rtc):
        parts = []
        for rank in range(cs):
            c0, c1 = h["c0"](rank, chunks, cs), h["c0"](rank + 1, chunks, cs)
            acc = torch.zeros(4, ntw, 32, 4, dtype=torch.float64)   # [warp][n tile][lane][r]
            for c in range(c0, c1):
                stage = torch.full((mt * 64,), float("nan"), dtype=torch.float64)
                for p in range(mt * 8):
                    tok, j = p >> 3, p & 7
                    off = h["xoff"](tok, j)
                    assert off % 16 == 0
                    stage[off // 2:off // 2 + 8] = bf(x[tok, 64 * c + 8 * j:64 * c + 8 * j + 8]) if tok < m else 0.0
                assert not torch.isnan(stage).any(), "x stage not fully written"
                for warp in range(4):
                    rt, tgi = h["wrt"](warp, tg), h["wtg"](warp, tg)
                    tile = cl * rtc + rt
                    tb = tgi * 8 * ntw
                    for jj in range(4):
                        step = 4 * c + jj
                        afr = []
                        for hh in range(2):
                            idx = h["aidx"](tile, kss, step, hh, lanes)
                            afr.append(planes[idx].view(torch.bfloat16).double())   # [32 lanes, 8]
                            if warp % tg == 0:
                                a_reads[tile, step, hh] += 1
                        amat = torch.zeros(2, 16, 16, dtype=torch.float64)
                        for hh in range(2):
                            for j in range(4):
                                for pp in range(2):
                                    amat[hh, g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + pp] = afr[hh][:, 2 * j + pp]
                        pairs = 1 if ntw == 1 else ntw // 2
                        for p in range(pairs):
                            lane_addr = lanes & 15 if ntw == 1 else lanes
                            addrs = h["xoff"](h["ltok"](tb, p, lane_addr), h["lkch"](jj, lane_addr))
                            rowdata = torch.stack([stage[int(a) // 2:int(a) // 2 + 8] for a in addrs])   # [32, 8]
                            nmat = 2 if ntw == 1 else 4
                            regs = torch.empty(nmat, 32, 2, dtype=torch.float64)
                            for mi in range(nmat):
                                for i in range(32):
                                    src = rowdata[8 * mi + i // 4]
                                    regs[mi, i] = src[2 * (i % 4):2 * (i % 4) + 2]
                            for half in range(nmat // 2):
                                bmat = torch.zeros(16, 8, dtype=torch.float64)
                                b0, b1 = regs[2 * half], regs[2 * half + 1]
                                for pp in range(2):
                                    bmat[2 * t + pp, g] = b0[:, pp]
                                    bmat[2 * t + 8 + pp, g] = b1[:, pp]
                                cmat = (amat[0] + amat[1]) @ bmat if False else amat[0] @ bmat + amat[1] @ bmat
                                n = 2 * p + half
                                for r in range(4):
                                    acc[warp, n, :, r] += cmat[g + 8 * (r >> 1), 2 * t + (r & 1)]
            part = torch.full((mt, rows), float("nan"), dtype=torch.float64)
            for warp in range(4):
                rt, tgi = h["wrt"](warp, tg), h["wtg"](warp, tg)
                tb = tgi * 8 * ntw
                for n in range(ntw):
                    for r in range(4):
                        part[h["ctok"](tb, n, t, r), h["crow"](rt, g, r)] = acc[warp, n, :, r]
            assert not torch.isnan(part).any(), "partial tile not fully written"
            parts.append(part)
        share, groups = mt // cs, rows // 4
        seen = torch.zeros(mt, rows, dtype=torch.int64)
        for rank in range(cs):
            for idx in range(share * groups):
                tok, row = h["etok"](rank, share, idx, groups), h["erow"](idx, groups)
                seen[tok, row:row + 4] += 1
                s = sum(parts[pr][tok, row:row + 4] for pr in range(cs))
                if tok < m:
                    out[tok, cl * rows + row:cl * rows + row + 4] = s
        assert torch.equal(seen, torch.ones_like(seen)), "outputs not reduced exactly once"
    assert int(a_reads.min()) == 1 and int(a_reads.max()) == 1, "weight fragments not read exactly once per tile"
    return out


def main():
    h = helpers()
    torch.manual_seed(6)
    w = torch.randn(256, 8192) / math.sqrt(8192)
    w[3, 7] = 1e-30
    w[200, 8191] = -2.5
    wp = rg_pack(w)
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    import p9_layout   # noqa: E402  (SGLang's split, as P9 uses it)
    shi, slo = p9_layout.p9_split_router(w) if hasattr(p9_layout, "p9_split_router") else (hi, lo)
    assert torch.equal(hi, shi) and torch.equal(lo, slo), "rg_pack's hi/lo differ from SGLang's split"
    wd = hi.double() + lo.double()
    configs = {32: [(1, 1, 8, 3, 1), (2, 2, 16, 4, 2)], 128: [(1, 4, 16, 3, 2), (2, 8, 8, 3, 1)],
               256: [(2, 16, 8, 3, 1)]}
    for m, insts in configs.items():
        g_ = torch.Generator().manual_seed(m)
        mm = m - 3 if m > 32 else m
        xn = torch.randn(mm, 8192, generator=g_).to(torch.bfloat16)
        xp = torch.zeros(mm, 8192, dtype=torch.bfloat16)
        xp[0, 7] = 4.0
        xp[mm - 1, 8191] = -0.25
        for inst in insts:
            for x, exact in ((xn, False), (xp, True)):
                got = emulate(inst, wp, x, h)
                ref = x.double() @ wd.T
                if exact:
                    assert torch.equal(got, ref), (m, inst, "probe not exact")
                else:
                    sc = x.double().abs() @ wd.abs().T
                    assert float(((got - ref).abs() / sc).max()) < 1e-12, (m, inst)
        print(f"ok: RG m{m} ({mm} tokens) emulation exact for {len(insts)} configs (probe rows bit exact)")
    print("ok: rg_pack's hi/lo equal SGLang's split; x stages, fragment reads and reduction cover everything once")


if __name__ == "__main__":
    sys.path.insert(0, str(LOCAL.parent / "p9"))
    main()

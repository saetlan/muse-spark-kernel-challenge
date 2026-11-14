"""Exactness of the DG12 weight format (proj/dg/dg12_layout.py) with the 7-op decode parsed from the CUDA text
(check_f12m's helpers): every unit decodes to 8 BF16 in k order, scaled by 2^(112 - base); escapes decode as the
placeholder 2^-15 and their list rebuilds the true value; scale x placeholder-corrected value = W bit for bit, on all
65,536 BF16 patterns and Gaussian weights at the real shapes, for every block size's escape lists.

Run: .venv/bin/python proj/local/check_dg12.py
"""

import math
import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))
sys.path.insert(0, str(LOCAL.parent / "f12v"))
sys.path.insert(0, str(LOCAL.parent / "dg"))
import check_f12m  # noqa: E402
from dg12_layout import dg12_pack, dg12_unpack_reference  # noqa: E402


def rebuild(pack, n, k, dec, bn):
    scaled = dg12_unpack_reference(pack, n, k, dec)                        # placeholder at escapes
    offs, ent = pack["esc"][bn]
    w = scaled.clone()
    for b in range(n // bn):
        for e in ent[int(offs[b]):int(offs[b + 1])].tolist():
            nl, kk, fb, _ = e
            row = b * bn + nl
            assert scaled[row, kk] == 2.0 ** -15, (row, kk, float(scaled[row, kk]))
            w[row, kk] = torch.tensor([fb], dtype=torch.int32).view(torch.float32).double()
    return w * pack["scale"].double().view(-1, 1)


def main():
    dec = check_f12m.helpers()["decode"]
    allbits = torch.arange(65536, dtype=torch.int64).to(torch.int16)
    perm = torch.randperm(65536, generator=torch.Generator().manual_seed(0))
    w = allbits[perm].view(torch.bfloat16).view(64, 1024)
    finite = torch.isfinite(w.float())
    w = torch.where(finite, w, torch.zeros_like(w))                       # NaN/Inf are not weights
    pack = dg12_pack(w, bn_list=(32, 64))
    if pack is None:
        print("ok: all-patterns case refused (an escape not exactly scalable), as designed")
    else:
        for bn in (32, 64):
            got = rebuild(pack, 64, 1024, dec, bn)
            assert torch.equal(got, w.double()), bn
        print(f"ok: all finite BF16 patterns rebuild bit exact (escapes {pack['n_esc']})")
    torch.manual_seed(1)
    for name, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "unpooler": (8192, 4096), "pooler": (4096, 8192)}.items():
        w = (torch.randn(n, k) / math.sqrt(k)).to(torch.bfloat16)
        w[0, :5] = 0
        w[1, 3] = 1e4
        w[2, 77] = 1e-30
        pack = dg12_pack(w)
        assert pack is not None, name
        for bn in sorted(pack["esc"]):
            if bn not in (32, 48, 64, 128):
                continue
            got = rebuild(pack, n, k, dec, bn)
            assert torch.equal(got, w.double()), (name, bn)
        print(f"ok: {name:8s} rebuild bit exact for BN {sorted(b for b in pack['esc'] if b in (32, 48, 64, 128))}; "
              f"escapes {pack['n_esc']} ({pack['n_esc'] / (n * k) * 100:.4f}%), max per 64-row block "
              f"{int((pack['esc'][64][0][1:] - pack['esc'][64][0][:-1]).max())}")


if __name__ == "__main__" and "--kernel" not in sys.argv:
    main()


# ---------------------------------------------------------------------------------------------------- kernel emulation
TEXT = [
    "constexpr uint32_t kRowsPerPass = kNumDecodeWarps * 4;",
    "constexpr uint32_t kPasses = (BLOCK_N + kRowsPerPass - 1) / kRowsPerPass;", "const uint32_t u = dg12_dunit(lane_idx);",
    "for (uint32_t p0 = 0; p0 < kPasses and (kDiag & 1) == 0; p0 += kBatch) {   // DIAG 1: no decode",
    "const uint32_t r = dg12_drow(dw * 4 + (p0 + i) * kRowsPerPass, lane_idx);", "if (p0 + i < kPasses and r < BLOCK_N) {",
    "qv[i] = dg12_lds32(q_base + dg12_qoff(r, u));", "dg12_lds64(l_base + dg12_loff(r, u), l0v[i], l1v[i]);",
    "dg12_sts128(b_base + dg12_boff(r, u), dg12_decode(qv[i], l0v[i], l1v[i]));",
    "const uint32_t q_base = dg12_smem_u32(smem_q[stage_idx]);", "const uint32_t b_base = dg12_smem_u32(smem_b[stage_idx]);",
    "const uint32_t l_base = dg12_smem_u32(smem_l[stage_idx]);",
    "const uint32_t dw = tma_warp == 3 ? 2 : tma_warp;", "full_barriers[i]->init(1 + kNumDecodeWarps);",
    "tma::copy<128, BLOCK_N / 4, 0, uint8_t>(&tensor_map_q, &raw_barrier, smem_q[stage_idx], 0,",
    "(k_block_idx * SHAPE_N + n_idx) / 4);", "(k_block_idx * SHAPE_N + n_idx) / 2);",
    "tma::copy<128, BLOCK_N / 2, 0, uint8_t>(&tensor_map_l, &raw_barrier, smem_l[stage_idx], 0,",
    "cutlass::arch::fence_view_async_shared();   // DIAG 4: no proxy fence (measures its cost)",
    "const uint32_t gr = lane_idx >> 2, tq = lane_idx & 3, wq = warp_idx % 4;",
    "constexpr uint32_t kMaxEsc = (BLOCK_N / 8) * kNumWaves >= 16 ? 6 : 12;", "constexpr uint32_t kRows = 2 * kNumWaves;",
    "e_lo = __ldg(esc_offs + n_block_idx * 4 + tq);", "e_hi = __ldg(esc_offs + n_block_idx * 4 + tq + 1);",
    "ent[i] = static_cast<int>(i) < e_hi - e_lo ? __ldg(esc_ent + e_lo + i) : make_int4(0, 0, 0, 0);",
    "const uint32_t row = dg12_row(WAVE_BLOCK_M, rr >> 1, math_wg_idx, WGMMA::M, wq, gr, rr & 1);",
    "const uint32_t mg = m_block_idx * BLOCK_M + row;",
    "xe[i][rr] = (static_cast<int>(i) < e_hi - e_lo and row < BLOCK_M and mg < shape_m)",
    "? __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(ent[i].y)]) : 0.0f;",
    "const uint32_t nl = static_cast<uint32_t>(ent[i].x), c = nl & 1u;", "if ((nl >> 3) == j) {",
    "float& a0 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)];",
    "float& a1 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)];",
    "a0 = fmaf(-xe[i][rr], 3.0517578125e-05f, a0);", "a0 = fmaf(xe[i][rr], val, a0);",
    "a1 = fmaf(-xe[i][rr], 3.0517578125e-05f, a1);", "a1 = fmaf(xe[i][rr], val, a1);",
    "for (int e = e_lo + static_cast<int>(kMaxEsc); e < e_hi; ++ e) {",
    "const uint32_t nl = static_cast<uint32_t>(en.x), c = nl & 1u;", "if (row < BLOCK_M and mg < shape_m) {",
    "const float xk = __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(en.y)]);",
    "a0 = fmaf(-xk, 3.0517578125e-05f, a0);", "a0 = fmaf(xk, val, a0);",
    "sc[j] = __ldg(reinterpret_cast<const float2*>(col_scale + n_block_idx * BLOCK_N + 8 * j + 2 * tq));",
    "accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)] *= sc[j].x;",
    "accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)] *= sc[j].y;",
    "if (e == 0) e = make_2d(&tq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(q), 128, (K / 64) * N / 4, 128, BN / 4, 128, 0);",
    "if (e == 0) e = make_2d(&tl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(l), 128, (K / 64) * N / 2, 128, BN / 2, 128, 0);",
]


def dg12_helpers():
    import check_f12v as cv
    sys.path.insert(0, str(LOCAL.parent / "dg"))
    import dg12
    cv.SRC = dg12.DG12_CU
    for stmt in TEXT:
        assert cv.SRC.count(stmt) >= 1, f"kernel text changed, update the emulation: {stmt}"
    return {n: cv.parse("dg12_" + n) for n in ("decode", "qoff", "loff", "boff", "drow", "dunit", "acc", "row")}


def sw128(off):
    """CUTLASS Swizzle<3,4,3> on a byte offset within a 1024-B aligned K-major tile (what TMA SWIZZLE_128B writes)."""
    return off ^ (((off >> 7) & 7) << 4)


def emulate_block(h, pack, x, n_block, m_block, bm, bn, k, nmath):
    """FP64 out[m rows of this block, bn cols] per the kernel: decoders build each stage's B tile from the raw planes
    (parsed offsets) and it must equal the TMA SW128 image of the decoded tile; the math is x @ B^T over all stages
    (DeepGEMM's wgmma, trusted); the epilogue per (warpgroup, warp, lane): escape corrections then scale, through the
    parsed accumulator/row maps, checked to touch each (row, col) as the PTX m64nNk16 layout says."""
    m = x.shape[0]
    q = pack["q"].to(torch.int64) & 0xFFFFFFFF
    l = pack["l"].to(torch.int64) & 0xFFFFFFFF
    kb_n = k // 64
    wsc = torch.zeros(bn, k, dtype=torch.float64)                      # decoded scaled weights of this n-block
    for kb in range(kb_n):
        # the TMA box image of the chunk-major planes: Q rows of 128 B at outer (kb * N + n_idx) / 4, BN / 4 of them;
        # L rows of 128 B at outer (kb * N + n_idx) / 2, BN / 2 of them -> smem bytes; read back as [bn][8] words
        n_all = q.shape[0]
        n_idx = n_block * bn
        qc = (pack["qc"].to(torch.int64) & 0xFFFFFFFF).reshape(-1, 32)            # 128-B rows (32 words)
        lc = (pack["lc"].to(torch.int64) & 0xFFFFFFFF).reshape(-1, 32)
        q0, l0_ = (kb * n_all + n_idx) // 4, (kb * n_all + n_idx) // 2
        assert (kb * n_all + n_idx) % 4 == 0
        qst = qc[q0:q0 + bn // 4].reshape(bn, 8)                                 # smem row r at r * 32 B
        lst = lc[l0_:l0_ + bn // 2].reshape(bn, 8, 2)                            # smem row r at r * 64 B
        assert torch.equal(qst, q[n_block * bn:(n_block + 1) * bn, kb * 8:(kb + 1) * 8]), "Q box image"
        assert torch.equal(lst, l[n_block * bn:(n_block + 1) * bn, kb * 8:(kb + 1) * 8]), "L box image"
        qbytes = {}
        for r in range(bn):
            for u in range(8):
                qbytes[int(h["qoff"](r, u))] = (r, u)
        assert sorted(qbytes) == [4 * i for i in range(bn * 8)], "Q stage offsets not the TMA box image"
        tile = {}
        seen = set()
        n_pass = (bn + 11) // 12                                       # kPasses
        for dw in range(3):
            for pp in range(n_pass):
                r0 = dw * 4 + pp * 12
                for lane in range(32):
                    r, u = int(h["drow"](r0, lane)), int(h["dunit"](lane))
                    if r >= bn:
                        continue
                    assert (r, u) not in seen, ("unit decoded twice", r, u)
                    seen.add((r, u))
                    assert int(h["qoff"](r, u)) == r * 32 + u * 4 and int(h["loff"](r, u)) == r * 64 + u * 8
                    regs = h["decode"](qst[r, u], lst[r, u, 0], lst[r, u, 1])
                    tile[int(h["boff"](r, u))] = (r, u, [int(v) & 0xFFFFFFFF for v in regs])
        assert len(seen) == bn * 8, ("units not covered", len(seen))
        for off, (r, u, regs) in tile.items():
            assert off == sw128(r * 128 + u * 16), ("B unit not where TMA SW128 puts it", r, u, off)
            for e in range(8):
                b = (regs[e >> 1] >> (16 * (e & 1))) & 0xFFFF
                wsc[r, kb * 64 + u * 8 + e] = torch.tensor([b << 16], dtype=torch.int64).to(torch.int32).view(torch.float32).double()
    rows_m = torch.arange(m_block * bm, min(m, (m_block + 1) * bm))
    acc_full = torch.zeros(max(bm, 64), bn, dtype=torch.float64)
    acc_full[:len(rows_m)] = x[rows_m].double() @ wsc.T
    # epilogue through the thread maps
    wgm = 64
    wave = bm if bm <= wgm else 2 * wgm
    n_waves = bm // wave
    n_acc = bn // 2                                                      # WGMMA::kNumAccum for m64nBNk16 (64*bn/128)
    touched = {}
    applied = []
    out = torch.full((len(rows_m), bn), float("nan"), dtype=torch.float64)
    for wg in range(nmath // 128):
        for wq in range(4):
            for lane in range(32):
                gr, tq = lane >> 2, lane & 3
                acc = {}
                for li in range(n_waves):
                    for j in range(bn // 8):
                        for hh in range(2):
                            for c in range(2):
                                row = int(h["row"](wave, li, wg, wgm, wq, gr, hh))
                                col = 8 * j + 2 * tq + c
                                idx = int(h["acc"](n_acc, li, j, hh, c))
                                assert 0 <= idx < n_acc * n_waves and idx not in acc
                                acc[idx] = [row, col, float(acc_full[row, col]) if row < len(rows_m) else 0.0]
                offs_t, ent_t = pack["et"][bn]
                lo, hi = int(offs_t[n_block * 4 + tq]), int(offs_t[n_block * 4 + tq + 1])
                kmax = 6 if (bn // 8) * n_waves >= 16 else 12
                for idx_e, e in enumerate(ent_t[lo:hi].tolist()):
                    nl, kk, fb, _ = e
                    assert 0 <= nl < bn and ((nl & 7) >> 1) == tq, ("entry in another thread's lane class", nl, tq)
                    applied.append((n_block * bn + nl, kk))
                    c, j = nl & 1, nl >> 3
                    val = torch.tensor([fb], dtype=torch.int32).view(torch.float32).double().item()
                    for li in range(n_waves):
                        for hh in range(2):
                            row = int(h["row"](wave, li, wg, wgm, wq, gr, hh))
                            mg = m_block * bm + row
                            if row < bm and mg < m:            # fast path: x = 0 for invalid rows; slow path: skipped
                                a = acc[int(h["acc"](n_acc, li, j, hh, c))]
                                assert a[1] == nl, ("correction on another column", a[1], nl)
                                xk = float(x[mg, kk].double())
                                a[2] = a[2] - xk * 2.0 ** -15 + xk * val
                for j in range(bn // 8):
                    sc = pack["scale"][n_block * bn + 8 * j + 2 * tq: n_block * bn + 8 * j + 2 * tq + 2].double()
                    for li in range(n_waves):
                        for hh in range(2):
                            acc[int(h["acc"](n_acc, li, j, hh, 0))][2] *= float(sc[0])
                            acc[int(h["acc"](n_acc, li, j, hh, 1))][2] *= float(sc[1])
                for idx, (row, col, v) in acc.items():
                    if row < len(rows_m) and row < bm:
                        assert (row, col) not in touched, ("output element owned twice", row, col)
                        touched[(row, col)] = True
                        out[row, col] = v
    assert len(touched) == len(rows_m) * bn, ("outputs not covered", len(touched), len(rows_m) * bn)
    ent8 = pack["ent8"]
    want = sorted((int(r), int(kk)) for r, kk, _, _ in ent8.tolist() if n_block * bn <= r < (n_block + 1) * bn)
    n_wg_lanes = (nmath // 128) * 4 * 8                                   # threads sharing one lane class (per tq)
    assert sorted(set(applied)) == want and len(applied) == len(want) * n_wg_lanes, ("escapes not applied once per owner",)
    return rows_m, out


def check_kernel():
    h = dg12_helpers()
    torch.manual_seed(5)
    cases = [("qkv_gate", 2304, 8192, 128, 64, 48, 128), ("pooler", 4096, 8192, 32, 32, 32, 128),
             ("o", 8192, 1024, 256, 128, 128, 256), ("unpooler", 8192, 4096, 128, 64, 128, 128)]
    for name, n, k, m, bm, bn, nmath in cases:
        w = (torch.randn(n, k) / math.sqrt(k)).to(torch.bfloat16)
        w[0, :5] = 0
        w[1, 3] = 1e4
        w[bn + 2, 77] = 1e-30
        w[bn + 9, k - 1] = -3e3
        w[16:24, 100:120] = 0                       # 40 escapes per lane class in block 0: exercises the overflow path
        pack = dg12_pack(w)
        assert pack is not None
        wd = w.double()
        gen = torch.Generator().manual_seed(m)
        mm = m - 5
        x_norm = torch.randn(mm, k, generator=gen).to(torch.bfloat16)
        x_probe = torch.zeros(mm, k, dtype=torch.bfloat16)
        x_probe[0, 3] = 0.5
        x_probe[1, 77] = -32.0
        x_probe[2, k - 1] = 2.0
        x_probe[mm - 1, 0] = 4.0
        x_probe[3, 105] = 8.0                       # a probe on an escape column of the overflow block
        for n_block in (0, 1):
            m_block = 0
            for xx, exact in ((x_norm, False), (x_probe, True)):
                rows, got = emulate_block(h, pack, xx, n_block, m_block, bm, bn, k, nmath)
                ref = xx[rows].double() @ wd[n_block * bn:(n_block + 1) * bn].T
                if exact:
                    assert torch.equal(got, ref), (name, n_block, "probe not exact")
                else:
                    scale = xx[rows].double().abs() @ wd[n_block * bn:(n_block + 1) * bn].abs().T
                    err = float(((got - ref).abs() / scale.clamp(min=1e-30)).max())
                    assert err < 1e-12, (name, n_block, err)
        print(f"ok: {name:8s} m{m} BM{bm} BN{bn} ({nmath} math threads): decoders cover every unit once at TMA SW128 "
              f"positions; epilogue owns every output once; exact vs FP64 (probe rows bit exact), n-blocks 0-1")


if __name__ == "__main__" and "--kernel" in sys.argv:
    check_kernel()

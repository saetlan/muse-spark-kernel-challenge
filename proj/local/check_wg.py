"""Checks of WG (proj/wg/wg.py, wgmma GEMM with in-cluster split-K) that can run without a GPU, on the
CUDA text (index helpers parsed and executed with check_f12v's translator):
  1. TMA box writes with 128B swizzle (16-B chunk c of row r stored at chunk c ^ (r % 8), CUDA
     programming guide, TMA swizzle) and wgmma reads through the K-major 128B-swizzle descriptor
     (PTX ISA 9.4 sec. 9.7.17.5.1.2: ((8,m),(T,2k)):((8T,SBO),(1,T)) o Swizzle<3,4,3>, SBO = 1024 B,
     start + wg_aoff / wg_boff) address the same bytes; emulating the stage (A 128 x 64, B 128 x 64) and
     the 4 k16 MMAs per warpgroup gives A @ B.T exactly (FP64), for every k16 step and both warpgroups;
  2. the accumulator map (wg_row, wg_tok) puts the 2 x 128 x 64 values of a CTA on the 128 x 128 tile
     exactly once; the descriptor fields (start >> 4, LBO 1, SBO 64, swizzle 1) decode as intended;
  3. wg_c0 splits every chunk count among KS ranks exactly once; reduction shares cover the 128 rows;
  4. the generated wgmma asm has 64 "+f" accumulators and the right operand numbers;
  5. the mbarrier protocol (full: 1 arrival + tx, empty: 256 arrivals, parities as in the kernel)
     never deadlocks and never lets a stage be overwritten before both warpgroups released it, for
     every chunk count 1..40 and ST 2..4 (a step-by-step simulation of the kernel's loops).

Run: .venv/bin/python proj/local/check_wg.py
"""

import re
import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))
sys.path.insert(0, str(LOCAL.parent / "wg"))
import check_f12v as cv  # noqa: E402
import wg  # noqa: E402

TEXT = [
    "const int tile = blockIdx.x / KS;", "const int rowtile = tile / TT, tt = tile % TT;",
    "const int c0 = wg_c0(static_cast<int>(rank), chunks, KS), c1 = wg_c0(static_cast<int>(rank) + 1, chunks, KS);",
    "mbar_init(&full[s], 1u);", "mbar_init(&empty[s], 256u);",
    # each kernel's producer parity, tied to its own stage size so one kernel's copy can't hide the other's
    "mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);\n        mbar_expect_tx(&full[s], static_cast<uint32_t>(STAGE_BYTES));",
    "mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);\n        mbar_expect_tx(&full[s], static_cast<uint32_t>(XB));",
    "tma_load_2d(smem + s * STAGE_BYTES, &tw, &full[s], 64 * c, 128 * rowtile);",
    "tma_load_2d(smem + s * STAGE_BYTES + A_BYTES, &tx, &full[s], 64 * c, 128 * tt);",
    "mbar_wait(&full[s], static_cast<uint32_t>((i / ST) & 1), failed);",
    "for (int j = 0; j < 4; ++j) wgmma_128(d, wg_desc(a0 + wg_aoff(w, j)), wg_desc(b0 + wg_boff(j)), 1u);",
    "wgmma_commit();", "wgmma_wait<1>();", "if (i > 0) mbar_arrive(&empty[(i - 1) % ST]);",
    "const int lane = tid & 31, q = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;",
    "constexpr int STAGE_BYTES = 32768;", "constexpr int A_BYTES = 16384;",
    "for (int e = 0; e < 64; ++e) red[wg_tok(t, e) * 128 + wg_row(wgi - 1, q, g, e)] = d[e];",
    "const int tk = wg_etok(idx, GROUPS), row = wg_erow(idx, GROUPS, static_cast<int>(rank), SHARE);",
    "const uint32_t a = red_addr + 4u * static_cast<uint32_t>(tk * 128 + row);",
    "lo[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row);",
    "hi[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row + 4);",
    "hi[p] = ld_peer_v4(a + 16u, static_cast<uint32_t>(p));",
    "for (int i = 0; i < 8; ++i) v[i] *= __uint_as_float((bases[row + i] + 15u) << 23);",
    "*reinterpret_cast<uint4*>(out + tok * n_rows + row0 + row) =",
    "for (int idx = tid; idx < GROUPS * 128; idx += 384) {",
    "wg_epilogue<KS, false>(d, smem, nullptr, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);",
    "wg_epilogue<KS, true>(d, smem, bases, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);",
]


WGF_TEXT = [
    "if (wgi == 0) bases[tid] = rec[2LL * (128LL * rowtile + tid)].x & 0xFFu;",
    "tma_load_2d(smem + s * XB, &tx, &full[s], 64 * c, 128 * tt);",
    "const long long mytile = wgf_tile(rowtile, w, q);",
    "const long long row = 16 * mytile + g + 8 * h;",
    "for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c0 + sl, j, lane), wv[sl][j]);",
    "for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c + D, j, lane), wv[sl][j]);",
    "a[j] = wgf_decode(wword(wv[sl][0], j), wword(lv, 2 * (j & 1)), wword(lv, 2 * (j & 1) + 1));",
    "const uint4& lv = wv[sl][1 + (j >> 1)];",
    "for (int j = 0; j < 4; ++j) wgmma_rs_128(d, a[j], wg_desc(b0 + wg_boff(j)), 1u);",
    "const int kk = static_cast<int>(esc[j] & 0xFFFFu) - 64 * c;",
    "const float wt = ldexpf(__uint_as_float(esc[j] & 0xFFFF0000u), 112 - static_cast<int>(base));",
    "if (((e >> 1) & 1) == h) {",
    "const float xk = __uint_as_float(static_cast<uint32_t>(xs[wgf_xoff(wg_tok(t, e), kk) >> 1]) << 16);",
    "d[e] = fmaf(-xk, 3.0517578125e-05f, d[e]);", "d[e] = fmaf(xk, wt, d[e]);",
]


def helpers():
    cv.SRC = wg.wg_source(wg.WG_CHECK_INSTANCES, wg.WGF_CHECK_INSTANCES)
    for stmt in TEXT + WGF_TEXT:
        assert cv.SRC.count(stmt) >= 1, f"kernel text changed, update check_wg.py: {stmt}"
    h = {n: cv.parse("wg_" + n) for n in ("c0", "row", "tok", "aoff", "boff", "etok", "erow")}
    h.update({n: cv.parse("wgf_" + n) for n in ("widx", "tile", "xoff", "decode")})
    return h


def swz(addr):
    """Swizzle<3,4,3> on byte addresses: bits 4-6 ^= bits 7-9 (128B mode)."""
    return addr ^ (((addr >> 7) & 7) << 4)


def tma_write(img, base, tile):
    """TMA 2D box (rows x 64 BF16, 128-B rows) into a 1024-B aligned smem image with 128B swizzle."""
    rows = tile.shape[0]
    for r in range(rows):
        for ch in range(8):
            dst = base + r * 128 + ((ch ^ (r % 8)) * 16)
            img[dst // 2:dst // 2 + 8] = tile[r, 8 * ch:8 * ch + 8]


def wgmma_read(img, start, rows):
    """The rows x 16 K-major operand a descriptor (start, SBO 1024, 128B swizzle) selects."""
    out = torch.empty(rows, 16, dtype=img.dtype)
    for r in range(rows):
        for kk in range(16):
            a = swz(start + (r % 8) * 128 + (r // 8) * 1024 + 2 * kk)
            out[r, kk] = img[a // 2]
    return out


def check_tile(h):
    g = torch.Generator().manual_seed(2)
    a = torch.randn(128, 64, generator=g, dtype=torch.float64)      # weight rows x k
    b = torch.randn(128, 64, generator=g, dtype=torch.float64)      # tokens x k
    img = torch.full((32768 // 2,), float("nan"), dtype=torch.float64)
    tma_write(img, 0, a)
    tma_write(img, 16384, b)
    assert not torch.isnan(img).any(), "stage not fully written"
    for w in range(2):
        acc = torch.zeros(64, 128, dtype=torch.float64)
        for j in range(4):
            aw = wgmma_read(img, 0 + h["aoff"](w, j), 64)
            bw = wgmma_read(img, 16384 + h["boff"](j), 128)
            acc += aw @ bw.T
        want = a[64 * w:64 * w + 64] @ b.T
        assert torch.allclose(acc, want, rtol=0, atol=1e-9), (w, float((acc - want).abs().max()))
    print("ok: TMA 128B-swizzle writes and the K-major descriptor reads agree; the 4 k16 MMAs of each warpgroup give A @ B.T")


def check_maps(h):
    seen = torch.zeros(128, 128, dtype=torch.int64)
    for w in range(2):
        for q in range(4):
            for lane in range(32):
                gg, t = lane >> 2, lane & 3
                for e in range(64):
                    seen[h["row"](w, q, gg, e), h["tok"](t, e)] += 1
    assert torch.equal(seen, torch.ones_like(seen)), "accumulator map does not cover the tile exactly once"
    for ks in (1, 2, 4, 8):
        for chunks in range(ks, 200):
            got = []
            for r in range(ks):
                got += list(range(h["c0"](r, chunks, ks), h["c0"](r + 1, chunks, ks)))
            assert got == list(range(chunks)), (ks, chunks)
        share = 128 // ks
        groups = share // 8
        cover = torch.zeros(128, 128, dtype=torch.int64)
        for r in range(ks):
            for idx in range(groups * 128):
                tk, row = h["etok"](idx, groups), h["erow"](idx, groups, r, share)
                assert row % 8 == 0 and r * share <= row < (r + 1) * share
                cover[tk, row:row + 8] += 1
        assert torch.equal(cover, torch.ones_like(cover)), ("epilogue items", ks)
    # descriptor fields, from the CUDA text's constants
    m = re.search(r"wg_desc\(uint32_t saddr\) \{\s*return (.*?);\s*\}", cv.SRC, re.S)
    assert m, "wg_desc not found"
    expr = " ".join(cv.c_expr(m.group(1)).replace("static_cast<uint64_t>", "").split())
    for saddr in (0, 1024, 16384 + 96, 131072 + 8192 + 32):
        d = eval(expr, {"saddr": saddr})
        assert d & 0x3FFF == (saddr & 0x3FFFF) >> 4 and (d >> 16) & 0x3FFF == 1 and (d >> 32) & 0x3FFF == 64
        assert (d >> 49) & 7 == 0 and (d >> 62) == 1, hex(d)
    print("ok: accumulator map covers the 128 x 128 tile once; chunk split and reduction shares exact; descriptor fields as intended")


def check_asm():
    asm = wg.wgmma_asm(128)
    assert asm.count('"+f"(d[') == 64 and "m64n128k16.f32.bf16.bf16" in asm
    assert "{%0, %1," in asm and "%63}, %64, %65, p, 1, 1, 0, 0;" in asm and "setp.ne.b32 p, %66, 0;" in asm
    print("ok: wgmma asm has 64 accumulators, descriptors %64/%65, scale-d %66")


def simulate(chunks, st):
    """Round-robin the producer and two consumer warpgroups through the kernel's loops; mbarrier
    semantics: a phase completes when its arrivals (and tx) are in; try_wait.parity(p) succeeds once the
    phase with parity p completed (a fresh barrier counts its 'previous' phase, parity 1, as done)."""
    full = [{"phase": 0, "arr": 0} for _ in range(st)]
    empty = [{"phase": 0, "arr": 0} for _ in range(st)]
    owner = [None] * st          # chunk currently in a stage

    def done(b, parity):
        return ((b["phase"] - 1) & 1) == parity if b["phase"] > 0 else parity == 1

    def arrive(b, need):
        b["arr"] += 1
        if b["arr"] == need:
            b["arr"] = 0
            b["phase"] += 1
    prod = {"i": 0}
    cons = [{"i": 0, "pending": None} for _ in range(2)]
    for _ in range(100000):
        progressed = False
        i = prod["i"]
        if i < chunks:
            s = i % st
            if done(empty[s], ((i // st) & 1) ^ 1):
                assert owner[s] is None, f"stage {s} overwritten while chunk {owner[s]} is in use"
                owner[s] = i
                arrive(full[s], 1)          # expect_tx arrival; the tx bytes land with it
                prod["i"] += 1
                progressed = True
        for c in cons:
            i = c["i"]
            if i < chunks:
                s = i % st
                if done(full[s], (i // st) & 1):
                    assert owner[s] == i, (chunks, st, i, owner)
                    if i > 0:                 # wait_group(1): the previous chunk's MMAs are done
                        for _ in range(128):
                            arrive(empty[(i - 1) % st], 256)
                        c.setdefault("released", set()).add(i - 1)
                    c["i"] += 1
                    progressed = True
        for s in range(st):  # a stage is free again once both warpgroups released its chunk
            if owner[s] is not None and all(owner[s] in c.get("released", set()) for c in cons):
                owner[s] = None
        if all(c["i"] == chunks for c in cons) and prod["i"] == chunks:
            return
        assert progressed, f"deadlock at chunks {chunks}, ST {st}: producer {prod['i']}, consumers {[c['i'] for c in cons]}"
    raise AssertionError("simulation did not finish")


def check_wgf(h):
    """WGF on a CPU: F12M natural planes, the parsed decode as the RS A fragments (per warp the mma.sync
    layout), B through the descriptor from a TMA-swizzled stage, escapes from the stage via wgf_xoff, the
    per-row scale after the sum. Exact on probe inputs, fp64 rounding on normal ones."""
    sys.path.insert(0, str(LOCAL.parent / "f12m"))
    sys.path.insert(0, str(LOCAL.parent / "f12v"))
    sys.path.insert(0, str(LOCAL.parent / "p9"))
    from f12m_layout import f12m_device_planes
    asm = wg.wgmma_rs_asm(128)
    assert asm.count('"+f"(d[') == 64 and "{%64, %65, %66, %67}, %68, p, 1, 1, 0;" in asm and "setp.ne.b32 p, %69, 0;" in asm
    g_ = torch.Generator().manual_seed(8)
    n, k = 128, 512
    w = (torch.randn(n, k, generator=g_) / 22).to(torch.bfloat16)
    w[0, 3] = 0
    w[5, 70] = 1e4
    w[64 + 9, 70] = 1e-30
    w[127, 511] = -3e3
    pl = f12m_device_planes(w, natural=True)
    assert pl is not None
    p4 = pl["p"].to(torch.int64).view(-1, 4) & 0xFFFFFFFF
    rec = pl["rec"].to(torch.int64).view(n, 8) & 0xFFFFFFFF
    chs = k // 64
    reads = torch.cat([torch.tensor([h["widx"](h["tile"](0, w_, q), chs, c, j, lane)
                                     for w_ in range(2) for q in range(4) for c in range(chs) for lane in range(32)])
                       for j in range(3)])
    assert torch.equal(torch.sort(reads).values, torch.arange(n // 16 * chs * 3 * 32)), "WGF plane slots not read exactly once"
    for xk_kind in ("normal", "probe"):
        m = 100
        if xk_kind == "normal":
            x = torch.randn(m, k, generator=g_).to(torch.bfloat16)
        else:
            x = torch.zeros(m, k, dtype=torch.bfloat16)
            x[0, 70] = -32.0
            x[m - 1, 511] = 0.5
            x[7, 3] = 2.0
        acc = torch.zeros(n, 128, dtype=torch.float64)
        for c in range(chs):
            stage = torch.full((8192,), float("nan"), dtype=torch.float64)      # 16 KB of BF16 as values
            xt = torch.zeros(128, 64, dtype=torch.float64)
            xt[:m] = x[:, 64 * c:64 * c + 64].double()                         # TMA zero-fills tokens >= m
            tma_write(stage, 0, xt)
            for w_ in range(2):
                for q in range(4):
                    tile = h["tile"](0, w_, q)
                    sl = [p4[h["widx"](tile, chs, c, j, torch.arange(32))] for j in range(3)]
                    for j in range(4):
                        lv = sl[1 + (j >> 1)]
                        regs = [cv.u32(r) for r in h["decode"](sl[0][:, j], lv[:, 2 * (j & 1)], lv[:, 2 * (j & 1) + 1])]
                        a = torch.zeros(16, 16, dtype=torch.float64)
                        gl, tl = torch.arange(32) >> 2, torch.arange(32) & 3
                        for r in range(4):
                            for pp in range(2):
                                a[gl + 8 * (r & 1), 2 * tl + 8 * (r >> 1) + pp] = cv.bits_to_f64(((regs[r] >> (16 * pp)) & 0xFFFF) << 16)
                        b = wgmma_read(stage, h["boff"](j), 128)                # [128 tokens, 16]
                        acc[16 * tile:16 * tile + 16] += a @ b.T
            # escapes with k in this chunk, x from the stage (wgf_xoff), scaled domain
            for row in range(n):
                base, count = int(rec[row, 0]) & 0xFF, (int(rec[row, 0]) >> 8) & 0xFF
                for jj in range(7):
                    e = int(rec[row, 1 + jj])
                    kk = (e & 0xFFFF) - 64 * c
                    if jj < count and 0 <= kk < 64:
                        wt = cv.struct_f32(e & 0xFFFF0000) * 2.0 ** (112 - base)
                        for tok in range(128):
                            xv = float(stage[h["xoff"](tok, kk) // 2])
                            acc[row, tok] += -xv * 2.0 ** -15      # as the kernel: two separate FMAs
                            acc[row, tok] += xv * wt
        base = (rec[:, 0] & 0xFF).double()
        got = (acc * torch.pow(2.0, base - 112).view(-1, 1))[:, :m]
        ref = (w.double() @ x.double().T)
        if xk_kind == "probe":
            assert torch.equal(got, ref), "WGF probe not exact"
        else:
            sc = w.double().abs() @ x.double().abs().T
            assert float(((got - ref).abs() / sc).max()) < 1e-12
    print("ok: WGF emulation (natural 12-bit planes, RS A fragments, swizzled B, in-loop escapes, row scale) exact; "
          "plane slots read once; RS asm operands right")


def main():
    h = helpers()
    check_tile(h)
    check_maps(h)
    check_asm()
    check_wgf(h)
    for st in (2, 3, 4):
        for chunks in range(1, 41):
            simulate(chunks, st)
    print("ok: mbarrier protocol simulated for 1-40 chunks and ST 2-4: no deadlock, no stage reused early")


if __name__ == "__main__":
    main()

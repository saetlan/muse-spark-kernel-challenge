"""Local checks of M16 (RS-wgmma for the large batches): the natural-K weight layout through M13's decode,
the shared-memory B operand as wgmma reads it (PTX ISA 9.4 K-major 128B-swizzle canonical layout, matrix
descriptor), the accumulator mapping, and the kernel text these models stand for.

Run: .venv/bin/python moe/local/check_m16.py
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_m11  # noqa: E402
import check_m13  # noqa: E402
import check_v0_lab as base  # noqa: E402

K = 4096


def m13_dec_regs(word):
    """M13's decode of a relaid word (int64 tensor): the 4 A registers before the scale multiply."""
    return check_m13.new_dec(word)


def scale_bf16_pair(byte):
    """scale2<B>: bf16 bits (b << 4) + 0x5C00 in both lanes = e4m3(b) * 2^64 (normal, positive scales)."""
    v = byte.to(torch.uint8).view(torch.float8_e4m3fn).float() * 2.0**64
    b = v.to(torch.bfloat16)
    return torch.stack((b, b), -1)


def dequant_from_m16(mod, words_relaid, scales):
    """words (M13 bit layout) int32 [RT, KG, 32, 4], scales int32 [RT, KG, 8, 2] -> W [RT*16, KG*64] float32 as
    wgmma sees it (value * 2^-62): register r, element h of lane (g, t) at k16 step j = row g + 8 (r odd),
    K = 64 kg + 16 j + 2 t + 8 (r >= 2) + h (wgmma's A fragment = mma.m16n8k16's, per warp)."""
    rt, kg = words_relaid.shape[:2]
    w = words_relaid.to(torch.int64) & 0xFFFFFFFF
    sc = scales.to(torch.int64) & 0xFFFFFFFF  # [rt, kg, 8, 2]
    out = torch.zeros((rt * 16, kg * 64))
    lane = torch.arange(32)
    g, t = lane // 4, lane % 4
    for j in range(4):
        regs = m13_dec_regs(w[..., j])  # 4 x [rt, kg, lane]
        sx = (sc[:, :, :, 0] >> (8 * j)) & 0xFF  # row g, block j: [rt, kg, 8]
        sy = (sc[:, :, :, 1] >> (8 * j)) & 0xFF  # row g + 8
        sgx, sgy = scale_bf16_pair(sx[:, :, g]), scale_bf16_pair(sy[:, :, g])  # [rt, kg, lane, 2]
        for r in range(4):
            val = check_m11.hmul2(regs[r], sgy if r % 2 else sgx).float()  # [rt, kg, lane, 2]
            for h in range(2):
                rows = torch.arange(rt)[:, None, None] * 16 + g[None, None, :] + 8 * (r % 2)
                ks = torch.arange(kg)[None, :, None] * 64 + 16 * j + 2 * t[None, None, :] + 8 * (r >= 2) + h
                rows, ks = torch.broadcast_tensors(rows, ks)
                out[rows.reshape(-1), ks.reshape(-1)] = val[..., h].reshape(-1)
    return out


def check_layout(mod):
    from nvfp4 import dequantize, random_weights
    gen = torch.Generator().manual_seed(16)
    packed, sf, _ = random_weights(256, K, generator=gen, device="cpu")
    n = sf.numel()  # every normal positive E4M3 value (M13's scale2 needs those; m13_scales_ok gates M16 too)
    good = torch.tensor([b for b in range(1, 127) if (b & 0x78) != 0], dtype=torch.uint8)
    sf = good[torch.arange(n) % good.numel()].view(torch.float8_e4m3fn).view(sf.shape)
    words, scales = mod.m16_shuffle(packed, sf)
    assert words.shape == (16, 64, 32, 4) and words.dtype == torch.int32
    assert scales.shape == (16, 64, 8, 2) and scales.dtype == torch.int32
    got = dequant_from_m16(mod, mod.m13_relayout(words), scales)
    want = dequantize(packed, sf, 1.0) * 2.0**-62
    assert torch.equal(got, want), f"M16 layout mismatch, max diff {float((got - want).abs().max())}"
    # lossless: every code nibble and scale byte appears exactly once
    assert sorted(scales.view(torch.uint8).flatten().tolist()) == sorted(sf.view(torch.uint8).flatten().tolist())
    print(f"ok: m16_shuffle + M13 relayout/decode + scale2 per 16-K block = every weight of a 256 x {K} matrix "
          f"exactly as RS-wgmma's natural-K A fragments meet them ({good.numel()} scale values)")


def stage_store(acts, nb):
    """M14/M16's load_b: pair pr's 16-B segment `part` (K part*8..part*8+7 of the chunk) at byte
    pr * 128 + ((part ^ (pr & 7)) << 4) of a 1024-B-aligned stage. Returns {byte offset: bf16 element value}."""
    mem = {}
    for pr in range(nb):
        for part in range(8):
            for e in range(8):
                mem[pr * 128 + ((part ^ (pr & 7)) << 4) + 2 * e] = (pr, part * 8 + e)
    return mem


def wgmma_b_address(start, sbo, n, k):
    """PTX ISA 9.4 K-major 128B swizzle canonical layout ((8, m), (T, 2k)):((8T, SBO), (1, T)), T = 8 bf16:
    element (n, k) of a k16 slice at start + (n % 8) * 128 + (n // 8) * SBO + (k // 8) * 16 + (k % 8) * 2, then
    the 128B swizzle (Swizzle<3,4,3> on byte addresses: bits 4-6 ^= bits 7-9; pattern starts at 1024 B)."""
    a = start + (n % 8) * 128 + (n // 8) * sbo + (k // 8) * 16 + (k % 8) * 2
    return a ^ (((a >> 7) & 7) << 4)


def desc_fields(desc):
    return (desc & 0x3FFF) << 4, ((desc >> 16) & 0x3FFF) << 4, ((desc >> 32) & 0x3FFF) << 4, (desc >> 49) & 7, desc >> 62


def check_b_operand(mod):
    """The kernel's descriptor for (stage, step j) makes wgmma read pair n, K 16 j + k of the chunk."""
    src = mod.m13_source()
    for line in M16_DESC_LINES:
        assert line in src, f"M16 descriptor changed, update check_m16: {line}"
    for nb in (8, 16, 32, 64, 128):
        mem = stage_store(None, nb)
        for slot_base in (0, 1024 * 3, 1024 * 37):
            for j in range(4):
                desc = mod_desc(slot_base, j)
                start, lbo, sbo, base_off, sw = desc_fields(desc)
                assert sw == 1 and base_off == 0 and lbo == 16 and sbo == 1024 and start == slot_base + 32 * j
                for n in range(nb):
                    for k in range(16):
                        a = wgmma_b_address(start, sbo, n, k) - slot_base
                        assert 0 <= a < nb * 128 and mem[a] == (n, 16 * j + k), (nb, j, n, k, a, mem.get(a))
    print("ok: M16 B operand: the descriptor (start + 32 j, SBO 1024, 128B swizzle) reads pair n, K 16 j + k of "
          "load_b's swizzled stage for every n < NB, k < 16, j < 4, NB 8..128; every read inside the stage")


# The kernel's descriptor, mirrored: start >> 4 | LBO (1, unused for swizzled K-major) << 16 | SBO 1024 >> 4 << 32
# | 128B swizzle (1) << 62.
def mod_desc(stage_addr, j):
    return (((stage_addr + 32 * j) & 0x3FFFF) >> 4) | (1 << 16) | ((1024 >> 4) << 32) | (1 << 62)


M16_DESC_LINES = (
    "return (static_cast<uint64_t>(((addr + 32u * j) & 0x3FFFFu) >> 4)) | (1ull << 16) | (64ull << 32) | (1ull << 62);",
)


def check_accumulators():
    """CUTLASS CLayout_64xN (checked against the CUTLASS v3.9.2 source on Nov 4): thread (t0, g, w), value
    (v0, v1, n8): offset 128 t0 + g + 16 w + 64 v0 + 8 v1 + 512 n8 = m + 64 n -> row 16 w + g + 8 v1,
    column 8 n8 + 2 t0 + v0: register 4 n8 + 2 v1 + v0 = M14's acc[n8][2 v1 + v0] (mma.sync's C)."""
    for N in (8, 16, 32, 64, 128):
        for w in range(4):
            for lane in range(32):
                t0, g = lane % 4, lane // 4
                for v in range(N // 2):
                    v0, v1, n8 = v % 2, (v // 2) % 2, v // 4
                    off = 128 * t0 + g + 16 * w + 64 * v0 + 8 * v1 + 512 * n8
                    m, n = off % 64, off // 64
                    assert (m, n) == (16 * w + g + 8 * v1, 8 * n8 + 2 * t0 + v0)
                    assert v == 4 * n8 + 2 * v1 + v0
    print("ok: wgmma accumulators = M14's acc[n8][r] layout (rows 16 w + g (+8), columns 8 n8 + 2 t (+1))")


WAIT_BLOCK = (
    "      if constexpr (DB) {\n"
    "        wg_wait1();       // this warp's wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage are free\n"
    "        fence_regs<TT>(Ab[s & 1]);\n"
    "      } else {\n"
    "        wg_wait0();       // this warp's wgmmas of the previous chunk are done: {free}\n"
    "      }\n"
    "      fence_acc<TT, NT>(acc);\n"
    "      cp_wait<P - 1>();\n"
    "      fence_async_smem();  // this thread's copies of this chunk -> visible to wgmma (async proxy)\n"
    "      __syncthreads();    // this chunk is in shared memory for all; no warp still reads the previous chunk's stage\n")
M16_ORDER = (
    WAIT_BLOCK.replace("{free}", "its A registers and stage are free") +
    "      issue_b();          // chunk + P -> the stage chunk - 1 (DB: chunk - 2) used\n"
    "      compute(s, cntp, cslot);\n"
    "      if (more) load_w(s);",
    "  constexpr int SBP = P + 1 + (DB ? 1 : 0);",
    "    auto& A = Ab[DB ? (s & 1) : 0];",
    "      fence_regs<TT>(A);       // the decode is complete before wgmma.fence (it can't sink between fence and wgmma)\n"
    "      fence_acc<TT, NT>(acc);  // so is any write of acc (the flush's zeroing)\n"
    "      wg_fence();\n      wg_chunk<TT, NT>(nv, acc, A, sbase);\n      wg_commit();",
    'for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(acc[i][n][r])::"memory");',
    "      wg_wait0();  // acc complete\n      fence_acc<TT, NT>(acc);\n      flush(C, seg0);",
    "      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;\n      ls[i] = sf + (row * KG + c.grp * D) * 8 + g;",
    "      rw[s][i] = __ldcs(lw[i] + s * 32);\n      rs[s][i] = __ldcs(ls[i] + s * 8);",
    "const uint32_t sg = scale2r(rs[s][i].x, j), sg8 = scale2r(rs[s][i].y, j);",
    "return __byte_perm(v, 0u, 0x4440u | b) * 0x00100010u + 0x5C005C00u;",
    "const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;",
    "const uint32_t sbase = sb0 + slot * STAGE;",
    "const int nv = (cntp + 7) >> 3;",
    "  for (int j = 0; j < 4; ++j) {\n    const uint64_t desc = b_desc(sbase, j);\n#pragma unroll\n"
    "    for (int i = 0; i < TT; ++i) wg_n<N>(&acc[i][0][0], A[i][j], desc);",
    'asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");',
    'for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(A[i][j][k])::"memory");',
    'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
    'asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");',
    "static_cast<size_t>(1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>())",
    "const int dyn = 1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>();",
    "const int64_t db = (pf >> 28) & 1;",
)


def m16_text(src):
    i0, i1 = src.index("namespace m16 {"), src.index("}  // namespace m16")
    return src[i0:i1]


def check_kernel_text(mod):
    src = mod.m13_source()
    m16 = m16_text(src)
    for line in M16_ORDER:
        assert line in src, f"M16 kernel changed, update check_m16: {line[:90]}"
    # M14's verified scan / cursor / ring / flush lines are all still in M16 (it was generated from M14)
    rewritten = ("issue_b();        // chunk + P -> the stage the previous chunk used", "compute(s, rsc, cntp, cslot);",
                 "constexpr int SBP = P + 1;  // activation stages")  # M16's own versions are in M16_ORDER
    for line in check_m13.M14_RING_LINES + check_m13.M14_DECODE[:4]:
        if line not in rewritten:
            assert line in m16, f"M16 lost an M14 line: {line}"
    assert "ld_cg4" not in m16 and "mma(acc" not in m16
    # every wgmma wrapper: N/2 accumulators, then a0..a3, the descriptor, and the scale-d predicate register
    for N in range(8, 129, 8):
        nd = N // 2
        body = m16[m16.index(f"void wg_n{N}(float* d"):]
        body = body[:body.index("\n}\n")]
        assert f"wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 " in body, N
        dl = ", ".join(f"%{k}" for k in range(nd))
        al = ", ".join(f"%{nd + k}" for k in range(4))
        assert f"{{{dl}}}, {{{al}}}, %{nd + 4}, p, 1, 1, 0;" in body and f"setp.ne.b32 p, %{nd + 5}, 0;" in body, N
        assert body.count('"+f"(d[') == nd and f'"+f"(d[{nd - 1}])' in body and f'"+f"(d[{nd}])' not in body, N
        assert '"r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1)' in body, N
        k = N // 8
        assert f"case {k}: if constexpr (NT >= {k}) wg_steps<{N}>(acc, A, sbase); break;" in m16, N
        assert f"constexpr (N == {N}) wg_n{N}(d, a, desc);" in m16, N
    print("ok: M16 kernel text: wgmma wrappers N 8..128 (operands, scale-d = 1, K-major B), dispatch by 8-pair blocks, "
          "wait -> cp.async wait -> proxy fence -> barrier -> stage reuse, natural-K bank addressing, 1024-B stages, "
          "and M14's ring and decode lines")


M17_LINES = (
    WAIT_BLOCK.replace("{free}", "that stage is free") +
    "      issue_b();          // chunk + P -> the stage the previous chunk used\n"
    "      compute(s, wslot, cntp, cslot);",
    "  constexpr int SBP = P + 1 + (DB ? 1 : 0);",
    "    auto& A = Ab[DB ? (s & 1) : 0];",
    "      wg_wait0();  // acc complete\n      fence_acc<TT, NT>(acc);\n      flush(C, seg0);",
    "bulk_g2s(dst + D * 512, sf + (row * KG + c.grp * D) * 8, 256, bar);",
    "const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);",
    "const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + s * 64 + g * 8);",
    "const uint32_t sg = scale2r(sc.x, j), sg8 = scale2r(sc.y, j);",
    "const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;",
    "const uint32_t sbase = sb0 + bslot * STAGE;",
    "    fence_regs<TT>(A);       // as M16: decode and acc writes complete before wgmma.fence\n"
    "    fence_acc<TT, NT>(acc);\n    wg_fence();\n    wg_chunk<TT, NT>(nv, acc, A, sbase);\n    wg_commit();",
    "return 4 * R * TT * TILE_BYTES + 1024 + (P + 1 + (DB ? 1 : 0)) * 8 * NT * 128; }",
    "using m16::wg_mma;",
)


def check_m17_text(mod):
    """M17 = M15 + M16's wgmma: its own lines, and M15's verified ring / cost-split lines unchanged except the two
    scale lines (natural-K layout: TMA source index and the per-chunk slot read)."""
    src = mod.m13_source()
    m17 = src[src.index("namespace m17 {"):src.index("}  // namespace m17")]
    for line in M17_LINES:
        assert line in m17, f"M17 kernel changed, update check_m16: {line[:90]}"
    changed = ("bulk_g2s(dst + D * 512, sf + (row * GPT + c.grp) * 32, 256, bar);",
               "const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + lane * 8);")
    for line in check_m13.M15_LINES:
        if line not in changed:
            assert line in m17, f"M17 lost an M15 line: {line}"
    for line in changed:
        assert line not in m17, f"M17 kept M15's scale layout: {line}"
    assert "mma(acc" not in m17 and "ld_cg4" not in m17
    # the TMA scale copy covers exactly the 4 chunks x 8 x uint2 the slot reads (s * 64 + g * 8 + 8 <= 256)
    assert all(s_ * 64 + g * 8 + 8 <= 256 for s_ in range(4) for g in range(8))
    print("ok: M17 kernel text: M15's weight ring, activation ring and cost split with M16's wgmma, natural-K scales "
          "(TMA source = M16's [row][KG][8] uint2; slot read per chunk and row g), 1024-B stages")


def check_db_ring():
    """M16/M17's activation ring with async MMAs: chunk c's wgmmas read stage c % SBP and finish only when this warp
    waits (wait_group 1 at the top of chunk c + 2 with DB; wait_group 0 at the top of chunk c + 1 without). The
    producer writes chunk c + P after the barrier of chunk c: no stage may be rewritten while a warp's wgmma may still
    read it or before its chunk was computed. A buffer c & 1 (DB) is rewritten at chunk c + 2, after the same wait."""
    cases = 0
    for db in (0, 1):
        for P in range(2, 9):
            sbp = P + 1 + db
            for n in range(1, 60):  # chunks in the range
                stage_of = {}       # chunk -> stage it was staged into
                for c in range(min(P, n)):
                    stage_of[c] = c % sbp
                for c in range(n):
                    # pending (still reading) after this chunk's wait: chunk c - 1 with DB, none without
                    pending = {c - 1} if db and c >= 1 else set()
                    newc = c + P
                    if newc < n:
                        st = newc % sbp
                        live = pending | set(range(c, newc))  # pending reads, this chunk, staged ones not yet computed
                        for o in live:
                            if 0 <= o < n:
                                assert stage_of[o] != st, ("stage reused too early", db, P, c, o)
                        stage_of[newc] = st
                    assert stage_of[c] == c % sbp
                    if db and c >= 2:
                        assert (c & 1) != ((c - 1) & 1)  # the pending chunk's A buffer is the other one
                cases += 1
    print(f"ok: M16/M17 ring with async MMAs over {cases} (DB, P, length) cases: no stage is rewritten while a "
          "chunk's wgmma may read it (wait_group 1 + P + 2 stages, or wait_group 0 + P + 1), A buffers alternate")


def check_bank(mod):
    """m16_bank on a small fake expert set: words = m13_relayout(m16_shuffle words), scales as m16_shuffle."""
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(5)
    w = {}
    for name, rows in (("w13", 64), ("w2", 32)):
        packed, sf, _ = random_weights(rows, K, generator=gen, device="cpu", leading=(3,))
        w[name], w[name + "_sf"] = packed, sf
    bank = {}
    mod.m16_bank(w, bank, 1, 2)
    for name, rows in (("w13", 64), ("w2", 32)):
        assert bank[name + "n"].shape == (2, rows // 16, 64, 32, 4) and bank[name + "n_s"].shape == (2, rows // 16, 64, 8, 2)
        for e in range(2):
            words, scales = mod.m16_shuffle(w[name][1 + e], w[name + "_sf"][1 + e])
            assert torch.equal(bank[name + "n"][e], mod.m13_relayout(words)) and torch.equal(bank[name + "n_s"][e], scales)
    print("ok: m16_bank = per-expert m16_shuffle (+ M13 bit relayout) for experts [start, start + count)")


def main():
    check_m2.install_stubs()
    mod = base.load("submission_m13")
    check_layout(mod)
    check_accumulators()
    check_b_operand(mod)
    check_kernel_text(mod)
    check_m17_text(mod)
    check_db_ring()
    check_bank(mod)


if __name__ == "__main__":
    main()

"""Local checks of M18 (decode to shared memory + SS-wgmma for the large batches): the row-chunk weight layout through
M13's decode, the A-tile store addresses against the wgmma descriptor's reads, and the kernel text.

Run: .venv/bin/python moe/local/check_m18.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_m11  # noqa: E402
import check_m13  # noqa: E402
import check_m16  # noqa: E402
import check_v0_lab as base  # noqa: E402

K = 4096


def dequant_from_m18(words_relaid, scales):
    """words (M13 bit layout) int32 [RT, KG, 32, 4], scales int16 [RT, KG/4, 32, 4] -> W [RT*16, KG*64] float32
    (value * 2^-62) at the K each decoded element is stored to: lane l = 2 r + hf, word m, register R, element e
    -> row 16 rt + r, K = 64 kg + 8 (4 hf + m) + 2 R + e; scale = byte 2 (kg % 4) + (m >> 1) of the lane's group."""
    rt, kg = words_relaid.shape[:2]
    w = words_relaid.to(torch.int64) & 0xFFFFFFFF
    sb = scales.contiguous().view(torch.uint8).view(rt, kg // 4, 32, 8).to(torch.int64)
    out = torch.zeros((rt * 16, kg * 64))
    lane = torch.arange(32)
    r, hf = lane // 2, lane % 2
    for m in range(4):
        regs = check_m13.new_dec(w[..., m])  # 4 x [rt, kg, lane]
        s_idx = 2 * (torch.arange(kg) % 4)[:, None] + (m >> 1)  # [kg, 1]
        byte = torch.gather(sb[:, torch.arange(kg) // 4], 3, s_idx[None, :, None, :].expand(rt, kg, 32, 1))[..., 0]
        sp = check_m16.scale_bf16_pair(byte)  # [rt, kg, lane, 2]
        for R in range(4):
            val = check_m11.hmul2(regs[R], sp).float()
            for e in range(2):
                rows = torch.arange(rt)[:, None, None] * 16 + r[None, None, :]
                ks = torch.arange(kg)[None, :, None] * 64 + 8 * (4 * hf[None, None, :] + m) + 2 * R + e
                rows, ks = torch.broadcast_tensors(rows, ks)
                out[rows.reshape(-1), ks.reshape(-1)] = val[..., e].reshape(-1)
    return out


def check_layout(mod):
    from nvfp4 import dequantize, random_weights
    gen = torch.Generator().manual_seed(18)
    packed, sf, _ = random_weights(256, K, generator=gen, device="cpu")
    good = torch.tensor([b for b in range(1, 127) if (b & 0x78) != 0], dtype=torch.uint8)
    sf = good[torch.arange(sf.numel()) % good.numel()].view(torch.float8_e4m3fn).view(sf.shape)
    words, scales = mod.m18_shuffle(packed, sf)
    assert words.shape == (16, 64, 32, 4) and words.dtype == torch.int32
    assert scales.shape == (16, 16, 32, 4) and scales.dtype == torch.int16
    got = dequant_from_m18(mod.m13_relayout(words), scales)
    want = dequantize(packed, sf, 1.0) * 2.0**-62
    assert torch.equal(got, want), f"M18 layout mismatch, max diff {float((got - want).abs().max())}"
    assert sorted(scales.view(torch.uint8).flatten().tolist()) == sorted(sf.view(torch.uint8).flatten().tolist())
    print(f"ok: m18_shuffle + M13 relayout/decode = every weight of a 256 x {K} matrix at the (row, K) its decoded "
          f"bytes are stored to, natural K order within each 16-B segment, one scale per word ({good.numel()} values)")


def a_store_addr(w, lane, m):
    """The kernel's A-tile store: warp w's lane (r = lane / 2, hf = lane % 2), word m -> 16 B at row 16 w + r,
    segment 4 hf + m, 128B-swizzled (segment ^ (row & 7)), rows of 128 B in 8-row atoms of 1024 B."""
    row = 16 * w + lane // 2
    seg = 4 * (lane % 2) + m
    return (row % 8) * 128 + (row // 8) * 1024 + ((seg ^ (row % 8)) << 4), row, seg


def check_a_tile():
    """Every (row, K) of the 64 x 64 A tile is stored once, where the wgmma descriptor (start + 32 j, SBO 1024,
    128B swizzle) reads row m, K 16 j + k; stores of one STS.128 phase (8 lanes) hit 8 different 16-B bank groups."""
    held = {}
    for w in range(4):
        for lane in range(32):
            for m in range(4):
                a, row, seg = a_store_addr(w, lane, m)
                for b in range(16):
                    held[a + b] = (row, 8 * seg + b // 2)  # 2 bytes per bf16
    assert len(held) == 64 * 128
    for j in range(4):
        for row in range(64):
            for k in range(16):
                a = check_m16.wgmma_b_address(32 * j, 1024, row, k)
                assert held[a] == (row, 16 * j + k) and held[a + 1] == (row, 16 * j + k), (j, row, k)
    for w in range(4):
        for m in range(4):
            for ph in range(4):
                groups = {(a_store_addr(w, lane, m)[0] // 16) % 8 for lane in range(8 * ph, 8 * ph + 8)}
                assert len(groups) == 8, ("bank conflict", w, m, ph)
    print("ok: M18 A tile: 4 warps x 32 lanes x 4 words store all 64 x 64 elements once, exactly where the K-major "
          "128B-swizzle descriptor reads (row m, K 16 j + k); every STS.128 phase is conflict-free")


M18_LINES = (
    "  const int arow = 16 * warp + (lane >> 1);\n  const uint32_t aoff = (arow & 7) * 128 + (arow >> 3) * 1024;",
    "      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;\n      unsigned char* tile = s_a + (par * TT + i) * ATILE + aoff;",
    "        const uint32_t sg = scale2r(sv, 2 * (s & 1) + (mm >> 1));\n        const int seg = 4 * (lane & 1) + mm;",
    "        v.x = hmul2(rotr(q, 2) & PM, sg);\n        v.y = hmul2(rotr(q, 6) & PM, sg);\n"
    "        v.z = hmul2(rotr(q, 12) & PM, sg);\n        v.w = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg);\n"
    "        *reinterpret_cast<uint4*>(tile + ((seg ^ (arow & 7)) << 4)) = v;",
    "      wg_wait1();          // this warp's (the warpgroup's) wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage\n"
    "      fence_acc<TT, NT>(acc);\n"
    "      if constexpr (MODE == 1 || MODE == 3) {  // recon: no decode (keep the loads)\n",
    "      } else {\n        decode_store(s, rsc, s & 1);\n      }\n      if (more) {\n        load_w(s);\n"
    "        if (s == 0) load_s();\n      }\n      cp_wait<P - 1>();\n"
    "      fence_async_smem();  // this thread's A stores and stage copies -> visible to wgmma (async proxy)\n"
    "      if constexpr (MODE == 4) {\n",
    "      } else {\n        __syncthreads();   // all of chunk s's A tile and stage are written\n      }\n"
    "      issue_b();           // chunk + P -> the stage chunk - 2 used\n"
    "      if constexpr (MODE != 1 && MODE != 2) {\n"
    "        ss_chunk<TT, NT>((cntp + 7) >> 3, acc, b_desc(sa0 + ((s & 1) * TT) * ATILE, 0),\n"
    "                         b_desc(sa0 + ((s & 1) * TT + TT - 1) * ATILE, 0), b_desc(sb0 + cslot * STAGE, 0));",
    "template <int TT, int NT, bool GATED, int P, int MODE = 0>",
    "      wg_wait0();  // acc complete\n      fence_acc<TT, NT>(acc);\n      flush(C, seg0);",
    "  constexpr int SBP = P + 2;",
    "  const uint32_t sa0 = smem_addr(s_a);\n  const uint32_t sb0 = sa0 + 2 * TT * ATILE;",
    "unsigned char* s_a = s_raw + ((1024u - (smem_addr(s_raw) & 1023u)) & 1023u);",
    "constexpr int dyn_bytes() { return 1024 + 2 * TT * ATILE + (P + 2) * stage_bytes<NT>(); }",
    "constexpr int ATILE = 64 * 128;",
)


def check_kernel_text(mod):
    src = mod.m13_source()
    m18 = src[src.index("namespace m18 {"):src.index("}  // namespace m18")]
    for line in M18_LINES:
        assert line in m18, f"M18 kernel changed, update check_m18: {line[:90]}"
    for line in check_m13.M14_RING_LINES:  # the activation producer / consumer slots are M14's
        if "compute(" not in line and "SBP = P + 1" not in line and "issue_b();        //" not in line:
            assert line in m18, f"M18 lost an M14 line: {line}"
    # every SS wrapper: TT x N/2 accumulators, descriptors stepped by 2 (32 B) per k16, fence inside, commit last
    for TT, kmax in ((2, 8), (1, 16)):
        for k in range(1, kmax + 1):
            N, nd = 8 * k, 4 * k
            body = m18[m18.index(f"void ss{TT}_n{N}("):]
            body = body[:body.index("\n}\n")]
            lines = [l.strip().strip('"').replace("\\n", "") for l in body.splitlines() if l.strip().startswith('"')]
            wg = [l for l in lines if l.startswith("wgmma.mma_async")]
            assert len(wg) == 4 * TT, (TT, N, len(wg))
            d0 = "{" + ", ".join(f"%{x}" for x in range(nd)) + "}"
            d1 = "{" + ", ".join(f"%{nd + x}" for x in range(nd)) + "}"
            for j in range(4):
                assert wg[TT * j] == f"wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {d0}, ra{j}, rb{j}, p, 1, 1, 0, 0;", (TT, N, j)
                if TT == 2:
                    assert wg[2 * j + 1] == f"wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {d1}, ra{4 + j}, rb{j}, p, 1, 1, 0, 0;"
            ia = TT * nd
            want_pre = [f"mov.b64 rb0, %{ia + TT};", "add.s64 rb1, rb0, 2;", "add.s64 rb2, rb0, 4;", "add.s64 rb3, rb0, 6;",
                        f"mov.b64 ra0, %{ia};", "add.s64 ra1, ra0, 2;", "add.s64 ra2, ra0, 4;", "add.s64 ra3, ra0, 6;"]
            if TT == 2:
                want_pre += [f"mov.b64 ra4, %{ia + 1};", "add.s64 ra5, ra4, 2;", "add.s64 ra6, ra4, 4;", "add.s64 ra7, ra4, 6;"]
            for l in want_pre:
                assert l in lines, (TT, N, l)
            fi, ci = lines.index("wgmma.fence.sync.aligned;"), lines.index("wgmma.commit_group.sync.aligned;")
            first, last = lines.index(wg[0]), lines.index(wg[-1])
            assert max(lines.index(l) for l in want_pre) < fi < first and last < ci, (TT, N)
            assert f"setp.ne.b32 p, %{ia + TT + 1}, 0;" in lines and body.count('"+f"(d') == TT * nd
            case = (f"case {k}: if constexpr (NT >= {k}) {{ if constexpr (TT == 2) ss2_n{N}" if k <= 8
                    else f"case {k}: if constexpr (TT == 1 && NT >= {k}) ss1_n{N}(")
            assert case in m18, (k, N)
    print("ok: M18 kernel text: decode-store addressing and scales, chunk order (wait 1 -> decode -> refill -> copies -> "
          "proxy fence -> barrier -> next stage -> wgmmas), P + 2 stages, 1024-B A/B layout, 24 SS wrappers (descriptors "
          "+32 B per k16 computed before wgmma.fence, 4 x TT wgmmas, commit last), dispatch for TT 1 and 2")


def check_bank(mod):
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(6)
    w = {}
    for name, rows in (("w13", 64), ("w2", 32)):
        packed, sf, _ = random_weights(rows, K, generator=gen, device="cpu", leading=(3,))
        w[name], w[name + "_sf"] = packed, sf
    bank = {}
    mod.m18_bank(w, bank, 1, 2)
    for name, rows in (("w13", 64), ("w2", 32)):
        assert bank[name + "r"].shape == (2, rows // 16, 64, 32, 4) and bank[name + "r_s"].shape == (2, rows // 16, 16, 32, 4)
        for e in range(2):
            words, scales = mod.m18_shuffle(w[name][1 + e], w[name + "_sf"][1 + e])
            assert torch.equal(bank[name + "r"][e], mod.m13_relayout(words)) and torch.equal(bank[name + "r_s"][e], scales)
    print("ok: m18_bank = per-expert m18_shuffle (+ M13 bit relayout), M13's bank shapes (so M14's loads apply)")


def main():
    check_m2.install_stubs()
    mod = base.load("submission_m13")
    check_layout(mod)
    check_a_tile()
    check_kernel_text(mod)
    check_m16.check_db_ring()  # M18's ring: P + 2 stages, wait_group 1, A buffers by chunk parity
    check_bank(mod)


if __name__ == "__main__":
    main()

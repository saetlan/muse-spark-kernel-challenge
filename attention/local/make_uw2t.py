"""Generates attention/local/cuda_uw2t.py: UW2t = UW2 (cuda_uw2.py, one launch per sliding verify layer) with W3t's TMA
bulk-copy copier (make_w3t.py, B7 / AF66): copier warp 8, lane l issues 128 B `cp.async.bulk` copies of the K and V rows
of keys l and l + 32 into row-major staging (Ks, Vs; pitch 144), one `mbarrier.arrive.expect_tx` of 64 x 2 x 128 =
16,384 B on the land barrier (count 1); page entries as UW2. The transposers, after the land barrier (and fresh_bar for
the block's rows), patch the block's new rows into Ks / Vs (UW2's fresh-row patch, AF36, now into K staging instead of the
tile), re-lay K from Ks into the K-major tile (16 B moves, warp p: keys 32p .. 32p+31), transpose V as UW2, fence and
arrive on full. Consumers, prologue, push and finalize: UW2's. Shared memory +9 KB per stage: variants at 4 stages.
Barrier watchdog as W3d. Host: uw2t(... as uw2 ...), uw2t_setup(), uw2t_attrs(), uw2t_hang().
Run: .venv/bin/python attention/local/make_uw2t.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_uw2t.py")
VARIANTS = [(4, 1, 4), (4, 1, 8), (4, 1, 16), (4, 0, 16)]


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


BULK = r'''__device__ __forceinline__ void bulk128(void* dst, const void* src, uint64_t* bar) {   // UW2t: one 128 B TMA bulk copy
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
'''


def render():
    text = open(os.path.join(HERE, "cuda_uw2.py")).read()
    for pat, rep in ((r"\bokc_uw2", "okc_uw2t"), (r"\bOKC_UW2_", "OKC_UW2T_"), (r"\buw2\(", "uw2t("), (r"\buw2_setup\b", "uw2t_setup"),
                     (r"\buw2_attrs\b", "uw2t_attrs"), (r"\buw2_kernel\b", "uw2t_kernel"), (r"\bCUDA_UW2_", "CUDA_UW2T_"),
                     (r'"uw2: ', '"uw2t: '), (r'"uw2 ', '"uw2t ')):
        text = re.sub(pat, rep, text)
    text = sub1(text, "template <int STAGES, int PP, int CS>\n__global__", BULK + "template <int STAGES, int PP, int CS>\n__global__", "helpers")
    text = sub1(text, "  constexpr int SLOTB = OFF_VT + 2 * VTH;\n",
                "  constexpr int OFF_KS = OFF_VT + 2 * VTH;            // UW2t: K rows land row-major here (pitch 144)\n"
                "  constexpr int SLOTB = OFF_KS + 64 * VS_PITCH;\n", "slot layout")
    text = sub1(text, 'asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&land_bar[s])));   // copier lanes',
                'asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&land_bar[s])));   // UW2t: the copier\'s expect-tx arrive',
                "land count")
    old_copy = """      uint8_t* slot = sm + st * SLOTB;
#pragma unroll
      for (int it = 0; it < 16; ++it) {
        const int key = 8 * (it >> 1) + (lane & 7), ch = 4 * (it & 1) + (lane >> 3);
        const long row = __shfl_sync(FULL, key < 32 ? r0 : r1, key & 31);
        if (ABL != 2) {
          cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
               kc + row * row_bytes + poff + 16 * ch);
          cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
        }
      }
"""
    new_copy = """      uint8_t* slot = sm + st * SLOTB;
      {   // UW2t: lane l -> keys l and l + 32: their K and V rows as 128 B bulk copies into row-major staging
        if (lane == 0) bar_expect_tx(&land_bar[st], 64u * 2u * 128u);
        __syncwarp();
        bulk128(slot + OFF_KS + lane * VS_PITCH, kc + r0 * row_bytes + poff, &land_bar[st]);
        bulk128(slot + OFF_VS + lane * VS_PITCH, vc + r0 * row_bytes + poff, &land_bar[st]);
        bulk128(slot + OFF_KS + (32 + lane) * VS_PITCH, kc + r1 * row_bytes + poff, &land_bar[st]);
        bulk128(slot + OFF_VS + (32 + lane) * VS_PITCH, vc + r1 * row_bytes + poff, &land_bar[st]);
      }
"""
    text = sub1(text, old_copy, new_copy, "copier copies")
    text = sub1(text, """        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&land_bar[st])) : "memory");
        issue_pt(i + STAGES);""", """        issue_pt(i + STAGES);                          // UW2t: the land barrier completes on the bulk copies' bytes""", "copier arrive")
    # patch the block's rows into K staging (row-major) instead of the K tile
    text = sub1(text, """              *reinterpret_cast<uint4*>(slot + (c >> 2) * KC_HEAD + (r >> 3) * SBO_K + (c & 3) * LBO_K + (r & 7) * 16) =
                  *reinterpret_cast<const uint4*>(new_k + nl * 128 + 16 * c);""",
                """              *reinterpret_cast<uint4*>(slot + OFF_KS + r * VS_PITCH + 16 * c) =   // UW2t: into K staging
                  *reinterpret_cast<const uint4*>(new_k + nl * 128 + 16 * c);""", "patch target")
    # K re-lay after the patch, before the V transpose
    text = sub1(text, """      const uint8_t* vs = sm + st * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = sm + st * SLOTB + OFF_VT + hh * VTH;""", """      {   // UW2t: K rows of keys 32p .. 32p+31 (p = this transposer) from Ks into the K-major tile, 16 B moves
        uint8_t* slot = sm + st * SLOTB;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
          const int key = 32 * (warp - 9) + 4 * it + (lane >> 3), ch = lane & 7;
          *reinterpret_cast<uint4*>(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16) =
              *reinterpret_cast<const uint4*>(slot + OFF_KS + key * VS_PITCH + 16 * ch);
        }
      }
      const uint8_t* vs = sm + st * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = sm + st * SLOTB + OFF_VT + hh * VTH;""", "K re-lay")
    text = sub1(text, "  return static_cast<size_t>(stages) * (okc_uw2t::OFF_VT + 2 * 8 * 528) +",
                "  return static_cast<size_t>(stages) * (okc_uw2t::OFF_VT + 2 * 8 * 528 + 64 * okc_uw2t::VS_PITCH) +", "host smem")
    m = re.findall(r"#define OKC_UW2T_VARIANTS\(X\) [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "#define OKC_UW2T_VARIANTS(X) " + " ".join(f"X({s}, {p}, {c})" for s, p, c in VARIANTS) + "\n")
    for s, p, c in VARIANTS:
        smem = s * (2 * 4096 + 64 * 144 + 2 * 8 * 528 + 64 * 144) + 2 * s * 64 * 4 + 4 * 4096 + 128 * 68 * 4
        assert smem + 4096 <= 227 * 1024, (s, smem)
    bw = ('__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  while (!done)\n'
          '    asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n'
          '                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n')
    assert text.count(bw) == 1
    text = text.replace(bw, '__device__ unsigned int okc_uw2t_hang = 0;   // watchdog flag (make_uw2t.py)\n'
                        + bw.replace('  uint32_t done = 0;\n  while (!done)\n', '  uint32_t done = 0, spins = 0;\n  while (!done) {\n')
                            .replace(' : "memory");\n}\n', ' : "memory");\n'
                                     '    if (!done && ++spins > (1u << 20)) {\n      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_uw2t_hang), "r"(1u) : "memory");\n      break;\n    }\n  }\n}\n'))
    host = ('\nint64_t uw2t_hang() {   // lab: 1 when a UW2t barrier wait gave up (watchdog); reads and clears the flag\n'
            '  unsigned int h = 0, z = 0;\n'
            '  cudaMemcpyFromSymbol(&h, okc_uw2t::okc_uw2t_hang, sizeof(h));\n'
            '  cudaMemcpyToSymbol(okc_uw2t::okc_uw2t_hang, &z, sizeof(z));\n'
            '  return h;\n}\n')
    k = text.index('"""\n\nCUDA_UW2T_DECL = ')
    text = text[:k] + host + text[k:]
    text = text.replace("std::vector<int64_t> uw2t_attrs(int64_t stages, int64_t pp, int64_t cs);'",
                        "std::vector<int64_t> uw2t_attrs(int64_t stages, int64_t pp, int64_t cs);\\nint64_t uw2t_hang();'")
    assert "int64_t uw2t_hang();" in text
    text = text.replace("UW2_VARIANTS", "UW2T_VARIANTS")
    if "\nUW2T_VARIANTS = " not in text:
        text = text.rstrip("\n") + "\nUW2T_VARIANTS = " + repr(VARIANTS) + "\n"
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_uw2t.py from cuda_uw2.py (do not edit): UW2t, UW2 with W3t\'s TMA bulk-copy copier (see\n'
            'make_uw2t.py). Original docstring follows.\n' + text[3:i] + text[i:])


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_uw2t.py up to date" if same else "cuda_uw2t.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")

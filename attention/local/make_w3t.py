"""Generates attention/local/cuda_w3t.py: W3t = W3b (cuda_w3b.py) with a TMA bulk-copy copier (B7 / A58).
W3b's copy side holds full verify at ~2.3 TB/s (AF56 / AF58): its two copier warps issue 16 B cp.async pieces and wait on
their own commit groups. W3t: each copier lane issues one `cp.async.bulk` of 128 B per K row and per V row of its key
(copier c: keys 32c .. 32c+31), into row-major staging (Ks, Vs; pitch 144), completion counted on the stage's land
barrier by expect-tx (each copier warp: one arrive.expect_tx of 32 x 2 x 128 = 8192 bytes; land_bar counts 2 arrivals).
The copier then waits only on empty slots and on its page entries (cp4 ring S stages ahead, as W3b). Because a linear
128 B copy cannot write the K-major canonical tile (a key's 16 B dim chunks sit LBO = 128 B apart), the transposer warps,
after the land barrier, re-lay K from Ks into the tile (16 B moves, warp p: keys 32p .. 32p+31) and transpose V as W3b,
then fence generic -> async proxy and arrive on full. Consumers, partials and host interface: W3b's. Shared memory: +9 KB
per stage (Ks), stages 4 / 5 at 4 consumer warpgroups. Barrier watchdog as W3d.
Run: .venv/bin/python attention/local/make_w3t.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w3t.py")
VARIANTS = [(5, 0, 4, 104, 56), (4, 0, 4, 104, 56), (5, 1, 4, 104, 56)]


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


BULK = r'''__device__ __forceinline__ void bulk128(void* dst, const void* src, uint64_t* bar) {   // W3t: one 128 B TMA bulk copy
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint64_t* bar, uint32_t bytes) {   // one arrival + expected transaction bytes
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
'''

COPY_OLD = """      uint8_t* slot = sm + st * SLOTB;
#pragma unroll
      for (int i2 = 0; i2 < 8; ++i2) {
        const int key = 32 * cw_ + 4 * i2 + (lane >> 3), ch = lane & 7;   // whole 128 B rows of 4 keys per instruction (AF56)
        const long row = __shfl_sync(FULL, r0, key & 31);
        if (ABL != 2) {
          cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
               kc + row * row_bytes + poff + 16 * ch);
          cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
        }
      }
"""
COPY_NEW = """      uint8_t* slot = sm + st * SLOTB;
      {   // W3t: lane -> key 32 cw_ + lane: its K and V rows as two 128 B bulk copies into row-major staging
        if (lane == 0) bar_expect_tx(&land_bar[st], 32u * 2u * 128u);
        __syncwarp();
        const int key = 32 * cw_ + lane;
        bulk128(slot + OFF_KS + key * VS_PITCH, kc + r0 * row_bytes + poff, &land_bar[st]);
        bulk128(slot + OFF_VS + key * VS_PITCH, vc + r0 * row_bytes + poff, &land_bar[st]);
      }
"""


def render():
    text = open(os.path.join(HERE, "cuda_w3b.py")).read()
    for pat, rep in ((r"\bokc_w3b", "okc_w3t"), (r"\bOKC_W3B_", "OKC_W3T_"), (r"\battend_w3b", "attend_w3t"),
                     (r"\bCUDA_W3B_", "CUDA_W3T_"), (r"\bW3B_VARIANTS", "W3T_VARIANTS")):
        text = re.sub(pat, rep, text)
    # helpers before the kernel template
    text = sub1(text, "template <int STAGES, int PP, int NWG, int RC, int RP, int SBOV, int WF>\n__global__",
                BULK + "template <int STAGES, int PP, int NWG, int RC, int RP, int SBOV, int WF>\n__global__", "helpers")
    # slot layout: Kc | Vs | Vt | Ks
    text = sub1(text, "  constexpr int SLOTB = OFF_VT + 2 * VTH;\n",
                "  constexpr int OFF_KS = OFF_VT + 2 * VTH;            // W3t: K rows land row-major here (pitch 144)\n"
                "  constexpr int SLOTB = OFF_KS + 64 * VS_PITCH;\n", "slot layout")
    text = sub1(text, 'asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(smem_u32(&land_bar[s])));   // both copiers\' lanes',
                'asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(smem_u32(&land_bar[s])));   // W3t: one expect-tx arrive per copier', "land count")
    # copier: bulk copies; the land arrive is the expect-tx one, so drop the noinc cp.async arrive
    text = sub1(text, COPY_OLD, COPY_NEW, "copier copies")
    text = sub1(text, """        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&land_bar[st])) : "memory");
        issue_pt(i + STAGES);""", """        issue_pt(i + STAGES);                          // W3t: the land barrier completes on the bulk copies' bytes""", "copier arrive")
    # transposers: K re-lay before the V transpose
    text = sub1(text, """      const uint8_t* vs = sm + st * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = sm + st * SLOTB + OFF_VT + hh * VTH;
#pragma unroll
      for (int pp = 0; pp < (ABL == 3 ? 0 : 2); ++pp) {""", """      {   // W3t: K rows of keys 32p .. 32p+31 (p = this transposer) from Ks into the K-major tile, 16 B moves
        uint8_t* slot = sm + st * SLOTB;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
          const int key = 32 * (warp - PW - 2) + 4 * it + (lane >> 3), ch = lane & 7;
          *reinterpret_cast<uint4*>(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16) =
              *reinterpret_cast<const uint4*>(slot + OFF_KS + key * VS_PITCH + 16 * ch);
        }
      }
      const uint8_t* vs = sm + st * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = sm + st * SLOTB + OFF_VT + hh * VTH;
#pragma unroll
      for (int pp = 0; pp < (ABL == 3 ? 0 : 2); ++pp) {""", "transposer K re-lay")
    # host smem + variants
    text = sub1(text, "  return static_cast<size_t>(stages) * (okc_w3t::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(2 * stages) * 64 * 4 +",
                "  return static_cast<size_t>(stages) * (okc_w3t::OFF_VT + 2 * 8 * 528 + 64 * okc_w3t::VS_PITCH) + static_cast<size_t>(2 * stages) * 64 * 4 +",
                "host smem")
    m = re.findall(r"#define OKC_W3T_VARIANTS\(X\) [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "#define OKC_W3T_VARIANTS(X) " + " ".join(f"X({s}, {p}, {n}, {rc}, {rp})" for s, p, n, rc, rp in VARIANTS) + "\n")
    m = re.findall(r"\nW3T_VARIANTS = [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "\nW3T_VARIANTS = " + repr(VARIANTS) + "\n")
    for s, p, n, rc, rp in VARIANTS:
        smem = s * (2 * 4096 + 64 * 144 + 2 * 8 * 528 + 64 * 144) + 2 * s * 64 * 4 + 2 * n * 4096
        assert smem + 1024 <= 227 * 1024, (s, smem)
    # watchdog (as make_w3d.py)
    bw = ('__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  while (!done)\n'
          '    asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n'
          '                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n')
    assert text.count(bw) == 1
    text = text.replace(bw, '__device__ unsigned int okc_w3t_hang = 0;   // watchdog flag (make_w3t.py)\n'
                        + bw.replace('  uint32_t done = 0;\n  while (!done)\n', '  uint32_t done = 0, spins = 0;\n  while (!done) {\n')
                            .replace(' : "memory");\n}\n', ' : "memory");\n'
                                     '    if (!done && ++spins > (1u << 20)) {\n      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_w3t_hang), "r"(1u) : "memory");\n      break;\n    }\n  }\n}\n'))
    host = ('\nint64_t attend_w3t_hang() {   // lab: 1 when a W3t barrier wait gave up (watchdog); reads and clears the flag\n'
            '  unsigned int h = 0, z = 0;\n'
            '  cudaMemcpyFromSymbol(&h, okc_w3t::okc_w3t_hang, sizeof(h));\n'
            '  cudaMemcpyToSymbol(okc_w3t::okc_w3t_hang, &z, sizeof(z));\n'
            '  return h;\n}\n')
    k = text.index('"""\n\nCUDA_W3T_DECL = ')
    text = text[:k] + host + text[k:]
    text = text.replace("std::vector<int64_t> attend_w3t_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);'",
                        "std::vector<int64_t> attend_w3t_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);\\nint64_t attend_w3t_hang();'")
    assert "int64_t attend_w3t_hang();" in text
    assert "cp.async.mbarrier.arrive.noinc" not in text.split("if (warp >= PW + 2) {")[0].split("if (warp == PW || warp == PW + 1) {")[1]
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_w3t.py from cuda_w3b.py (do not edit): W3t, W3b with a TMA bulk-copy copier (see\n'
            'make_w3t.py). Original docstring follows.\n' + text[3:i] + text[i:])


def coverage():
    """Every K and V row of a stage is bulk-copied exactly once (copier c lane l -> key 32c + l); the transposers' K re-lay
    moves every (key, 16 B chunk) once (warp p, it, lane -> key 32p + 4 it + lane / 8, chunk lane % 8), to W3b's tile formula."""
    keys = sorted(32 * c + l for c in range(2) for l in range(32))
    assert keys == list(range(64))
    seen = {}
    for p in range(2):
        for it in range(8):
            for lane in range(32):
                k, ch = 32 * p + 4 * it + (lane >> 3), lane & 7
                seen[(k, ch)] = seen.get((k, ch), 0) + 1
    assert sorted(seen) == [(k, c) for k in range(64) for c in range(8)] and set(seen.values()) == {1}
    # expect-tx bytes per stage = what the copies deliver: 2 copiers x 32 lanes x 2 rows x 128 B
    assert 2 * (32 * 2 * 128) == 64 * 2 * 128


if __name__ == "__main__":
    coverage()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w3t.py up to date" if same else "cuda_w3t.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"coverage ok; wrote {OUT}")

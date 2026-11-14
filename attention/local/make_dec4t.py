"""Generates attention/local/cuda_dec4t.py: D3t = D3 (cuda_dec4.py, full decode) with TMA bulk copies (B7 / AF66).
D3's warp-private rings stage each 16-key chunk row-major (pitch 144: K rows, then V rows; page entries in the K rows'
pad bytes 128..143). D3t: per chunk, lane l issues one 128 B `cp.async.bulk` (lanes 0-15: the K row of token l, 16-31:
the V row of token l - 16) completing on a per-(warp, stage) mbarrier (lane 0's arrive.expect_tx of 32 x 128 = 4,096 B
per chunk; a generic -> async proxy fence before the copies orders the slot's earlier reads); the page entries keep
their cp.async groups (D3's ring arithmetic unchanged), and before reading chunk i the warp also waits its data barrier
(parity (i / STAGES) & 1). Everything else byte for byte. Barrier watchdog. Host: attend_dec4t(... as attend_dec4 ...),
attend_dec4t_setup(), attend_dec4t_hang().
Run: .venv/bin/python attention/local/make_dec4t.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_dec4t.py")


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


HELPERS = r'''__device__ unsigned int okc_dec4t_hang = 0;   // watchdog flag (make_dec4t.py)
__device__ __forceinline__ void bulk128(void* dst, const void* src, uint64_t* bar) {   // D3t: one 128 B TMA bulk copy
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {   // with the lab watchdog
  uint32_t done = 0, spins = 0;
  while (!done) {
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (!done && ++spins > (1u << 20)) {
      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_dec4t_hang), "r"(1u) : "memory");
      break;
    }
  }
}

'''


def render():
    text = open(os.path.join(HERE, "cuda_dec4.py")).read()
    for pat, rep in ((r"\bokc_dec4", "okc_dec4t"), (r"\bOKC_DEC4_", "OKC_DEC4T_"), (r"\battend_dec4", "attend_dec4t"),
                     (r"\bCUDA_DEC4_", "CUDA_DEC4T_")):
        text = re.sub(pat, rep, text)
    text = sub1(text, "template <int WARPS, int STAGES>\n__global__", HELPERS + "template <int WARPS, int STAGES>\n__global__", "helpers")
    text = sub1(text, "  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;\n",
                "  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;\n"
                "  __shared__ __align__(8) uint64_t dbar[WARPS][STAGES];   // D3t: chunk data barriers per (warp, stage)\n"
                "  if (lane == 0) {\n"
                "#pragma unroll\n"
                "    for (int s = 0; s < STAGES; ++s) asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 1;\" ::\"r\"(smem_u32(&dbar[warp][s])));\n"
                "    asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n"
                "  }\n"
                "  __syncwarp();\n", "barrier init")
    text = sub1(text, """      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
""", """      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
      {   // D3t: lane l < 16: the K row of token l, lane l >= 16: the V row of token l - 16 (one 128 B bulk copy each)
        if (lane == 0) bar_expect_tx(&dbar[warp][i % STAGES], 2u * CHUNK * 128u);
        __syncwarp();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the slot's earlier generic reads first
        const int tk = lane & 15;
        if (lane < CHUNK) bulk128(st + tk * PITCH, kc + my_row * row_bytes + poff, &dbar[warp][i % STAGES]);
        else bulk128(st + CHUNK * PITCH + tk * PITCH, vc + my_row * row_bytes + poff, &dbar[warp][i % STAGES]);
      }
""", "issue copies")
    text = sub1(text, """  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
""", """  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));   // page entries (D3's groups)
    __syncwarp();
    mbar_wait(&dbar[warp][i % STAGES], static_cast<uint32_t>((i / STAGES) & 1));   // D3t: chunk i's rows landed
""", "loop wait")
    host = ('\nint64_t attend_dec4t_hang() {   // lab: 1 when a D3t barrier wait gave up (watchdog); reads and clears the flag\n'
            '  unsigned int h = 0, z = 0;\n'
            '  cudaMemcpyFromSymbol(&h, okc_dec4t::okc_dec4t_hang, sizeof(h));\n'
            '  cudaMemcpyToSymbol(okc_dec4t::okc_dec4t_hang, &z, sizeof(z));\n'
            '  return h;\n}\n')
    k = text.index('"""\n\nCUDA_DEC4T_DECL = ')
    text = text[:k] + host + text[k:]
    m = re.search(r"CUDA_DEC4T_DECL = \((.*?)\)\n", text, re.S) or re.search(r"CUDA_DEC4T_DECL = '(.*?)'\n", text, re.S)
    assert m, "decl"
    decl_line = text[m.start():m.end()]
    ns = {}
    exec(decl_line, ns)
    newdecl = ns["CUDA_DEC4T_DECL"] + "\nint64_t attend_dec4t_hang();"
    text = text.replace(decl_line, "CUDA_DEC4T_DECL = " + repr(newdecl) + "\n")
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_dec4t.py from cuda_dec4.py (do not edit): D3t, D3 with TMA bulk copies (see\n'
            'make_dec4t.py). Original docstring follows.\n' + text[3:i] + text[i:])


def coverage():
    """Per chunk, lanes 0-15 copy the 16 K rows and 16-31 the 16 V rows once each; 2 x 16 x 128 = 4,096 expected bytes;
    rows land at bytes 0..127 of their pitch-144 slot rows (the page-entry pads 128..143 untouched)."""
    rows = sorted(("K" if l < 16 else "V", l & 15) for l in range(32))
    assert rows == sorted([("K", t) for t in range(16)] + [("V", t) for t in range(16)])
    assert 2 * 16 * 128 == 4096 and 128 <= 144 - 16


if __name__ == "__main__":
    coverage()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_dec4t.py up to date" if same else "cuda_dec4t.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"coverage ok; wrote {OUT}")

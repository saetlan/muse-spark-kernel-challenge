"""F12M helper warp for the escape corrections, in decode_layer's F12M v6 source form (pj_F12M_SRC: template <NT, RTW, RG,
KSW, NS, D, F12>, one `load` lambda, bx = blockIdx.x). From bf16_proj's lab/f12mh.py (PE46; judge: o m8 +3.8%, qkv m16 +2.9%,
o m1 vs fma +9% in v13's check 28597f4e; loses where 288 threads x registers > 64K for 2 CTAs per SM: qkv m8 D4 at 114 regs
0.82x, unpooler m16 D8 0.87x). Changes:
  - one more warp (launch bounds and the wrapper's thread count + 32); warp RG * KSW is the helper and skips the prologue loads,
    the main loop and the part[] writes; the streaming warps no longer load the row records;
  - the helper loads each row's record (one lane per row), writes recs[], and computes corr[row][tok] = sum_j fmaf(xk, w_true,
    -xk * wdec) (FP32, fixed order) while the others stream;
  - the epilogue: sum = fmaf(sum, scale, corr[row][tok]) with no global reads (FP32 rounding differs slightly from F12M).
Includes the part[] padding (F64) when it isn't there yet. Use per shape (hybrid), not everywhere: keep F12M where it loses.
Verified by bf16_proj against decode_layer/scored/submission_v20.py's pj_F12M_SRC: every anchor matches exactly once, and the
rendered source (5 instances) passes clang -fsyntax-only with proj/local/cuda_stubs.
"""

MH_EDITS_V6 = [('template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12>\n__global__ void __launch_bounds__(RG * KSW * 32)', 'template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12>\n__global__ void __launch_bounds__(RG * KSW * 32 + 32)'), ('  __shared__ uint4 recs[F12 ? 2 * RB : 1];\n', '  __shared__ uint4 recs[F12 ? 2 * RB : 1];\n  __shared__ float corr[F12 ? RB : 1][MT + 3];\n'), ('  const long long bx = blockIdx.x;\n\n  bool tok_ok[NT];', '  const long long bx = blockIdx.x;\n  const bool helper = warp == RG * KSW;   // the escape warp\n\n  bool tok_ok[NT];'), ('#pragma unroll\n  for (int s = 0; s < D; ++s) load(s, s);\n', '  if (!helper) {\n#pragma unroll\n  for (int s = 0; s < D; ++s) load(s, s);\n'), ('  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);\n  if (F12 && tid < RB) {\n    ld_x(rec + f12m_ridx(bx * RB + tid, 0), rc0);\n    ld_x(rec + f12m_ridx(bx * RB + tid, 1), rc1);\n  }\n', ''), ('      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][f12m_ctok(n, t, r)] = acc[i][n][r];\n  if (F12 && tid < RB) {\n    recs[2 * tid] = rc0;\n    recs[2 * tid + 1] = rc1;\n  }\n  __syncthreads();\n', '      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][f12m_ctok(n, t, r)] = acc[i][n][r];\n  }\n  if (F12 && helper) {\n    const unsigned short* xe = reinterpret_cast<const unsigned short*>(x);\n    for (int row = lane; row < RB; row += 32) {\n      uint4 q0, q1;\n      ld_x(rec + f12m_ridx(bx * RB + row, 0), q0);\n      ld_x(rec + f12m_ridx(bx * RB + row, 1), q1);\n      recs[2 * row] = q0;\n      recs[2 * row + 1] = q1;\n      const uint32_t cnt = (q0.x >> 8) & 0xFFu;\n      const float wd = f12m_wdec(q0.x & 0xFFu);\n      const uint32_t es[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};\n      for (int tok = 0; tok < MT; ++tok) {\n        float c = 0.0f;\n        if (tok < m) {\n#pragma unroll\n          for (int j = 0; j < 7; ++j) {\n            if (j < cnt) {\n              const float xk = __uint_as_float(static_cast<uint32_t>(xe[static_cast<long long>(tok) * (8 * K8) + (es[j] & 0xFFFFu)]) << 16);\n              c = fmaf(-xk, wd, c);\n              c = fmaf(xk, __uint_as_float(es[j] & 0xFFFF0000u), c);\n            }\n          }\n        }\n        corr[row][tok] = c;\n      }\n    }\n  }\n  __syncthreads();\n'), ('      sum *= f12m_scale(base);\n      const float wdec = f12m_wdec(base);\n      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};\n#pragma unroll\n      for (int j = 0; j < 7; ++j) {\n        if (j < count) {\n          const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);\n          sum = fmaf(-xk, wdec, sum);\n          sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);\n        }\n      }\n', '      (void)count;\n      sum = fmaf(sum, f12m_scale(base), corr[row][tok]);\n')]


def f12mh_v6_source(rendered):
    """`rendered` = pj_f12m_source(instances) (pj_F12M_SRC with the wrappers); returns the helper version (namespace f12mh,
    wrapper names pj_f12m_run_ -> f12mh_run_ is left to the caller's naming)."""
    src = rendered
    if "__shared__ float part[KSW][RB][MT];" in src:
        src = src.replace("__shared__ float part[KSW][RB][MT];", "__shared__ float part[KSW][RB][MT + 3];")
    for a, b in MH_EDITS_V6:
        assert src.count(a) == 1, ("anchor not found exactly once", a[:80])
        src = src.replace(a, b)
    assert ", 256, 0, stream>>>" in src or ", 128, 0, stream>>>" in src
    src = src.replace(", 256, 0, stream>>>", ", 288, 0, stream>>>").replace(", 128, 0, stream>>>", ", 160, 0, stream>>>")
    return src.replace("namespace f12m {", "namespace f12mh {").replace("f12m::", "f12mh::")

"""F12MH (PE46): F12M with a helper warp for the escape corrections (PE44: the escape loop costs 0.2-0.4 us on every F12M
shape, all of it in the epilogue's dependent x reads). Built from the base file's F12M source text by exact edits:
  - one more warp (launch bounds and wrapper threads + 32); warp RG * KSW is the helper, the others stream as before;
  - the helper loads each row's record (one lane per row), writes recs[], and for every token computes the row's correction
    sum over its escapes, c = sum_j (x_k * w_true - x_k * w_dec) (FP32, fixed order), into corr[row][tok] while the others stream;
  - the streaming warps no longer load records; the epilogue applies sum * scale + corr[row][tok] (no global reads);
  - part[] keeps v12's MT + 3 row stride.
Same math as F12M up to FP32 rounding (the corrections are summed apart, then added once)."""

MH_EDITS = [
    ("template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12, int CS, int XD>\n__global__ void __launch_bounds__(RG * KSW * 32)",
     "template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12, int CS, int XD>\n__global__ void __launch_bounds__(RG * KSW * 32 + 32)"),
    ("  __shared__ uint4 recs[F12 ? 2 * RB : 1];\n",
     "  __shared__ uint4 recs[F12 ? 2 * RB : 1];\n  __shared__ float corr[F12 ? RB : 1][MT + 3];\n"),
    ("  const long long bx = blockIdx.x / CS;\n\n  bool tok_ok[NT];",
     "  const long long bx = blockIdx.x / CS;\n  const bool helper = warp == RG * KSW;   // the escape warp\n\n  bool tok_ok[NT];"),
    # streaming warps only: prologue loads, main loop, part writes
    ("#pragma unroll\n  for (int s = 0; s < D; ++s) {\n    loadw(s, s);\n    if (s < XD) loadx(s, s);\n  }\n",
     "  if (!helper) {\n#pragma unroll\n  for (int s = 0; s < D; ++s) {\n    loadw(s, s);\n    if (s < XD) loadx(s, s);\n  }\n"),
    ("  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);\n  if (F12 && tid < RB) {\n"
     "    ld_x(rec + f12m_ridx(bx * RB + tid, 0), rc0);\n    ld_x(rec + f12m_ridx(bx * RB + tid, 1), rc1);\n  }\n",
     ""),
    ("      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][f12m_ctok(n, t, r)] = acc[i][n][r];\n"
     "  if (F12 && tid < RB) {\n    recs[2 * tid] = rc0;\n    recs[2 * tid + 1] = rc1;\n  }\n  __syncthreads();\n",
     "      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][f12m_ctok(n, t, r)] = acc[i][n][r];\n"
     "  }\n"
     "  if (F12 && helper) {\n"
     "    const unsigned short* xe = reinterpret_cast<const unsigned short*>(x);\n"
     "    for (int row = lane; row < RB; row += 32) {\n"
     "      uint4 q0, q1;\n"
     "      ld_x(rec + f12m_ridx(bx * RB + row, 0), q0);\n"
     "      ld_x(rec + f12m_ridx(bx * RB + row, 1), q1);\n"
     "      recs[2 * row] = q0;\n"
     "      recs[2 * row + 1] = q1;\n"
     "      const uint32_t cnt = (q0.x >> 8) & 0xFFu;\n"
     "      const float wd = f12m_wdec(q0.x & 0xFFu);\n"
     "      const uint32_t es[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};\n"
     "      for (int tok = 0; tok < MT; ++tok) {\n"
     "        float c = 0.0f;\n"
     "        if (tok < m) {\n"
     "#pragma unroll\n"
     "          for (int j = 0; j < 7; ++j) {\n"
     "            if (j < cnt) {\n"
     "              const float xk = __uint_as_float(static_cast<uint32_t>(xe[static_cast<long long>(tok) * (8 * K8) + (es[j] & 0xFFFFu)]) << 16);\n"
     "              c = fmaf(-xk, wd, c);\n"
     "              c = fmaf(xk, __uint_as_float(es[j] & 0xFFFF0000u), c);\n"
     "            }\n"
     "          }\n"
     "        }\n"
     "        corr[row][tok] = c;\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  __syncthreads();\n"),
    ("      sum *= f12m_scale(base);\n      const float wdec = f12m_wdec(base);\n"
     "      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};\n#pragma unroll\n      for (int j = 0; j < 7; ++j) {\n"
     "        if (j < count) {\n"
     "          const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);\n"
     "          sum = fmaf(-xk, wdec, sum);\n          sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);\n        }\n      }\n",
     "      (void)count;\n      sum = fmaf(sum, f12m_scale(base), corr[row][tok]);\n"),
]


def f12mh_source(instances, base_source):
    """F12M's rendered source (base_source(instances)) with MH_EDITS, the padded part[] and the +32 wrapper threads."""
    src = base_source(instances)
    if "__shared__ float part[KSW][RB][MT];" in src:
        src = src.replace("__shared__ float part[KSW][RB][MT];", "__shared__ float part[KSW][RB][MT + 3];")
    for a, b in MH_EDITS:
        assert src.count(a) == 1, ("f12mh edit anchor not found exactly once", a[:80])
        src = src.replace(a, b)
    for inst in instances:
        nt, rtw, rg, ksw = inst[:4]
        cs = inst[7] if len(inst) > 7 else 1
        assert cs == 1, "f12mh: cluster instances not handled"
        threads = 32 * rg * ksw
        tmpl_launch = f", {threads}, 0, stream>>>"
        assert tmpl_launch in src, inst
    src = src.replace(", 256, 0, stream>>>", ", 288, 0, stream>>>").replace(", 128, 0, stream>>>", ", 160, 0, stream>>>")
    return (src.replace("namespace f12m {", "namespace f12mh {").replace("f12m::", "f12mh::")
            .replace("void f12m_run_", "void f12mh_run_").replace("int64_t f12m_run_", "int64_t f12mh_run_"))

"""Generate M19 (namespace m19, kind "m19") from M14's source in moe/m13/m13.py and splice it in (E19, MF21).

M19 = M14 with its 32-register weight ring (rw[D][TT]) moved to shared memory: each thread cp.asyncs its own
16-B code words for the next group's chunk s into a per-warp ring of D + 1 chunk slots (the slot chunk s - 1
just released), before that chunk's activation issue, so the existing cp.async.wait_group<P - 1> covers it for
P <= D; compute LDS-es its words. Scales stay in registers. __launch_bounds__(128, MB) with MB in {3, 4}: with
the ring out of the register file, 4 CTAs per SM may fit (M14: 166 registers, 3 CTAs).

Run once: .venv/bin/python moe/local/make_m19.py   (refuses to run twice)
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "m13" / "m13.py"


def sub(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"make_m19: expected {count} of {old[:80]!r}, found {n}")
    return text.replace(old, new)


def make_kernel(m14):
    k = m14
    k = sub(k, "namespace m14 {", "namespace m19 {")
    k = sub(k, "}  // namespace m14", "}  // namespace m19")
    k = sub(k, "template <int NT>\nconstexpr int stage_bytes() { return 8 * NT * 128; }\n",
            "template <int NT>\nconstexpr int stage_bytes() { return 8 * NT * 128; }\n"
            "// activation stages, then each warp's code ring: D + 1 chunk slots x TT tiles x 32 lanes x 16 B\n"
            "template <int TT, int NT, int P>\n"
            "constexpr int dyn_bytes() { return (P + 1) * stage_bytes<NT>() + 4 * (D + 1) * TT * 512; }\n")
    k = sub(k, "template <int TT, int NT, bool GATED, int P, int MODE = 0>\n__global__ void __launch_bounds__(128) ck_kernel(",
            "template <int TT, int NT, bool GATED, int P, int MB, int MODE = 0>\n"
            "__global__ void __launch_bounds__(128, MB) wr_kernel(")
    k = sub(k, '  static_assert(P >= 2 && P <= 8, "activation lead");',
            '  static_assert(P >= 2 && P <= D, "the code ring (D chunks ahead) must cover the activation lead");\n'
            '  static_assert(MODE == 0, "M19 has no recon modes");')
    k = sub(k, "  const uint32_t sb0 = smem_addr(s_b);\n",
            "  const uint32_t sb0 = smem_addr(s_b);\n"
            "  constexpr int WSL = D + 1;  // code ring slots per warp: the next group's D chunks + the chunk in use\n"
            "  auto woff = [&](int slot, int i) { return SBP * STAGE + ((warp * WSL + slot) * TT + i) * 512 + lane * 16; };\n")
    k = sub(k, "  uint4 rw[D][TT];\n", "")
    k = sub(k, """  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) rw[s][i] = __ldcs(lw[i] + s * 32);
  };""", """  auto issue_w = [&](int s, int slot) {  // chunk s of the group lw points at -> code ring slot
#pragma unroll
    for (int i = 0; i < TT; ++i) cp_async16(sb0 + woff(slot, i), lw[i] + s * 32, 16u);
  };""")
    k = sub(k, "  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp, int slot) {",
            "  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp, int slot, int wslot) {")
    k = sub(k, """    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;""", """    for (int i = 0; i < TT; ++i) {
      const uint4 rwv = *reinterpret_cast<const uint4*>(s_b + woff(wslot, i));  // own words, landed (wait_group)
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;""")
    k = sub(k, "        const uint32_t q = word(rw[s][i], j);", "        const uint32_t q = word(rwv, j);")
    k = sub(k, """  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  load_s();""", """  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) issue_w(s, s);  // the first group's chunks -> slots 0..D-1 (first commit group)
  load_s();""")
    k = sub(k, "  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)\n",
            "  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)\n"
            "  int wc = 0;     // chunks consumed: chunk wc's codes are in ring slot wc % WSL\n")
    k = sub(k, """      cp_wait<P - 1>();
      __syncthreads();  // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      issue_b();        // chunk + P -> the stage the previous chunk used
      compute(s, rsc, cntp, cslot);
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;""", """      cp_wait<P - 1>();
      __syncthreads();  // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      // the next group's chunk s, D chunks ahead, into the slot chunk wc - 1 released; issued before issue_b so
      // it joins the commit group of activation chunk wc + P <= wc + D, which wait_group covers by then
      if (more) issue_w(s, (wc + D) % WSL);
      issue_b();        // chunk + P -> the stage the previous chunk used
      compute(s, rsc, cntp, cslot, wc % WSL);
      if (more && s == 0) load_s();
      ++wc;
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;""")
    if "rw[" in k or "load_w" in k:
        raise SystemExit("make_m19: register ring left in M19")
    return k


HOST = r'''
#define M19_CASE(G, TT, NT, P, MB)                                                                          \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && mb == (MB)) {                               \
    m13_launch(m19::wr_kernel<TT, NT, G, P, MB>, static_cast<int>(grid), 128,                              \
               static_cast<size_t>(m19::dyn_bytes<TT, NT, P>()), static_cast<int>(pdl),                    \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP; column 3 = P | MB << 4 (MB = CTAs per SM the launch bounds ask for).
#define M19_SETUP(G, TT, NT, P, MB)                                                                         \
  {                                                                                                        \
    const int dyn = m19::dyn_bytes<TT, NT, P>();                                                           \
    cudaFuncSetAttribute(m19::wr_kernel<TT, NT, G, P, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m19::wr_kernel<TT, NT, G, P, MB>);                                        \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m19::wr_kernel<TT, NT, G, P, MB>, 128, dyn);   \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((MB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m19_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m19::MAXP, "m19_gemm: ", n_pairs, " pairs, at most ", 8 * m19::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t mb = ((pf >> 28) & 1) ? 4 : 3;                    // launch-bounds CTAs per SM
  @@M19_CASES@@
  TORCH_CHECK(false, "m19_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa, " MB=", mb);
  return out;
}

torch::Tensor m19_setup() {
  std::vector<int64_t> rows;
  @@M19_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}
'''

PROTO = '''torch::Tensor m19_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m19_setup();
'''


def main():
    s = SRC.read_text()
    if "namespace m19 {" in s:
        raise SystemExit("make_m19: M19 already spliced in")
    i, j = s.index("namespace m14 {"), s.index("}  // namespace m14") + len("}  // namespace m14")
    m19 = ("\n// M19 (E19): M14 with the weight-code ring in shared memory (cp.async, D + 1 chunk slots per warp) "
           "instead of\n// 32 registers, and __launch_bounds__(128, MB), so 4 CTAs per SM may fit. Generated by "
           "moe/local/make_m19.py.\n" + make_kernel(s[i:j]) + "\n")
    anchor = "\n\n\n// Launch with programmatic stream serialization when pdl"
    s = sub(s, anchor, "\n" + m19 + anchor)
    # host side, after m18_recon's definition
    a2 = '  TORCH_CHECK(false, "m18_recon: mode ", mode);\n  return out;\n}\n'
    s = sub(s, a2, a2 + HOST)
    a3 = "torch::Tensor m18_setup();\n"
    s = sub(s, a3, a3 + PROTO)
    s = sub(s, '"m11p_gemm", "m18_gemm", "m18_setup", "m18_recon"],\n        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + ([]',
            '"m11p_gemm", "m18_gemm", "m18_setup", "m18_recon", "m19_gemm", "m19_setup"],\n'
            '        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + ([]')
    s = sub(s, "M18_INSTANCES = ((1, 2, 8, 2),",
            "# M19 (E19): (gated, tt, nt, P, MB): M14's best stages with the code ring in shared memory, 3 or 4 CTAs per SM\n"
            "M19_INSTANCES = ((1, 2, 2, 2, 3), (1, 2, 2, 2, 4), (0, 2, 2, 4, 3), (0, 2, 2, 4, 4))\n"
            "M18_INSTANCES = ((1, 2, 8, 2),")
    s = sub(s, "m17_instances=None, m18_instances=None):",
            "m17_instances=None, m18_instances=None,\n               m19_instances=M19_INSTANCES):")
    s = sub(s, "    args18 = [f\"{'true' if g else 'false'}, {tt}, {nt}, {pa}\" for g, tt, nt, pa in m18_instances]\n",
            "    args18 = [f\"{'true' if g else 'false'}, {tt}, {nt}, {pa}\" for g, tt, nt, pa in m18_instances]\n"
            "    args19 = [f\"{'true' if g else 'false'}, {tt}, {nt}, {pa}, {mb}\" for g, tt, nt, pa, mb in m19_instances]\n")
    s = sub(s, '           .replace("@@M18_CASES@@", "\\n  ".join(f"M18_CASE({a})" for a in args18))\n',
            '           .replace("@@M19_CASES@@", "\\n  ".join(f"M19_CASE({a})" for a in args19))\n'
            '           .replace("@@M19_SETUPS@@", "\\n  ".join(f"M19_SETUP({a})" for a in args19))\n'
            '           .replace("@@M18_CASES@@", "\\n  ".join(f"M18_CASE({a})" for a in args18))\n')
    s = sub(s, '    state["m18_resources"] = state["m11_ext"].m18_setup().tolist()\n',
            '    state["m18_resources"] = state["m11_ext"].m18_setup().tolist()\n'
            '    state["m19_resources"] = state["m11_ext"].m19_setup().tolist()\n')
    # glue: kind lists, the MB bit (pf bit 28), the split's kinds
    s = sub(s, 'if kind not in ("m13", "m14", "m15", "m16", "m17", "m18"):', 'if kind not in ("m13", "m14", "m15", "m16", "m17", "m18", "m19"):', 2)
    s = sub(s, '        gemm(*args, pf | (s1.get("p", 0) << 24) | (s1.get("db", 0) << 28) | (thr << 16))',
            '        gemm(*args, pf | (s1.get("p", 0) << 24) | (s1.get("db", 0) << 28) | (thr << 16) | ((s1.get("mb", 3) == 4) << 28))')
    s = sub(s, '        args[-1] |= (s2.get("p", 0) << 24) | (s2.get("db", 0) << 28) | (thr << 16)',
            '        args[-1] |= (s2.get("p", 0) << 24) | (s2.get("db", 0) << 28) | (thr << 16) | ((s2.get("mb", 3) == 4) << 28)')
    s = sub(s, 'assert not thr or (kind in ("m14", "m18") and', 'assert not thr or (kind in ("m14", "m18", "m19") and', 2)
    SRC.write_text(s)
    print(f"make_m19: spliced M19 into {SRC} ({len(m19)} chars of kernel)")


if __name__ == "__main__":
    main()

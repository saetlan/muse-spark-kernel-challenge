"""Generates attention/local/cuda_w1dbg.py: W1 (cuda_w1.py) with a debug dump, for the L22 lab (AF44: the
wgmma unit test is correct on the GPU, W1 is not). The kernel text is cuda_w1.py's, copied by anchor, with:
- template <int STAGES, int SBOV, int WF, int DBG> and an extra float* dbg argument;
- for block (split 0, request 0 x KV pair 0), stage 0 only, the values each step produced, in dbg (floats):
    [    0,  8192)  raw Q.K accumulators per head h: h*4096 + row*64 + key          (after wait_group, before masks)
    [ 8192, 16384)  P (exp2 values, before the e4m3 pack), same layout
    [16384, 24576)  P.V accumulators (opv), h*4096 + row*64 + dim
    [24576, 32768)  Q A-register bytes, h*4096 + row*64 + dim (byte values 0..255)
    [32768, 40960)  K tile bytes as the consumer sees them after full_bar, h*4096 + offset
    [40960, 49408)  Vt bytes, h*4224 + offset (8*SBOV <= 4224)
    [49408, 58624)  V staging bytes (64 rows x 144)
    [58624, 58688)  the copier's cache rows of keys 0..63 (r0 lanes 0..31, r1 lanes 0..31)
  row = 16*w + g + 8*half (token 2w + half, query head (kv0+h)*8 + g), the kernel's own mapping;
- host attend_w1dbg(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, dbg, batch, hq, hkv, max_pages,
  nsplit, page, window, sm_scale): STAGES 4, SBOV 528, WF 0 (L20's fastest variant).
The lab compares every region with torch (lab_l22_block.py); emulate_cuda_w1.py records the same regions
(DUMP) so the analysis is checked locally against an independent implementation first.
Run: .venv/bin/python attention/local/make_w1dbg.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w1dbg.py")
DBG_FLOATS = 65536


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def sub1(src, a, b, what):
    assert src.count(a) == 1, f"{what}: anchor x{src.count(a)}"
    return src.replace(a, b)


W1 = load("cuda_w1.py", "CUDA_W1_SRC")
src = W1[:W1.index("#define OKC_W1_VARIANTS")]
src = src.replace("namespace okc_w1 {", "namespace okc_w1d {").replace("}  // namespace okc_w1\n", "}  // namespace okc_w1d\n")
assert "okc_w1d" in src and "namespace okc_w1 " not in src
src = sub1(src, "template <int STAGES, int SBOV, int WF>\n__global__ void __launch_bounds__(352, 1) attend_w1_kernel(",
           "template <int STAGES, int SBOV, int WF, int DBG>\n__global__ void __launch_bounds__(352, 1) attend_w1_kernel(",
           "template")
src = sub1(src, "float* __restrict__ l_part,\n    int hq,", "float* __restrict__ l_part, float* __restrict__ dbg,\n    int hq,",
           "signature")

ON = "DBG && blockIdx.x == 0 && blockIdx.y == 0 && i == 0"
ROWCOL = ("const int row = 16 * (warp & 3) + g + 8 * ((k >> 1) & 1), col = 8 * (k >> 2) + 2 * t4 + (k & 1);")

src = sub1(src, "      const long r1 = static_cast<long>(pe[32 + lane]) * page + tok(i, 32 + lane) % page;\n",
           "      const long r1 = static_cast<long>(pe[32 + lane]) * page + tok(i, 32 + lane) % page;\n"
           f"      if ({ON}) {{\n"
           "        dbg[58624 + lane] = static_cast<float>(r0);\n"
           "        dbg[58656 + lane] = static_cast<float>(r1);\n"
           "      }\n", "copier rows")

src = sub1(src, "    fence_a(qa[1]);\n    if (s0 + SKEYS > end",
           "    fence_a(qa[1]);\n"
           f"    if ({ON}) {{\n"
           "#pragma unroll\n"
           "      for (int k = 0; k < 32; ++k) {\n"
           f"        {ROWCOL}\n"
           "        dbg[h * 4096 + row * 64 + col] = sc[k];\n"
           "      }\n"
           "#pragma unroll\n"
           "      for (int s = 0; s < 2; ++s)\n"
           "#pragma unroll\n"
           "        for (int r = 0; r < 4; ++r)\n"
           "#pragma unroll\n"
           "          for (int by = 0; by < 4; ++by) {\n"
           "            const int row = 16 * (warp & 3) + g + 8 * (r & 1), dim = 32 * s + 4 * t4 + by + 16 * (r >> 1);\n"
           "            dbg[24576 + h * 4096 + row * 64 + dim] = static_cast<float>((qa[s][r] >> (8 * by)) & 0xFFu);\n"
           "          }\n"
           "      const uint8_t* kt = sm + st * SLOTB + h * KC_HEAD;\n"
           "      for (int x = tid & 127; x < KC_HEAD; x += 128) dbg[32768 + h * 4096 + x] = static_cast<float>(kt[x]);\n"
           "    }\n"
           "    if (s0 + SKEYS > end", "after Q.K")

src = sub1(src, "    // P as A registers of k-step c",
           f"    if ({ON}) {{\n"
           "#pragma unroll\n"
           "      for (int k = 0; k < 32; ++k) {\n"
           f"        {ROWCOL}\n"
           "        dbg[8192 + h * 4096 + row * 64 + col] = sc[k];\n"
           "      }\n"
           "    }\n"
           "    // P as A registers of k-step c", "before pack")

src = sub1(src, "    __syncwarp();                                     // this warp's wgmma reads of the stage are complete\n",
           f"    if ({ON}) {{\n"
           "#pragma unroll\n"
           "      for (int k = 0; k < 32; ++k) {\n"
           f"        {ROWCOL}\n"
           "        dbg[16384 + h * 4096 + row * 64 + col] = opv[k];\n"
           "      }\n"
           "      const uint8_t* vtp = sm + st * SLOTB + OFF_VT + h * VTH;\n"
           "      for (int x = tid & 127; x < VTH; x += 128) dbg[40960 + h * 4224 + x] = static_cast<float>(vtp[x]);\n"
           "      if (h == 0)\n"
           "        for (int x = tid & 127; x < 64 * VS_PITCH; x += 128)\n"
           "          dbg[49408 + x] = static_cast<float>(sm[st * SLOTB + OFF_VS + x]);\n"
           "    }\n"
           "    __syncwarp();                                     // this warp's wgmma reads of the stage are complete\n",
           "after P.V")

HOST = r"""
void attend_w1dbg(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                  torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                  torch::Tensor l_part, torch::Tensor dbg, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages,
                  int64_t nsplit, int64_t page, int64_t window, double sm_scale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "attend_w1dbg: 8 query heads per KV head, KV heads in pairs");
  TORCH_CHECK(dbg.numel() >= DBG_FLOATS, "attend_w1dbg: dbg buffer");   // data_ptr<float>() checks the dtype
  constexpr int S = 4, V = 528;
  const size_t smem = static_cast<size_t>(S) * (okc_w1d::OFF_VT + 2 * 8 * V) + static_cast<size_t>(2 * S) * 64 * 4;
  cudaError_t e = cudaFuncSetAttribute(okc_w1d::attend_w1_kernel<S, V, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(smem));
  TORCH_CHECK(e == cudaSuccess, "attend_w1dbg smem attribute: ", cudaGetErrorString(e));
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  okc_w1d::attend_w1_kernel<S, V, 0, 1><<<grid, 352, smem, stream>>>(
      reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      dbg.data_ptr<float>(), static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages),
      static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale));
  e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w1dbg launch failed: ", cudaGetErrorString(e));
}
""".replace("DBG_FLOATS", str(DBG_FLOATS))

DECL = ("void attend_w1dbg(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
        "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
        "torch::Tensor l_part, torch::Tensor dbg, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, "
        "int64_t nsplit, int64_t page, int64_t window, double sm_scale);")


def render():
    body = src + HOST
    assert body.count(f"if ({ON})") == 4
    return ('"""Generated by make_w1dbg.py (do not edit): W1 with a stage-0 debug dump (see make_w1dbg.py)."""\n\n'
            f"CUDA_W1DBG_SRC = r\"\"\"\n{body}\"\"\"\n\nCUDA_W1DBG_DECL = {DECL!r}\n\nW1DBG_FLOATS = {DBG_FLOATS}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w1dbg.py up to date" if same else "cuda_w1dbg.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")

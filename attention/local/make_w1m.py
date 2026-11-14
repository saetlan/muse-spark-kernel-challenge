"""Generates attention/local/cuda_w1m.py and cuda_w1c.py, the AF46 bisection for the L25 lab:
- W1-M (cuda_w1m.py): the frozen L22/L23 debug kernel (cuda_w1dbg_l22.py, the text W1-N is built from) with a
  template MASK choosing which of its four dump branches exist (bit 0 copier rows, 1 after Q.K, 2 before the P
  pack, 3 after P.V), each also guarded by dbg != nullptr (the host passes nullptr): MASK 15 = W1-N, MASK 0 = no
  dump code at all (only the extra kernel argument and template parameter). Host attend_w1m(..., stages, mask)
  with stages 4 and mask in {0, 1, 2, 4, 8, 15}.
- W1-C (cuda_w1c.py): the current W1 (cuda_w1.py: wgmma pairs in one asm statement) with a "memory" clobber on
  every cp.async commit_group / wait_group / wait_all, so no shared-memory load (the copier's page entries) can be
  scheduled above the wait that makes its cp.async data visible. Host attend_w1c(..., stages, sbov, wf).
Run: .venv/bin/python attention/local/make_w1m.py [--check]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = {"m": os.path.join(HERE, "cuda_w1m.py"), "c": os.path.join(HERE, "cuda_w1c.py")}


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:80]!r}"
    return src.replace(old, new)


def render_m():
    d = load("cuda_w1dbg_l22.py", "CUDA_W1DBG_L22_SRC")
    kern = d[:d.index("\nvoid attend_w1dbg(")]
    on = "if (DBG && blockIdx.x == 0 && blockIdx.y == 0 && i == 0)"
    assert kern.count(on) == 4
    for bit in range(4):    # occurrences in text order: copier rows, after Q.K, before pack, after P.V
        kern = kern.replace(on, f"if ((MASK & {1 << bit}) && dbg != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && i == 0)", 1)
    assert on not in kern
    kern = sub1(kern, "template <int STAGES, int SBOV, int WF, int DBG>\n", "template <int STAGES, int SBOV, int WF, int DBG, int MASK>\n", "template")
    kern = kern.replace("namespace okc_w1d {", "namespace okc_w1m {").replace("}  // namespace okc_w1d", "}  // namespace okc_w1m")
    assert "okc_w1d" not in kern
    host = r"""
#define OKC_W1M_VARIANTS(X) X(0) X(1) X(2) X(4) X(8) X(15)

void attend_w1m(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t mask) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0 && stages == 4, "attend_w1m: GQA 8, KV pairs, 4 stages");
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(4) * (okc_w1m::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(8) * 64 * 4;
  bool launched = false;
#define OKC_W1M_LAUNCH(M)                                                                                         \
  if (!launched && mask == M) {                                                                                   \
    okc_w1m::attend_w1_kernel<4, 528, 0, 1, M><<<grid, 352, smem, stream>>>(                                      \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        nullptr, static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages),                       \
        static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)); \
    launched = true;                                                                                              \
  }
  OKC_W1M_VARIANTS(OKC_W1M_LAUNCH)
#undef OKC_W1M_LAUNCH
  TORCH_CHECK(launched, "attend_w1m: no variant mask=", mask);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w1m launch failed: ", cudaGetErrorString(e));
}

int64_t attend_w1m_setup() {
  int err = 0;
#define OKC_W1M_SMEM(M) \
  err |= cudaFuncSetAttribute(okc_w1m::attend_w1_kernel<4, 528, 0, 1, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * (okc_w1m::OFF_VT + 2 * 8 * 528) + 8 * 64 * 4);
  OKC_W1M_VARIANTS(OKC_W1M_SMEM)
#undef OKC_W1M_SMEM
  (void)cudaGetLastError();
  return err;
}
"""
    decl = ("void attend_w1m(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
            "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
            "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
            "int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t mask);\nint64_t attend_w1m_setup();")
    return ('"""Generated by make_w1m.py (do not edit): W1-M, the AF46 dump-branch bisection (see make_w1m.py)."""\n\n'
            f"CUDA_W1M_SRC = r\"\"\"\n{kern + host}\"\"\"\n\nCUDA_W1M_DECL = {decl!r}\n")


def render_c():
    src = load("cuda_w1.py", "CUDA_W1_SRC")
    decl = load("cuda_w1.py", "CUDA_W1_DECL")
    n0 = src.count('asm volatile("cp.async.commit_group;");') + src.count('asm volatile("cp.async.wait_group 0;");')
    src = src.replace('asm volatile("cp.async.commit_group;");', 'asm volatile("cp.async.commit_group;" ::: "memory");')
    src = src.replace('asm volatile("cp.async.wait_group 0;");', 'asm volatile("cp.async.wait_group 0;" ::: "memory");')
    src = sub1(src, 'asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));', 'asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1) : "memory");', "wait n")
    src = sub1(src, 'asm volatile("cp.async.wait_all;");', 'asm volatile("cp.async.wait_all;" ::: "memory");', "wait all")
    assert n0 >= 5 and 'cp.async.commit_group;");' not in src and 'cp.async.wait_group 0;");' not in src
    src = src.replace("okc_w1", "okc_w1c").replace("attend_w1", "attend_w1c").replace("OKC_W1_", "OKC_W1C_")
    decl = decl.replace("attend_w1", "attend_w1c")
    return ('"""Generated by make_w1m.py (do not edit): W1-C, W1 with memory clobbers on its cp.async waits (see make_w1m.py)."""\n\n'
            f"CUDA_W1C_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_W1C_DECL = {decl!r}\n")


if __name__ == "__main__":
    texts = {"m": render_m(), "c": render_c()}
    if "--check" in sys.argv:
        bad = [k for k, t in texts.items() if not (os.path.exists(OUTS[k]) and open(OUTS[k]).read() == t)]
        print("cuda_w1m.py / cuda_w1c.py " + ("up to date" if not bad else f"STALE: {bad}"))
        sys.exit(1 if bad else 0)
    for k, t in texts.items():
        open(OUTS[k], "w").write(t)
        print(f"wrote {OUTS[k]} ({len(t)} bytes)")

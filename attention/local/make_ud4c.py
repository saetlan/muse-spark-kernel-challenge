"""Generates attention/local/cuda_ud4c.py: UD4c = UD4 (cuda_ud4.py) byte for byte in the kernel, with more
(warps, stages, cluster size) instantiations and an occupancy query. L34 tried only clusters of 8 and 16 and saw
larger clusters cost much more at Sd32 (cs16 40.4 vs cs8 24.6 us raw): cluster placement / waves are suspects.
UD4's split is already generic (nsplit = cs at run time; row owners need 16 / CS <= WARPS), so clusters of 4 and 2 only
need host dispatch, shared-memory attributes and the variant list.
Host: ud4c(... as ud4 ...), ud4c_setup(), ud4c_attrs(warps, stages, cs), ud4c_occ(warps, stages, cs, grid_y) ->
(max active clusters for that grid, max CTAs per SM).
Run: .venv/bin/python attention/local/make_ud4c.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_ud4c.py")
VARIANTS = [(4, 3, 8), (8, 2, 8), (4, 3, 16), (8, 2, 16), (8, 3, 8), (4, 3, 4), (4, 4, 4), (8, 2, 4), (8, 3, 4), (8, 3, 2), (8, 4, 2)]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:90]!r}"
    return src.replace(old, new)


def render(variants=None, multirow=False):
    VARIANTS_ = variants or VARIANTS
    text = open(os.path.join(HERE, "cuda_ud4.py")).read()
    for pat, rep in ((r"\bokc_ud4", "okc_ud4c"), (r"\bOKC_UD4_", "OKC_UD4C_"), (r"\bud4\(", "ud4c("), (r"\bud4_setup\b", "ud4c_setup"),
                     (r"\bud4_attrs\b", "ud4c_attrs"), (r"\bud4_kernel\b", "ud4c_kernel"), (r"\bCUDA_UD4_", "CUDA_UD4C_")):
        text = re.sub(pat, rep, text)
    for w, s, c in VARIANTS_:
        assert 16 % c == 0 and (multirow or 16 // c <= w), (w, s, c)
        # the kernel's own static_assert: merge scratch fits the ring
        assert w * 2 * 8 * 64 * 4 + 2 * w * 2 * 8 * 4 + 16 * (w + 2) * 4 <= w * s * 2 * 16 * 144, (w, s, c)
        assert w * s * 2 * 16 * 144 + c * 16 * 68 * 4 <= 227 * 1024
    # host dispatch
    a = text.index("  if (cs == 16) {\n    if (warps == 8) okc_ud4c::ud4c_kernel<8, 2, 16><<<")
    b = text.index("#undef OKC_UD4C_ARGS\n")
    disp = "  bool launched = false;\n"
    for w, s, c in VARIANTS_:
        disp += (f"  if (!launched && warps == {w} && stages == {s} && cs == {c}) {{\n"
                 f"    okc_ud4c::ud4c_kernel<{w}, {s}, {c}><<<grid, {w * 32}, smem, stream>>>(OKC_UD4C_ARGS);\n    launched = true;\n  }}\n")
    disp += '  TORCH_CHECK(launched, "ud4c: no variant (warps, stages, cs) = (", warps, ", ", stages, ", ", cs, ")");\n'
    text = text[:a] + disp + text[b:]
    # setup
    a = text.index("#define OKC_UD4C_SMEM(W, S, C)")
    b = text.index("  (void)cudaGetLastError();\n  return err;\n}\n", a)
    st = "#define OKC_UD4C_SMEM(W, S, C) \\\n  err |= cudaFuncSetAttribute(okc_ud4c::ud4c_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES + C * 16 * LBP * 4)\n"
    for w, s, c in VARIANTS_:
        st += f"  OKC_UD4C_SMEM({w}, {s}, {c});\n"
        if c > 8:
            st += f"  err |= cudaFuncSetAttribute(okc_ud4c::ud4c_kernel<{w}, {s}, {c}>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n"
    st += "#undef OKC_UD4C_SMEM\n"
    text = text[:a] + st + text[b:]
    # attrs + occupancy
    a = text.index("std::vector<int64_t> ud4c_attrs(int64_t warps, int64_t cs) {")
    b = text.index('"""', a)
    at = ("template <int W, int S, int C>\nstd::vector<int64_t> okc_ud4c_info(int64_t grid_y) {\n"
          "  cudaFuncAttributes a;\n  cudaFuncGetAttributes(&a, okc_ud4c::ud4c_kernel<W, S, C>);\n"
          "  const size_t smem = static_cast<size_t>(W) * S * okc_ud4c::STAGE_BYTES + static_cast<size_t>(C) * 16 * okc_ud4c::LBP * 4;\n"
          "  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(C, static_cast<unsigned>(grid_y), 1);\n  cfg.blockDim = dim3(W * 32, 1, 1);\n"
          "  cfg.dynamicSmemBytes = smem;\n  int clusters = -1, per_sm = -1;\n"
          "  if (cudaOccupancyMaxActiveClusters(&clusters, okc_ud4c::ud4c_kernel<W, S, C>, &cfg) != cudaSuccess) clusters = -1;\n"
          "  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, okc_ud4c::ud4c_kernel<W, S, C>, W * 32, smem) != cudaSuccess) per_sm = -1;\n"
          "  (void)cudaGetLastError();\n"
          "  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(smem), clusters, per_sm};\n}\n\n"
          "std::vector<int64_t> ud4c_attrs(int64_t warps, int64_t stages, int64_t cs, int64_t grid_y) {\n")
    for w, s, c in VARIANTS_:
        at += f"  if (warps == {w} && stages == {s} && cs == {c}) return okc_ud4c_info<{w}, {s}, {c}>(grid_y);\n"
    at += "  return {-1, -1, -1, -1, -1};\n}\n"
    text = text[:a] + at + text[b:]
    text = text.replace("std::vector<int64_t> ud4c_attrs(int64_t warps, int64_t cs);",
                        "std::vector<int64_t> ud4c_attrs(int64_t warps, int64_t stages, int64_t cs, int64_t grid_y);")
    assert "ud4c_attrs(int64_t warps, int64_t stages, int64_t cs, int64_t grid_y);" in text
    text = text.replace('"ud3 launch failed: "', '"ud4c launch failed: "')
    text += f"\nUD4C_VARIANTS = {VARIANTS_!r}\n"
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_ud4c.py from cuda_ud4.py (do not edit): UD4c, UD4 with clusters of 2 / 4 / 8 / 16 and an\n'
            'occupancy query (see make_ud4c.py). Original docstring follows.\n' + text[3:i] + text[i:])


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ud4c.py up to date" if same else "cuda_ud4c.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")

"""Generates attention/local/cuda_ver5t.py: V5t = V5 (cuda_ver5.py) with V re-laid once per stage by the
producer warp (A56, the user's data-movement point: V5's 4 consumer warps of a head each read the
same V rows with 8 scattered 8-byte loads and redo the same 24 byte-permutes per 32-key chunk).
Every piece is V5's text, transformed at anchors that must occur exactly once (asserted):
1. the producer's copies arrive on a new land_bar[stage] (count 32, cp.async noinc) instead of
   full_bar; full_bar now counts 1 arrival: the producer's lane 0 after the stage's V is re-laid;
2. transpose(j): wait land_bar, then each producer lane runs V5's own consumer V-fragment code for
   lane (g, t4) for both chunks and both heads (V5's text with the head variable renamed), and
   stores its 16 words as 4 uint4 at vt[stage][(c*2 + h)*4 + k][lane] (k-major: each 16 B store and
   load instruction touches 512 contiguous bytes, no bank conflicts); __syncwarp, lane 0 arrives
   on full_bar. The producer transposes stage i - (STAGES-1) after issuing stage i, so STAGES
   stages of copies stay in flight; the last STAGES-1 stages are transposed after the loop;
3. consumers load their fragments with 4 x 16 B loads per chunk instead of V5's 8 x 8 B + 4 tr4;
4. dynamic shared memory per stage += VT_BYTES (8 KB: 2 chunks x 2 heads x 4 x 32 x 16 B).
Run: .venv/bin/python attention/local/make_ver5t.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_ver5t.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var], ns[var.replace("_SRC", "_DECL")]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor found {n} times: {old[:90]!r}"
    return src.replace(old, new)


V_START = "      const uint8_t* vr = sv + (32 * c) * VP + 64 * h + 8 * g;\n"
V_END = "      vb1[c][4] = t[0]; vb1[c][5] = t[1]; vb1[c][6] = t[2]; vb1[c][7] = t[3];\n"


def render():
    src, decl = load("cuda_ver5.py", "CUDA_VER5_SRC")
    src = src.replace("okc_ver5", "okc_ver5t").replace("attend_ver5", "attend_ver5t").replace("OKC_VER5", "OKC_VER5T")
    decl = decl.replace("attend_ver5", "attend_ver5t")
    # V5's consumer V-fragment code (verified), reused verbatim by the producer's transpose
    i, j = src.index(V_START), src.index(V_END)
    assert src.count(V_START) == 1 and src.count(V_END) == 1 and i < j
    vblock = src[i:j + len(V_END)]
    tblock = (vblock.replace("64 * h + 8 * g", "64 * hv + 8 * g").replace("vb0[c][", "w0[").replace("vb1[c][", "w1[")
              .replace("\n      ", "\n          "))
    tblock = "    " + tblock
    assert "vb0" not in tblock and "vb1" not in tblock and "64 * h " not in tblock
    src = sub1(src, "constexpr float P_OFF = 8.0f;",
               "constexpr float P_OFF = 8.0f;\nconstexpr int VT_BYTES = 2 * 2 * 4 * 32 * 16;   // V5t: re-laid V per stage", "VT_BYTES")
    src = sub1(src, "  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];\n",
               "  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES], land_bar[STAGES];\n"
               "  uint8_t* vtb = ring + STAGES * STAGE_BYTES;          // V5t: V tiles in fragment order, VT_BYTES per stage\n",
               "barriers")
    src = sub1(src, "      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 32;\" ::\"r\"(smem_u32(&full_bar[s])));   // producer lanes\n",
               "      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 32;\" ::\"r\"(smem_u32(&land_bar[s])));   // producer lanes' copies\n"
               "      asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 1;\" ::\"r\"(smem_u32(&full_bar[s])));    // V re-laid (producer lane 0)\n",
               "init")
    src = sub1(src, "asm volatile(\"cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\" ::\"r\"(smem_u32(&full_bar[st])) : \"memory\");",
               "asm volatile(\"cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\" ::\"r\"(smem_u32(&land_bar[st])) : \"memory\");",
               "land arrive")
    transpose = (
        "    // V5t: stage j landed -> its V tile in MMA B-fragment order (each lane runs V5's consumer code for\n"
        "    // lane (g, t4), both chunks, both heads) -> hand the stage to the consumers\n"
        "    auto transpose = [&](int j) {\n"
        "      const int sj = j % STAGES;\n"
        "      const uint32_t lpar = static_cast<uint32_t>((j / STAGES) & 1);\n"
        "      uint32_t landed = 0;\n"
        "      while (!landed)\n"
        "        asm volatile(\"{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}\"\n"
        "                     : \"=r\"(landed) : \"r\"(smem_u32(&land_bar[sj])), \"r\"(lpar) : \"memory\");\n"
        "      const uint8_t* sv = ring + sj * STAGE_BYTES + SKEYS * KP;\n"
        "      uint4* vt = reinterpret_cast<uint4*>(vtb + sj * VT_BYTES);\n"
        "#pragma unroll\n"
        "      for (int c = 0; c < 2; ++c)\n"
        "#pragma unroll\n"
        "        for (int hv = 0; hv < 2; ++hv) {\n"
        "          uint32_t w0[8], w1[8];\n"
        + tblock +
        "          uint4* dst = vt + ((c * 2 + hv) * 4) * 32 + lane;\n"
        "          dst[0] = make_uint4(w0[0], w0[1], w0[2], w0[3]);\n"
        "          dst[32] = make_uint4(w0[4], w0[5], w0[6], w0[7]);\n"
        "          dst[64] = make_uint4(w1[0], w1[1], w1[2], w1[3]);\n"
        "          dst[96] = make_uint4(w1[4], w1[5], w1[6], w1[7]);\n"
        "        }\n"
        "      __syncwarp();\n"
        "      if (lane == 0) {\n"
        "        uint64_t state;\n"
        "        asm volatile(\"mbarrier.arrive.shared::cta.b64 %0, [%1];\" : \"=l\"(state) : \"r\"(smem_u32(&full_bar[sj])) : \"memory\");\n"
        "        (void)state;\n"
        "      }\n"
        "    };\n")
    src = sub1(src, "    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip\n",
               transpose + "    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip\n", "transpose def")
    src = sub1(src, "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n      __syncwarp();\n    }\n"
                    "    asm volatile(\"cp.async.wait_all;\");\n",
               "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n      __syncwarp();\n"
               "      if (i >= STAGES - 1) transpose(i - (STAGES - 1));   // STAGES stages of copies stay in flight\n    }\n"
               "    for (int j = max(nst - (STAGES - 1), 0); j < nst; ++j) transpose(j);\n"
               "    asm volatile(\"cp.async.wait_all;\");\n", "transpose calls")
    src = sub1(src, vblock,
               "      const uint4* vt4 = reinterpret_cast<const uint4*>(vtb + st * VT_BYTES) + ((c * 2 + h) * 4) * 32 + lane;   // V5t\n"
               "      const uint4 v0 = vt4[0], v1 = vt4[32], v2 = vt4[64], v3 = vt4[96];\n"
               "      vb0[c][0] = v0.x; vb0[c][1] = v0.y; vb0[c][2] = v0.z; vb0[c][3] = v0.w;\n"
               "      vb0[c][4] = v1.x; vb0[c][5] = v1.y; vb0[c][6] = v1.z; vb0[c][7] = v1.w;\n"
               "      vb1[c][0] = v2.x; vb1[c][1] = v2.y; vb1[c][2] = v2.z; vb1[c][3] = v2.w;\n"
               "      vb1[c][4] = v3.x; vb1[c][5] = v3.y; vb1[c][6] = v3.z; vb1[c][7] = v3.w;\n", "consumer V loads")
    src = sub1(src, "const size_t smem = static_cast<size_t>(stages) * okc_ver5t::STAGE_BYTES;",
               "const size_t smem = static_cast<size_t>(stages) * (okc_ver5t::STAGE_BYTES + okc_ver5t::VT_BYTES);", "host smem")
    src = sub1(src, "cudaFuncAttributeMaxDynamicSharedMemorySize, S * okc_ver5t::STAGE_BYTES);",
               "cudaFuncAttributeMaxDynamicSharedMemorySize, S * (okc_ver5t::STAGE_BYTES + okc_ver5t::VT_BYTES));", "setup smem")
    assert "okc_ver5::" not in src and "attend_ver5_" not in src.replace("attend_ver5t", "")
    return ('"""Generated by make_ver5t.py (do not edit): V5t, V5 with V re-laid once per stage by the producer '
            '(see make_ver5t.py)."""\n\n'
            f"CUDA_VER5T_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_VER5T_DECL = {decl!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ver5t.py up to date" if same else "cuda_ver5t.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")

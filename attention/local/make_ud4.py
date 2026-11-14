"""Generates attention/local/cuda_ud4.py: UD4 = UD3 (cuda_ud3.py) with L33's anatomy fixes (AF59), by asserted
text substitutions (everything else byte for byte):
1. row-owner finalize: CTA `rank` of the cluster finalizes output rows hr = rank + CS*k (k < 16/CS: 1 row at CS 16,
   2 at CS 8), one warp per row, instead of the leader merging all 16 rows with 4 rows per warp in sequence (1.86 us
   at Sd1). Each CTA pushes its partial of row hr (64 outputs, m, l) to slot [rank][hr] of the owner hr % CS's buffer
   (the leader buffer's layout, present in every CTA); one cluster barrier; owners read only their own buffer.
   Gates: each owner loads the gate rows it finalizes at entry (UD3: the leader loaded all 16).
2. the new token's K / V values (warp 0: 4 bf16 of k and of v per lane) are loaded at entry with seqlens, so the CTA
   holding position n-1 no longer pays a dependent qkv trip before its fold (L33: fold 2.0 us there vs 0.78 median).
Host: ud4(..., warps, stages, cs) as ud3; ud4_setup(), ud4_attrs(warps, cs).
Run: .venv/bin/python attention/local/make_ud4.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_ud4.py")


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:90]!r}"
    return src.replace(old, new)


def render():
    text = open(os.path.join(HERE, "cuda_ud3.py")).read()
    for pat, rep in ((r"\bokc_ud3", "okc_ud4"), (r"\bOKC_UD3_", "OKC_UD4_"), (r"\bud3\(", "ud4("), (r"\bud3_setup\b", "ud4_setup"),
                     (r"\bud3_attrs\b", "ud4_attrs"), (r"\bud3_kernel\b", "ud4_kernel"), (r"\bCUDA_UD3_", "CUDA_UD4_")):
        text = re.sub(pat, rep, text)
    # 2. new row's qkv values at entry (warp 0), next to the other seqlens-independent loads
    text = sub1(text, "  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n",
                "  uint2 nk_raw, nv_raw;                                 // UD4: the new token's k / v (4 bf16 each)\n  nk_raw.x = nk_raw.y = nv_raw.x = nv_raw.y = 0u;\n"
                "  if (warp == 0) {\n"
                "    const int h_ = lane >> 4, d0_ = 4 * (lane & 15);\n"
                "    nk_raw = *reinterpret_cast<const uint2*>(qkv_b + static_cast<long>(hq) * 64 + (kv0 + h_) * 64 + d0_);\n"
                "    nv_raw = *reinterpret_cast<const uint2*>(qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h_) * 64 + d0_);\n"
                "  }\n"
                "  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n", "new row prefetch")
    text = sub1(text, "      k4[j] = bf16f(kp[j]);\n      v4[j] = bf16f(vp[j]);\n",
                "      const uint32_t kw_ = (j >> 1) ? nk_raw.y : nk_raw.x, vw_ = (j >> 1) ? nv_raw.y : nv_raw.x;\n"
                "      k4[j] = bf16f(static_cast<uint16_t>((j & 1) ? (kw_ >> 16) : (kw_ & 0xffffu)));\n"
                "      v4[j] = bf16f(static_cast<uint16_t>((j & 1) ? (vw_ >> 16) : (vw_ & 0xffffu)));\n", "new row from prefetch")
    text = sub1(text, "    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;\n"
                      "    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;\n",
                "    (void)h;\n    (void)d0;\n", "drop new row pointers")
    # 1a. gates per owner
    text = sub1(text, """  float gate0[RPW], gate1[RPW];                         // leader: gate values of its output rows
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    gate0[k2] = gate1[k2] = 0.f;
    if (rank == 0) {
      const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + (kv0 * 8 + warp + k2 * WARPS) * 64;
      gate0[k2] = bf16f(gp[lane]);
      gate1[k2] = bf16f(gp[32 + lane]);
    }
  }
""", """  constexpr int ROWS = 16 / CS;                         // UD4: output rows this CTA finalizes: hr = rank + CS * warp
  static_assert(16 % CS == 0 && ROWS <= WARPS, "row owners: one warp per owned row");
  float gate0 = 0.f, gate1 = 0.f;                       // gate values of this warp's owned row
  if (warp < ROWS) {
    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + (kv0 * 8 + static_cast<int>(rank) + CS * warp) * 64;
    gate0 = bf16f(gp[lane]);
    gate1 = bf16f(gp[32 + lane]);
  }
""", "owner gates")
    # 1b. push each row's partial to its owner
    text = sub1(text, """  const uint32_t lead = map_rank(smem_u32(lbuf), 0) + rank * (16 * LBP * 4);
  for (int idx = threadIdx.x; idx < 16 * 64; idx += WARPS * 32) {
    const int hr = idx >> 6, d = idx & 63, h = hr >> 3, row = hr & 7;
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) o += sm_w[hr * WARPS + w] * sm_o[w][h][row][d];
    st_cluster(lead + (hr * LBP + d) * 4, o);
  }
  if (threadIdx.x < 16) {
    st_cluster(lead + (threadIdx.x * LBP + 64) * 4, sm_mm[threadIdx.x]);
    st_cluster(lead + (threadIdx.x * LBP + 65) * 4, sm_lt[threadIdx.x]);
  }
  cluster_sync();                                   // every partial in the leader's buffer
  if (rank != 0) return;
""", """  const uint32_t slot0 = smem_u32(lbuf) + rank * (16 * LBP * 4);   // UD4: slot [rank][hr] in the owner hr % CS's buffer
  for (int idx = threadIdx.x; idx < 16 * 64; idx += WARPS * 32) {
    const int hr = idx >> 6, d = idx & 63, h = hr >> 3, row = hr & 7;
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) o += sm_w[hr * WARPS + w] * sm_o[w][h][row][d];
    st_cluster(map_rank(slot0 + (hr * LBP + d) * 4, static_cast<uint32_t>(hr % CS)), o);
  }
  if (threadIdx.x < 16) {
    const uint32_t owner = static_cast<uint32_t>(threadIdx.x % CS);
    st_cluster(map_rank(slot0 + (threadIdx.x * LBP + 64) * 4, owner), sm_mm[threadIdx.x]);
    st_cluster(map_rank(slot0 + (threadIdx.x * LBP + 65) * 4, owner), sm_lt[threadIdx.x]);
  }
  cluster_sync();                                   // every partial in its owner's buffer
  if (warp >= ROWS) return;
""", "push to owners")
    # 1c. finalize: one warp per owned row
    text = sub1(text, """  // ---- 9. leader: merge the CS partials from its own buffer, RMSNorm x sigmoid(gate), BF16 out
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    const int hr = warp + k2 * WARPS;
""", """  // ---- 9. UD4 row owner: warp w merges row hr = rank + CS * w from its own buffer, RMSNorm x sigmoid(gate), BF16 out
  {
    const int hr = static_cast<int>(rank) + CS * warp;
""", "owner finalize")
    text = sub1(text, "    op[lane] = f2bf16(o0 * rn / (1.f + __expf(-gate0[k2])));\n    op[32 + lane] = f2bf16(o1 * rn / (1.f + __expf(-gate1[k2])));\n",
                "    op[lane] = f2bf16(o0 * rn / (1.f + __expf(-gate0)));\n    op[32 + lane] = f2bf16(o1 * rn / (1.f + __expf(-gate1)));\n",
                "owner gates use")
    assert "rank == 0" not in text and "gate0[" not in text
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_ud4.py from cuda_ud3.py (do not edit): UD4, UD3 with row-owner finalize and the new row\n'
            'prefetched (AF59). Original docstring follows.\n' + text[3:i] + text[i:])


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ud4.py up to date" if same else "cuda_ud4.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")

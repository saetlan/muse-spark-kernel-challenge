"""CPU emulation of dgrs.py (no GPU here), with bf16_proj's own F12 encoder (read-only: f12m_encode from
proj/f12m/f12m_layout.py and the F12V helpers from proj/submission_v10.py, pulled out by AST).

1. Pack + decode: dgrs_pack's chunk-major planes read at the kernel's offsets (stage block of row block rb and chunk
   kc, row tile 4c + w, slots +512 / +1024, lane * 16), decoded with a port of f12m_decode, placed at wgmma RS's A
   positions (row 16 rt + g + 8 (reg & 1), k 16 step + 2 t + 8 (reg >> 1) + half) == W * 2^(112 - base) exactly,
   escapes = 2^-15.
2. x tile: the producer's swizzled cp.async destinations read back through the SW128 descriptor rule == x.
3. Whole GEMM per layer (qkv, o, pooler, unpooler at m = 32 / 128 / 256 with the split counts below): per CTA slice
   fp32 partials, last arriver sums slices in order, F12M epilogue (scale, escapes from rec, x from global), BF16;
   vs FP32 x @ W^T: relative error and worst row.
4. v3 split-K finalize map: every (row, token) exactly once over cluster ranks and threads, S = 1..8.
5. Pins: the CUDA lines these ports mirror.

Run: .venv/bin/python shared/gemm/dg12rs/emu_dgrs.py [quick]
"""

import ast
import re
import random
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"

# ---- bf16_proj's encoder, read-only
f12 = {"torch": torch, "M32": 0xFFFFFFFF}
tree = ast.parse((ROOT / "proj" / "submission_v10.py").read_text())
want = {"f12v_bases", "f12v_words", "as_int32", "f12v_records"}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module([node], []), "submission_v10.py", "exec"), f12)
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("F12V_SLOTS", "M32"):
        exec(compile(ast.Module([node], []), "submission_v10.py", "exec"), f12)
assert want <= set(f12), set(f12)
lay = (ROOT / "proj" / "f12m" / "f12m_layout.py").read_text()
lay = lay.replace("from f12v_layout import as_int32, f12v_bases, f12v_records, f12v_words\n", "")
lay = lay.replace("from p9_layout import k_perm, p9_pack\n", "")
exec(compile(lay, "f12m_layout.py", "exec"), f12)

sys.modules.setdefault("torch.utils.cpp_extension", types.SimpleNamespace(load_inline=None))
dg = {"__name__": "dgrs_emu"}
exec(compile((HERE / "dgrs.py").read_text().replace("from torch.utils.cpp_extension import load_inline",
                                                     "load_inline = None"), "dgrs.py", "exec"), dg)
CUDA = dg["dgrs_source"]()

# ---------------------------------------------------------------------------------------------- 5. pins
PINS = [
    "  const uint32_t r4 = __funnelshift_l(q, q, 4);",
    "  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;",
    "  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;",
    "  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),",
    "                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));",
    "  const uint4 l01 = *reinterpret_cast<const uint4*>(sw + woff + 512);",
    "  const uint4 l23 = *reinterpret_cast<const uint4*>(sw + woff + 1024);",
    "    const uint4& l = j < 2 ? l01 : l23;",
    "    const uint4 r = f12m_decode(word(q, j), word(l, 2 * (j & 1)), word(l, 2 * (j & 1) + 1));",
    "    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024",
    "    const uint8_t* wsrc = wp + (static_cast<size_t>(kc0) * (N / 16) + static_cast<size_t>(rb) * 8) * 1536;",
    "    const size_t wstride = static_cast<size_t>(N / 16) * 1536;  // one chunk of all rows",
    "        if (!(DIAG & 4)) tma_2d(smem_u32(sX + s * X_BYTES), &xmap, (kc0 + i) * 64, tt * BM, full0 + 8 * s);",
    "  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(bm)};",
    "  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};",
    "  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |",
    "         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);",
    "    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \\",
    "  const int per = (KC + splits - 1) / splits;",
    "  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)",
    "      const int r0 = 64 * c + 16 * w + g;  // rows r0, r0 + 8; tokens 8j + 2q (+1)",
    "        part[(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N + rb * 128 + (idx & 127)] = tile[idx];",
    "        const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;",
    "          atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again",
    "                   : glob ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row)",
    "      for (int t0 = tl; t0 < BM; t0 += TL * FU) {",
    "          const int t = t0 + TL * u;",
    "            const uint32_t a = tile0 + static_cast<uint32_t>(((t0 + TL * u) * 128 + rlo + ri) * 4);",
    "__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }",
    "__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }",
    "      mbar_init(full0 + 8 * s, 1);  // thread 0's expect_tx arrive (bytes: the weight copy + the x TMA)",
    "  constexpr uint32_t TX = ((DIAG & 8) ? 0u : W_CHUNK) + ((DIAG & 4) ? 0u : X_BYTES);",
    "      if (i >= NP && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - NP) % S));",
    "      if (i + NP < nk) load(i + NP, nxt);",
    "    const int mw = (tid >> 7) - 1, c = mw & 1, p = mw >> 1;",
    "      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == p);",
    '          mtile[t * 128 + r0] = acc[4 * j + h];',
    '          mtile[t * 128 + r0 + 8] = acc[4 * j + 2 + h];',
    '      float* mtile = tile + p * (BM * 128);',
    '        for (int idx = mt; idx < BM * 128; idx += TM) tile[idx] += tile[BM * 128 + idx];',
    '  constexpr int TM = 128 * MATH;   // math threads: the epilogue is theirs alone',
    '  constexpr int FU = 16 / MATH;    // tokens per finalize step',
    '      const int mt = tid - 128;',
    '  const int rank = glob ? 0 : sl;',
    '  const int rlo = rank * L.rs;',
    '  const int rhi = rlo + L.rs < 128 ? rlo + L.rs : 128;',
    '      const int TL = rn > 0 ? TM / rn : 0;',
    '      if (rn > 0 && mt < TL * rn && (!glob || s_last)) {',
    '        const int ri = mt % rn, tl = mt / rn;',
    '  L.rs = (128 + nr - 1) / nr;',
    '  const int stage = bm * 128 + W_CHUNK;',
    '  const int corr = (bm * L.cs * 4 + 15) / 16 * 16;',
    '  if (math == 4) S &= ~1;  // even: slot s only ever holds chunks of parity s % 2, i.e. of pair s % 2',
    '  L.bars = S * stage;',
    '  L.scale = L.bars + 16 * SMAX;',
    '  L.esc = L.scale + 128 * 4;',
    'constexpr int SMEM_DYN1 = 226 * 1024;  // dynamic shared memory per CTA at 1 CTA/SM (227 KB opt-in max, F47)',
    'constexpr int SMEM_DYN2 = 112 * 1024;  // per CTA at 2 CTAs/SM (228 KB per SM, 1 KB reserved per CTA)',
    'constexpr int PRE_WARPS = 3;           // producer warps 1-3 precompute the finalize rows (pre_corr)',
    'constexpr int ESC_CAP = 304;           // escapes listed per warp: ceil(128 / 3) rows x 7 = 301',
    '  const int RW = (rn + PRE_WARPS - 1) / PRE_WARPS;',
    '  const int wlo = pw * RW;',
    '  const int whi = wlo + RW < rn ? wlo + RW : rn;',
    '  uint2* list = sEsc + pw * ESC_CAP;',
    '    for (int ri = wlo + lane; ri < whi; ri += 32) sCorr[t * cs + ri] = 0.f;',
    '    const uint4 q0 = v ? rec[2 * (row0 + ri)] : make_uint4(0u, 0u, 0u, 0u);',
    '    const uint32_t base = q0.x & 0xFFu;',
    '    uint32_t cnt = (q0.x >> 8) & 0xFFu;',
    '    cnt = cnt < 7u ? cnt : 7u;',
    '    const uint4 q1 = cnt > 3u ? rec[2 * (row0 + ri) + 1] : make_uint4(0u, 0u, 0u, 0u);',
    '    if (v) sScale[ri] = f12m_scale(base);',
    '    for (int o = 1; o < 32; o <<= 1) {',
    '      if (lane >= o) inc += nb;',
    '    const int off = ne + inc - static_cast<int>(cnt);',
    '    const uint32_t tag = static_cast<uint32_t>(ri) | (base << 8);',
    '      if (e < static_cast<int>(cnt)) list[off + e] = make_uint2(e < 3 ? word(q0, e + 1) : word(q1, e - 3), tag);',
    '    ne += __shfl_sync(0xFFFFFFFFu, inc, 31);',
    '  for (int v2 = 0; v2 < BM / 32; ++v2) {',
    '    const int t = lane + 32 * v2;',
    '    const bool live = tok0 + t < M;',
    '    } else if (tid >= 32 && (DIAG & 49) == 0) {',
    '          const int ns = push ? splits * NP : splits;',
    '          for (int s2 = 0; s2 < ns; ++s2) {  // slice (then pair) order: deterministic',
    '                     : push ? sRecv[(s2 * BM + t0 + TL * u) * L.rsp + ri]',
    '  const int nr = (red == 1 && splits > 1) ? 1 : splits;',
    '  L.push = red == 2 && splits > 1;',
    '  L.rsp = L.rs + ((4 - L.rs) & 7);',
    '  const int fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8 + PRE_WARPS * 2 * GATHER * 4;',
    '  const int recv = L.push ? splits * (math / 2) * bm * L.rsp * 4 : 0;',
    '  const int avail = (cps == 1 ? SMEM_DYN1 : SMEM_DYN2) - 1024 - fixed - corr - recv;',
    '  L.stage = L.esc + PRE_WARPS * ESC_CAP * 8;',
    '  L.corr = L.stage + PRE_WARPS * 2 * GATHER * 4;',
    '  L.recv = L.corr + corr;',
    '  L.end = L.recv + recv;',
    '  return L.S >= 4 && (L.push || (math / 2) * bm * 128 * 4 <= L.bars);',
    'constexpr int GATHER = 512;            // pre_corr gather buffer (u32 words): GATHER / BM list entries x BM tokens',
    '  constexpr int G = GATHER / BM;',
    '  uint32_t* stg = sStage + pw * (2 * GATHER);',
    '      const uint32_t k2 = list[e].x & 0xFFFEu;  // the 4-byte word holding element k',
    '        cp_async4(stg0 + static_cast<uint32_t>(((buf * G + e - b) * BM + t) * 4),',
    '                  xh + static_cast<size_t>(live ? tok0 + t : 0) * K + k2, live ? 4u : 0u);',
    '      issue(b + G, buf ^ 1);',
    "      cp_async_wait<1>();  // batch b landed (this lane's own words)",
    '        const uint32_t wv = stg[(buf * G + e - b) * BM + t];',
    '        const float xv = __uint_as_float((d.x & 1u) ? (wv & 0xFFFF0000u) : (wv << 16));',
    '        float* cp = sCorr + t * cs + (d.y & 0xFFu);',
    '        c = fmaf(-xv, wdec, c);',
    '        c = fmaf(xv, wtrue, c);',
    '      const float wdec = f12m_wdec(d.y >> 8), wtrue = __uint_as_float(d.x & 0xFFFF0000u);',
    '      pre_corr<BM>(rec, reinterpret_cast<const uint16_t*>(x), sScale, sEsc, sStage, sCorr, rb * 128 + rlo, rn, L.cs,',
    '                   tt * BM, M, K, (tid >> 5) - 1, tid & 31);',
    '  const bool glob = red == 1 && splits > 1;',
    '        const int ra = r0 / L.rs, rb8 = (r0 + 8) / L.rs;',
    '        const int slot = (sl * NP + p) * BM * L.rsp;',
    '        const uint32_t ga = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 - ra * L.rs) * 4), static_cast<uint32_t>(ra));',
    '        const uint32_t gb = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 + 8 - rb8 * L.rs) * 4), static_cast<uint32_t>(rb8));',
    '            const uint32_t t = static_cast<uint32_t>(8 * j + 2 * q + h);',
    '  uint32_t* stamps = reinterpret_cast<uint32_t*>(part) +',
    '                     16 * ((static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);',
    '  constexpr bool STAMP = (DIAG & 64) != 0;',
    '  TORCH_CHECK(!(diag & 64) || (!(red == 1 && splits > 1) && part.numel() >= 16 * (N / 128) * tiles * splits),',
    '  cfg.numAttrs = (splits > 1 && red != 1) ? 1 : 0;  // a real cluster only for DSMEM split-K (PF41: size 1 costs ~30%)',
    '            st_cluster_f32_if(ga + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + h], real);',
    '            st_cluster_f32_if(gb + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + 2 + h], real);',
    '            if (live[u] && real)',
    '    if (real && p < nk) {',
    '        if (real) cl_wait();  // phase A: every CTA of the cluster is running (its receive area exists)',
    '      } else if (real) {',
    "        bar_sync_na(2, THREADS);  // the producer warps' scales and corrections",
    "        if (splits > 1 && red == 0) cluster_sync();  // peers may still read this CTA's tile",
    '        asm volatile("" : "+r"(pass));  // opaque: one copy of the body, executed twice',
    '        body(pass == 1);',
    '      body(true);',
    '  TORCH_CHECK(!(diag & 256) || (red == 2 && splits > 1), "dgrs_gemm: DIAG 256 (epilogue warm-up) needs red 2, splits > 1");',
    '        const float scale = sScale[RING ? rlo + ri : ri];',
    '                  __float2bfloat16_rn(fmaf(sum[u], scale, RING ? 0.f : sCorr[(t0 + TL * u) * L.cs + ri]));',
    '  L.cs = ring ? 129 : L.rs + 1;  // ring corrections (DIAG 512) cover all 128 rows of the slice',
    '      mbar_init(empty0 + 8 * s, RING ? 2 + PRE_WARPS : 2);  // both math warpgroups (+ the producer warps 1-3, DIAG 512)',
    '      ok &= pre_ring<BM>(rec, sScale, sEsc, sStage, sCorr, sX, rb * 128, L.cs, kc0, nk, S, full0, empty0, err,',
    '                         (tid >> 5) - 1, tid & 31);',
    '        if (p == 0) {             // fold once per CTA: acc += corr * 2^(112 - base) (exact power-of-two reciprocal)',
    '          const int rf = 64 * c + 16 * w + g;',
    '          const float i0 = __frcp_rn(sScale[rf]), i8 = __frcp_rn(sScale[rf + 8]);',
    '              acc[4 * j + h] = fmaf(sCorr[t * L.cs + rf], i0, acc[4 * j + h]);',
    '              acc[4 * j + 2 + h] = fmaf(sCorr[t * L.cs + rf + 8], i8, acc[4 * j + 2 + h]);',
    '      if (real && !RING) {',
    '  constexpr int RWR = (128 + PRE_WARPS - 1) / PRE_WARPS;  // 43',
    '  const int wlo = pw * RWR;',
    '  const int whi = wlo + RWR < 128 ? wlo + RWR : 128;',
    '  uint2* buck = reinterpret_cast<uint2*>(sStage + pw * (2 * GATHER));',
    '  int* ends = reinterpret_cast<int*>(buck + ESC_CAP);',
    '  for (int c = lane; c < nk; c += 32) ends[c] = 0;',
    '      const int ch = static_cast<int>(((e < 3 ? word(q0, e + 1) : word(q1, e - 3)) & 0xFFFFu) >> 6) - kc0;',
    '      if (e < static_cast<int>(cnt) && ch >= 0 && ch < nk) ++m;',
    '    int off = ne + inc - m;',
    '      const int ch = static_cast<int>((wd & 0xFFFFu) >> 6) - kc0;',
    '      if (e < static_cast<int>(cnt) && ch >= 0 && ch < nk) list[off++] = make_uint2(wd, tag);',
    '    const int ch = e < ne ? static_cast<int>((list[e].x & 0xFFFFu) >> 6) - kc0 : -1;',
    '    const unsigned grp = __match_any_sync(0xFFFFFFFFu, ch);',
    '    if (ch >= 0 && lane == __ffs(grp) - 1) ends[ch] += __popc(grp);',
    '      const int c = 8 * lane + u;',
    '      if (c < nk) ends[c] = run;  // start of bucket c (the scatter turns it into the end)',
    '    const int pos = ch >= 0 ? ends[ch] + __popc(grp & ((1u << lane) - 1u)) : 0;',
    '      buck[pos] = d;',
    '      if (lane == __ffs(grp) - 1) ends[ch] += __popc(grp);',
    '    ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);',
    '    const int lo = i > 0 ? ends[i - 1] : 0, hi = ends[i];',
    '    const int k0 = (kc0 + i) * 64;',
    '      const int kk = static_cast<int>(d.x & 0xFFFFu) - k0;',
    '        const uint16_t hx = *reinterpret_cast<const uint16_t*>(xs + t * 128 + ((((kk >> 3) ^ (t & 7)) & 7) << 4) + (kk & 7) * 2);',
    "    if (lane == 0) mbar_arrive(empty0 + 8 * s);  // this warp is done with chunk i's slot",
    '  TORCH_CHECK(!(diag & 512) || (per <= 256 && !(diag & 32)), "dgrs_gemm: DIAG 512 needs <= 256 chunks per slice, no DIAG 32");',
    'template <int BM, int CPS, int DIAG, int MATH, int RED = -1, int SPL = 0>',
    '  if constexpr (SPL > 0) splits = SPL;  // dgrs_gemm picks this instance only when splits == SPL',
    '  if constexpr (RED >= 0) red = RED;    // ... and red == RED',
    '  const int rn = (128 % L.rs == 0) ? L.rs : (rhi > rlo ? rhi - rlo : 0);  // equal shares fold to a constant',
    '  if (bm == BM && cps == CPS && diag == DIAG && math == MATH && (RED < 0 || red == RED) &&                    \\',
    '      (SPL == 0 || splits == SPL)) {                                                                          \\',
    '    dgrs_launch<BM, CPS, DIAG, MATH, RED, SPL>(grid, static_cast<int>(splits), static_cast<int>(red), xmap, w, rc, xb, o,      \\',
]
missing = [p for p in PINS if CUDA.count(p + "\n") < 1]
assert not missing, f"dgrs.py's CUDA changed; update the ports: {missing[:3]}"
print(f"ok 5: {len(PINS)} CUDA lines mirrored by the ports are unchanged")


def barrier_match():
    """No-timeout barriers must match (a mismatch hangs the run): in rs_kernel's epilogue (DIAG & 17 == 0) the producer
    warpgroup (128 threads) arrives once on named barrier 2 and passes two cluster barriers when splits > 1 && !glob;
    the math warpgroups (TM threads) sync once on barrier 2 and pass the same two cluster barriers (after the tile and
    after the finalize), and nothing else uses barrier 2 or cluster barriers. 128 + TM = THREADS."""
    k = CUDA[CUDA.index("__global__ void __launch_bounds__(128 + 128 * MATH, CPS) rs_kernel"):]
    k = k[:k.index("\n}\n") + 3]
    prod = k[k.index("  if (tid < 128) {"):k.index("  } else {\n    if (MATH == 4) setmaxnreg_inc<112>();")]
    math = k[k.index("  } else {\n    if (MATH == 4) setmaxnreg_inc<112>();"):]
    pe = prod[prod.index("    if constexpr ((DIAG & 17) == 0) {"):]
    me = math[math.index("    } else if constexpr ((DIAG & 1) == 0) {"):]
    assert pe.count("bar_arrive_na(2, THREADS);") == 1 and prod.count("bar_arrive_na(") == 1, "producer: one arrive"
    assert prod.count("bar_sync_na(") == 0 and prod.count("named_sync(") == 0, "producer: no blocking CTA barrier"
    # math syncs barrier 2 once per launch: before the fold (DIAG 512) or before the finalize (otherwise)
    assert math.count("bar_sync_na(2, THREADS);") == 2 and math.count("bar_sync_na(") == 2, "math: barrier 2 sites"
    assert "    if constexpr (RING && (DIAG & 17) == 0) {\n      if (real) {\n        bar_sync_na(2, THREADS);" in math
    assert "      if (real && !RING) {\n        bar_sync_na(2, THREADS);" in me
    assert math.count("bar_arrive_na(") == 0
    # red 0 (splits > 1): two cluster_sync per role
    assert pe.count("cluster_sync();") == 2 and prod.count("cluster_sync();") == 2
    assert "      } else if (splits > 1 && !glob) {  // red 0: the math warpgroups' two cluster barriers\n" in pe
    assert me.count("cluster_sync();") == 2 and math.count("cluster_sync();") == 2
    assert "        } else if (splits > 1) {\n          cluster_sync();" in me   # inside `else` (not push), not glob
    assert "        if (splits > 1 && red == 0) cluster_sync();  // peers may still read this CTA's tile\n" in me
    # red 2 (push): every thread arrives on phase A once at the start (same condition), then wait A, arrive B, wait B
    assert k.count("cl_arrive_relaxed();") == 1
    assert "  if ((DIAG & 17) == 0 && push) cl_arrive_relaxed();  // red 2 phase A" in k
    head = k[:k.index("  if (tid < 128) {")]
    assert "cl_arrive_relaxed();" in head and head.index("__syncthreads();") < head.index("cl_arrive_relaxed();")
    pp = pe[pe.index("      if (push) {"):pe.index("      } else if (splits > 1 && !glob)")]
    assert [x for x in re.findall(r"cl_(arrive|wait)\(\)", pp)] == ["wait", "arrive", "wait"], pp
    mp = me[me.index("      if (push) {"):me.index("      } else if (real) {\n        // red 0 / 1 / S = 1")]
    assert [x for x in re.findall(r"cl_(arrive|wait)\(\)", mp)] == ["wait", "arrive", "wait"], mp
    assert prod.count("cl_") == 3 and math.count("cl_") == 3, "no other split cluster barrier"
    ma, mb = mp.index("cl_wait();"), mp.rindex("cl_arrive();")
    assert ma < mp.index("st_cluster_f32_if(") < mb, "pushes between phase A's wait and phase B's arrive"
    assert "  constexpr int THREADS = 128 + 128 * MATH;\n" in k and "  constexpr int TM = 128 * MATH;" in k
    assert k.count("named_sync(1, TM);") == 5 and "named_sync(1, 128" not in k   # barrier 1: math threads only
    assert k.count("__syncthreads();") == 1   # the barrier-init sync, before the role split
    return True


barrier_match()


def spec_checks():
    """v9 specializations: the folded row share equals the runtime one for every S / red / rank; the dispatcher lists
    specialized instances before generic ones (first match wins) and only matches equal red / splits."""
    for splits in range(1, 9):
        for red in (0, 1, 2):
            L = layout(32, 1, 4, splits, red, 8)
            nr = 1 if (red == 1 and splits > 1) else splits
            for rank in range(nr):
                rlo = rank * L["rs"]
                rhi = min(128, rlo + L["rs"])
                runtime = max(0, rhi - rlo)
                folded = L["rs"] if 128 % L["rs"] == 0 else runtime
                assert folded == runtime, (splits, red, rank)
    src = dg["dgrs_source"]([(32, 1, 0, 4), (32, 1, 0, 4, 2, 2), (64, 1, 512, 4, 0, 3)])
    cases = re.findall(r"DGRS_CASE\(([^)]*)\)", src.split("@@")[0] if "@@" in src else src)
    cases = [c for c in cases if c[0].isdigit()]
    assert cases[-1] == "32, 1, 0, 4, -1, 0", cases          # generic last
    assert "32, 1, 0, 4, 2, 2" in cases[:2]
    return True



def stamp_checks():
    """DIAG 64: every stamp write is guarded by STAMP, lands in slots 0..11 of the CTA's 16 (slot 5 on one of two
    exclusive paths), and the CTA index covers the grid exactly once (16 words each, the host checks part's size)."""
    k = CUDA[CUDA.index("__global__ void __launch_bounds__(128 + 128 * MATH, CPS) rs_kernel"):]
    k = k[:k.index("\n}\n") + 3]
    writes = re.findall(r"stamps\[([^\]]+)\] = ", k)
    slots = []
    for w in writes:
        slots += [int(v) for v in re.findall(r"\d+", w.split("?")[-1])]   # `tid == 128 ? 4 : 11` -> 4, 11
    assert sorted(slots) == [0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], slots   # 3: pre_corr or pre_ring
    lines = k.splitlines()
    for i, line in enumerate(lines):
        if "stamps[" in line and "=" in line.split("stamps[")[1]:
            if "STAMP" in line:
                continue
            j = i - 1      # a bare `stamps[k] = ...` must sit in a block opened by `if (STAMP ...) {`
            while lines[j].strip().startswith("stamps["):
                j -= 1
            assert lines[j].strip().startswith("if (STAMP") and lines[j].rstrip().endswith("{"), line
    for gx, gy, gz in ((64, 1, 2), (18, 2, 7), (32, 1, 3)):
        ids = sorted(((z * gy + y) * gx + xx) for z in range(gz) for y in range(gy) for xx in range(gx))
        assert ids == list(range(gx * gy * gz))
    return True


stamp_checks()


def warm_checks():
    """DIAG 256 dry pass (real = false) must not wait, arrive, store or do math: in the math body, every barrier sits
    under `if (real)` (or a block opened by it, or `else if (real)`), DSMEM stores are the predicated form taking
    `real`, the finalize's global store is guarded by `real`, the main loop by `real`; the body is called once with
    true, or in the opaque two-pass loop (one copy)."""
    k = CUDA[CUDA.index("__global__ void __launch_bounds__(128 + 128 * MATH, CPS) rs_kernel"):]
    k = k[:k.index("\n}\n") + 3]
    body = k[k.index("    auto body = [&](const bool real) {"):k.index("    if constexpr (WARM) {")]
    assert "st_cluster_f32(" not in body and body.count("st_cluster_f32_if(") == 2
    assert body.count(", real);") == 2
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\b(cl_wait|cl_arrive|cluster_sync|bar_sync_na|named_sync)\(", line):
            ok = "if (real)" in line
            ind = len(line) - len(line.lstrip())
            j = i - 1
            while not ok and j >= 0:          # walk out to enclosing blocks
                lj = lines[j]
                ij = len(lj) - len(lj.lstrip())
                if lj.strip() and ij < ind and lj.rstrip().endswith("{"):
                    if "if (real" in lj or "else if (real)" in lj:
                        ok = True
                    ind = ij
                j -= 1
            assert ok, ("barrier reachable in the dry pass", line)
    assert "            if (live[u] && real)\n" in body and "    if (real && p < nk) {\n" in body
    assert k.count("body(") == 2 + 0 and "      body(true);\n" in k and "        body(pass == 1);\n" in k
    return True


warm_checks()
print("ok 5d: DIAG 256 dry pass: every barrier in the math body sits under real, DSMEM stores predicated by real, no "
      "finalize store or math without real; one body copy (called once, or twice in the opaque loop)")
print("ok 5c: DIAG 64 stamps: guarded writes to slots 0..11 of 16 per CTA, one slot per CTA id, grid covered once")
print("ok 5b: no-timeout barriers match: producer 1 arrive on bar 2, math 1 sync on bar 2; red 0: 2 cluster_sync per "
      "role; red 2: one relaxed arrive at the start, then wait / arrive / wait per role, pushes between; bar 1 "
      "math-only, one __syncthreads before the role split")

M32 = 0xFFFFFFFF


def rotl(v, n):
    return ((v << n) | (v >> (32 - n))) & M32


def byte_perm(a, b, sel):
    """__byte_perm(a, b, s): bytes of {b, a} (a = bytes 0-3, b = 4-7) picked by the 4 selector nibbles."""
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(a)
    for i in range(4):
        sidx = (sel >> (4 * i)) & 0x7
        out |= pool[sidx] << (8 * i)
    return out


def f12m_decode(q, l0, l1):
    r4 = rotl(q, 4)
    ha = (q & 0x87878787) | 0x38383838
    hb = (r4 & 0x87878787) | 0x38383838
    return (byte_perm(l0, ha, 0x5140), byte_perm(l0, ha, 0x7362), byte_perm(l1, hb, 0x5140), byte_perm(l1, hb, 0x7362))


def bf16_of(bits16):
    return (bits16.to(torch.int32) << 16).view(torch.float32)


def decode_all(wcm, n, k):
    """Chunk-major planes [K/64][N/16][3][32][4] -> fp32 [N, K] of decoded A values, read at the kernel's offsets
    (stage = (kc, rb): byte (kc * N/16 + 8 rb) * 1536; row tile 4c + w; slot s at +512 s; lane * 16)."""
    words = wcm.to(torch.int64) & M32                     # [KC, RT, 3, 32, 4]
    kc_n, rt_n = words.shape[0], words.shape[1]
    q = words[:, :, 0]                                      # [KC, RT, 32, 4]: Q word of step j
    l01, l23 = words[:, :, 1], words[:, :, 2]
    out = torch.zeros(n, k)
    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    for j in range(4):
        l = l01 if j < 2 else l23
        regs = f12m_decode(q[..., j], l[..., 2 * (j & 1)], l[..., 2 * (j & 1) + 1])
        for r, reg in enumerate(regs):
            for half in range(2):
                val = bf16_of((reg >> (16 * half)) & 0xFFFF)  # [KC, RT, 32]
                row = 16 * torch.arange(rt_n).view(1, -1, 1) + g.view(1, 1, -1) + 8 * (r & 1)
                col = (64 * torch.arange(kc_n).view(-1, 1, 1) + 16 * j + 2 * t.view(1, 1, -1) + 8 * (r >> 1) + half)
                row, col = torch.broadcast_tensors(row, col)
                out[row.reshape(-1), col.reshape(-1)] = val.reshape(-1)
    return out


def read_x_tile(x, tt, bm, kc):
    """Producer: unit u of token t -> byte t*128 + ((u ^ (t & 7)) << 4); wgmma B read through the SW128 descriptor:
    element (n, kk) of k16 step ks at ks*32 + (n//8)*1024 + (n%8)*128 + (kk//8)*16 + (kk%8)*2, XOR bits [4:6] with [7:9]."""
    m, k = x.shape
    buf = torch.full((bm * 128 // 2,), float("nan"))
    for t in range(bm):
        tok = tt * bm + t
        if tok >= m:  # TMA's out-of-bounds rows read as zeros
            off = t * 128
            buf[off // 2: off // 2 + 64] = 0.0
            continue
        for u in range(8):
            off = t * 128 + ((u ^ (t & 7)) << 4)
            buf[off // 2: off // 2 + 8] = x[tok, kc * 64 + u * 8: kc * 64 + u * 8 + 8]
    nn = torch.arange(bm).view(-1, 1, 1)
    ks = torch.arange(4).view(1, -1, 1)
    kk = torch.arange(16).view(1, 1, -1)
    addr = ks * 32 + (nn // 8) * 1024 + (nn % 8) * 128 + (kk // 8) * 16 + (kk % 8) * 2
    phys = addr ^ (((addr >> 7) & 7) << 4)
    return buf[(phys // 2).reshape(-1)].view(bm, 64)


def make_w(n, k, gen):
    w = (torch.randn(n, k, generator=gen) * 0.02)
    # plant out-of-window weights (escapes) on top of the Gaussian's own: 2 per row in 1/8 of the rows (F12 allows 7)
    rows = torch.randperm(n, generator=gen)[: n // 8]
    for r in rows.tolist():
        cols = torch.randint(0, k, (2,), generator=gen)
        w[r, cols] = torch.tensor([8.0, -40.0])
    return w.to(torch.bfloat16)


def emulate_layer(name, n, k, m, bm, splits, gen, ring=False):
    w = make_w(n, k, gen)
    enc = f12["f12m_encode"](w, natural=True)
    rec = f12["f12v_records"](enc, n)
    assert rec is not None
    p = f12["as_int32"](enc["p"]).contiguous()
    wcm = dg["dgrs_pack"](p)
    base = (rec[:, 0] & 0xFF).to(torch.int64)
    # 1. decode == W * 2^(112 - base), escapes 2^-15
    dec = decode_all(wcm, n, k)
    scale = torch.pow(2.0, (112 - base).double()).view(-1, 1)
    want = (w.double() * scale)
    escm = torch.zeros(n, k, dtype=torch.bool)
    cnt_r = (rec[:, 0] >> 8) & 0xFF
    for r in range(n):
        for e in range(int(cnt_r[r])):
            escm[r, int(rec[r, 1 + e] & 0xFFFF)] = True
    want = torch.where(escm, torch.full_like(want, 2.0 ** -15), want)
    assert torch.equal(dec.double(), want), f"{name}: decoded A values differ from W * 2^(112 - base)"
    # 2. x tiles
    x = (torch.randn(m, k, generator=gen)).to(torch.bfloat16).float()
    tiles = -(-m // bm)
    for tt in range(tiles):
        for kc in (0, k // 64 - 1):
            got = read_x_tile(x, tt, bm, kc)
            v = min(bm, m - tt * bm)
            assert torch.equal(got[:v], x[tt * bm: tt * bm + v, kc * 64: kc * 64 + 64]), (name, tt, kc)
    # 3. split-K GEMM + last-arriver epilogue
    kc_n = k // 64
    per = -(-kc_n // splits)
    assert (splits - 1) * per < kc_n, (name, splits)
    mp = tiles * bm
    part = torch.zeros(splits, mp, n)
    xp = torch.zeros(mp, k)
    xp[:m] = x
    for sl in range(splits):
        kc0 = sl * per
        nk = min(per, kc_n - kc0)
        ks = slice(64 * kc0, 64 * (kc0 + nk))
        part[sl] = xp[:, ks] @ dec[:, ks].T       # [mp, n] in fp32 (MMA order differs: tolerance below)
    if ring:   # DIAG 512: each slice folds its own K range's corrections into its partials (pair 0), then sum * scale
        rec64r = rec.to(torch.int64)
        tcache = {}
        for sl in range(splits):
            kc0 = sl * per
            nk = min(per, kc_n - kc0)
            for tt in range(tiles):
                xt = xp[tt * bm:(tt + 1) * bm]
                for rb in range(n // 128):
                    corr, scale, _ = pre_ring_port(rec64r[rb * 128:(rb + 1) * 128], xt, kc0, nk, cache=tcache, ckey=tt)
                    inv = torch.tensor([1.0 / v for v in scale], dtype=torch.float64)
                    blk = part[sl, tt * bm:(tt + 1) * bm, rb * 128:(rb + 1) * 128]
                    part[sl, tt * bm:(tt + 1) * bm, rb * 128:(rb + 1) * 128] = (corr[:, :128].double() * inv
                                                                                  + blk.double()).float()
        tot = part.sum(0)
        sc_all = torch.pow(2.0, (base - 112).float())
        out = (tot.double() * sc_all.double()).float()
        out = out[:m].to(torch.bfloat16).float()
        ref = x @ w.float().T
        err = float((out - ref).norm() / ref.norm())
        row = float(((out - ref).norm(dim=0) / ref.norm(dim=0).clamp(min=1e-30)).max())
        assert err < 1e-2 and row < 5e-2, (name, "ring", err, row)
        return err, row, int(escm.sum())
    tot = part.sum(0)                              # slices in order s = 0..S-1 (the finalize loop)
    out = torch.zeros(mp, n)
    rec64 = rec.to(torch.int64)
    rs = -(-128 // splits)
    for rb in range(n // 128):
        for tt in range(tiles):
            xt = xp[tt * bm:(tt + 1) * bm]           # rows >= m are zeros (pre_corr's `live`)
            for sl in range(splits):
                rlo = sl * rs
                rn = max(0, min(128, rlo + rs) - rlo)
                if rn == 0:
                    continue
                rows = slice(rb * 128 + rlo, rb * 128 + rlo + rn)
                corr, scale, _ = pre_port(rec64[rows], xt, rn, rs + 1)
                sc = torch.tensor(scale[:rn], dtype=torch.float64)
                out[tt * bm:(tt + 1) * bm, rows] = (tot[tt * bm:(tt + 1) * bm, rows].double() * sc
                                                     + corr[:, :rn].double()).float()   # fmaf(sum, scale, corr)
    tot = out
    out = tot[:m].to(torch.bfloat16).float()
    ref = x @ w.float().T
    err = float((out - ref).norm() / ref.norm())
    row = float(((out - ref).norm(dim=0) / ref.norm(dim=0).clamp(min=1e-30)).max())
    assert err < 1e-2 and row < 5e-2, (name, err, row)
    return err, row, int(escm.sum())


def counter_protocol():
    rng = random.Random(3)
    for rbs, tiles, splits in ((18, 1, 7), (64, 2, 2), (32, 1, 4), (8, 2, 1)):
        cnt = [0] * (rbs * tiles)
        for call in range(3):
            done = {}
            ctas = [(rb, tt, sl) for rb in range(rbs) for tt in range(tiles) for sl in range(splits)]
            rng.shuffle(ctas)
            finished = set()
            for rb, tt, sl in ctas:
                finished.add((rb, tt, sl))
                i = tt * rbs + rb
                old = cnt[i]
                cnt[i] += 1
                if old == splits - 1:
                    cnt[i] = 0
                    assert all((rb, tt, s2) in finished for s2 in range(splits)), "finalize before all slices"
                    assert (rb, tt) not in done, "finalized twice"
                    done[(rb, tt)] = True
            assert len(done) == rbs * tiles and not any(cnt), (len(done), call)
    return True


def finalize_coverage():
    """v5 finalize (math threads only): rank sl of S takes rows [sl * RS, min(128, (sl + 1) * RS)), RS = ceil(128 / S);
    math thread mt < TL * rn -> (row ri = mt % rn, token lane tl = mt // rn), TL = TM // rn (TM = 128 * MATH), tokens
    t = tl + TL (FU g + u) < BM, FU = 16 // MATH: every (row, token) exactly once over all ranks and threads."""
    for math, bm in ((2, 32), (2, 64), (2, 128), (4, 32), (4, 64)):
        tm, fu = 128 * math, 16 // math
        for splits in range(1, 9):
            rs = -(-128 // splits)
            seen = {}
            for sl in range(splits):
                rlo = sl * rs
                rhi = min(128, rlo + rs)
                rn = max(0, rhi - rlo)
                if rn == 0:
                    continue
                tl_n = tm // rn
                for mt in range(tl_n * rn):
                    ri, tl = mt % rn, mt // rn
                    for t0 in range(tl, bm, tl_n * fu):
                        for u in range(fu):
                            tt = t0 + tl_n * u
                            if tt < bm:
                                seen[(rlo + ri, tt)] = seen.get((rlo + ri, tt), 0) + 1
            assert len(seen) == 128 * bm and set(seen.values()) == {1}, (math, bm, splits, len(seen))


# ---- v5 shared-memory layout (port of dgrs_layout / dgrs_layout_ok)
SMEM_DYN1, SMEM_DYN2, SMAX, PRE_WARPS, ESC_CAP, W_CHUNK, GATHER = 226 * 1024, 112 * 1024, 12, 3, 304, 12288, 512


def layout(bm, cps, math, splits, red, smax):
    nr = 1 if (red == 1 and splits > 1) else splits
    push = red == 2 and splits > 1
    rs = (128 + nr - 1) // nr
    cs = rs + 1
    rsp = rs + ((4 - rs) & 7)
    stage = bm * 128 + W_CHUNK
    fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8 + PRE_WARPS * 2 * GATHER * 4
    corr = (bm * cs * 4 + 15) // 16 * 16
    recv = splits * (math // 2) * bm * rsp * 4 if push else 0
    avail = (SMEM_DYN1 if cps == 1 else SMEM_DYN2) - 1024 - fixed - corr - recv
    S = min(avail // stage, smax, SMAX)
    if math == 4:
        S &= ~1
    bars = S * stage
    sc = bars + 16 * SMAX
    esc = sc + 128 * 4
    stg = esc + PRE_WARPS * ESC_CAP * 8
    co = stg + PRE_WARPS * 2 * GATHER * 4
    L = dict(S=S, rs=rs, cs=cs, rsp=rsp, push=push, bars=bars, scale=sc, esc=esc, stage=stg, corr=co, recv=co + corr,
             end=co + corr + recv)
    L["ok"] = S >= 4 and (push or (math // 2) * bm * 128 * 4 <= bars)
    return L


def layout_checks():
    """Every accepted launch: regions in order, aligned, inside the dynamic allocation minus the 1 KB alignment pad;
    the epilogue tiles inside the ring; MATH 4 ring even. Every CPS 1 config the lab uses (BM 32/64 MATH 2/4, BM 128
    MATH 2; S 1..8; red 0/1; smax 8) is accepted. Hardware: 226 KB + s_last <= 227 KB opt-in; 2 x (112 KB + 1 KB) <=
    228 KB per SM."""
    assert SMEM_DYN1 + 16 <= 227 * 1024 and 2 * (SMEM_DYN2 + 16 + 1024) <= 228 * 1024
    stages = {}
    for bm, cps, math in ((32, 1, 2), (64, 1, 2), (128, 1, 2), (32, 1, 4), (64, 1, 4), (32, 2, 2), (64, 2, 2)):
        for splits in range(1, 9):
            for red in (0, 1, 2):
                for smax in (4, 6, 8, 12):
                    L = layout(bm, cps, math, splits, red, smax)
                    if cps == 1 and smax >= 8 and (red < 2 or bm <= 64):
                        assert L["ok"], (bm, cps, math, splits, red, smax, L)
                    if cps == 1 and smax == 8 and red == 2 and bm == 128:
                        stages[(bm, math, splits, red)] = L["S"] if L["ok"] else 0
                    if not L["ok"]:
                        continue
                    assert L["S"] <= smax and (math == 2 or L["S"] % 2 == 0)
                    assert L["bars"] % 1024 == 0 and L["scale"] % 16 == 0 and L["esc"] % 8 == 0 and L["corr"] % 16 == 0
                    assert L["stage"] % 16 == 0 and L["recv"] % 16 == 0 and L["rsp"] % 8 == 4 and L["rsp"] >= L["rs"]
                    assert L["bars"] + 16 * SMAX <= L["scale"] < L["esc"] < L["stage"] < L["corr"] < L["recv"] <= L["end"]
                    assert L["push"] == (red == 2 and splits > 1)
                    if L["push"]:
                        assert L["recv"] + splits * (math // 2) * bm * L["rsp"] * 4 == L["end"]
                    assert L["end"] <= (SMEM_DYN1 if cps == 1 else SMEM_DYN2) - 1024, L
                    assert L["corr"] + 4 * bm * L["cs"] <= L["end"]
                    if cps == 1 and smax == 8:
                        stages[(bm, math, splits, red)] = L["S"]
    return stages


# ---- v5 pre_corr port (producer warps 1-3): the kernel's row split, warp scan, list order and per-escape fma chain
def fma32(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def pre_port(recr, xt, rn, cs, u_n=None):
    """recr: int64 [rn][8] records of the finalize rows; xt: fp32 [BM][K] x tile (rows >= M are zeros, as `live`).
    Returns corr fp32 [BM][cs] (NaN where never written), scale list, per-warp list lengths."""
    bm = xt.shape[0]
    corr = torch.full((bm, cs), float("nan"))
    scale = [None] * 128
    rw = (rn + PRE_WARPS - 1) // PRE_WARPS
    lens = []
    for pw in range(PRE_WARPS):
        wlo = pw * rw
        whi = min(wlo + rw, rn)
        lst = [None] * ESC_CAP
        if whi > wlo:
            corr[:, wlo:whi] = 0.0
        ne = 0
        for r0 in range(wlo, whi, 32):
            cnts, q = [], []
            for lane in range(32):
                ri = r0 + lane
                v = ri < whi
                q0 = [int(recr[ri, j]) & M32 for j in range(4)] if v else [0] * 4
                base = q0[0] & 0xFF
                cnt = min((q0[0] >> 8) & 0xFF, 7)
                q1 = [int(recr[ri, 4 + j]) & M32 for j in range(4)] if cnt > 3 else [0] * 4
                if v:
                    scale[ri] = 2.0 ** (base - 112)
                cnts.append(cnt)
                q.append((ri, base, q0, q1))
            inc = list(cnts)
            o = 1
            while o < 32:
                prev = list(inc)
                for lane in range(32):
                    if lane >= o:
                        inc[lane] = prev[lane] + prev[lane - o]
                o <<= 1
            for lane in range(32):
                ri, base, q0, q1 = q[lane]
                off = ne + inc[lane] - cnts[lane]
                tag = ri | (base << 8)
                for e in range(7):
                    if e < cnts[lane]:
                        assert off + e < ESC_CAP, "escape list overflow"
                        lst[off + e] = (q0[e + 1] if e < 3 else q1[e - 3], tag)
            ne += inc[31]
        lens.append(ne)
        g_n = GATHER // bm
        xb = (xt.contiguous().view(torch.int32).to(torch.int64) >> 16) & 0xFFFF      # bf16 bits [BM][K]
        stg = torch.zeros(2, g_n, bm, dtype=torch.int64)
        for b in range(0, ne, g_n):            # lane = token: all BM / 32 rounds at once (disjoint tokens)
            buf = (b // g_n) & 1
            for e in range(b, min(b + g_n, ne)):   # cp.async: the 4-byte word at element k & ~1 (zeros: dead tokens)
                k2 = lst[e][0] & 0xFFFE
                assert ((buf * g_n + e - b) * bm + bm - 1) < 2 * GATHER
                stg[buf, e - b] = xb[:, k2] | (xb[:, k2 + 1] << 16)
            for e in range(b, min(b + g_n, ne)):
                if True:
                    dx, dy = lst[e]
                    ri, base = dy & 0xFF, dy >> 8
                    wv = stg[buf, e - b]
                    xv = bf16_of(((wv >> 16) if (dx & 1) else wv) & 0xFFFF)
                    wdec = torch.tensor(2.0 ** (base - 127))
                    true = bf16_of(torch.tensor([(dx >> 16) & 0xFFFF]))[0]
                    c = corr[:, ri]
                    c = fma32(-xv, wdec, c)
                    c = fma32(xv, true, c)
                    corr[:, ri] = c
    return corr, scale, lens


def pre_reference(recr, xt, rn):
    """Direct per-row chain (escapes in record order) -- what each (token, row) entry must equal exactly."""
    ref = torch.zeros(xt.shape[0], rn)
    for ri in range(rn):
        w0 = int(recr[ri, 0]) & M32
        base, cnt = w0 & 0xFF, min((w0 >> 8) & 0xFF, 7)
        c = torch.zeros(xt.shape[0])
        for e in range(cnt):
            d = int(recr[ri, 1 + e]) & M32
            xv = xt[:, d & 0xFFFF]
            c = fma32(-xv, torch.tensor(2.0 ** (base - 127)), c)
            c = fma32(xv, bf16_of(torch.tensor([(d >> 16) & 0xFFFF]))[0], c)
        ref[:, ri] = c
    return ref


def push_checks():
    """red 2: thread (pair p, half c, warp w, lane g q) of slice sl stores acc[4j + h] (row r0 = 64c + 16w + g, token
    8j + 2q + h) and acc[4j + 2 + h] (row r0 + 8) at word (slot BM + t) RSP + (row - rank RS) of rank row / RS,
    slot = sl NP + p. Checks: every (slice, pair, row, token) lands once, inside the receive area; rank r's finalize
    (rows [r RS, ...), slots 0 .. splits NP - 1) reads exactly them; each warp store hits 32 distinct banks among the
    lanes going to the same rank."""
    for math, bm in ((2, 32), (2, 64), (4, 32), (4, 64)):
        np_ = math // 2
        for splits in range(2, 9):
            L = layout(bm, 1, math, splits, 2, 8)
            rs, rsp = L["rs"], L["rsp"]
            size = splits * np_ * bm * rsp
            seen = {}
            for sl in range(splits):
                for p in range(np_):
                    for c in range(2):
                        for w in range(4):
                            for j in range(bm // 8):
                                for h in range(2):
                                    for half in range(2):
                                        banks = {}
                                        for lane in range(32):
                                            g, q = lane >> 2, lane & 3
                                            r = 64 * c + 16 * w + g + 8 * half
                                            t = 8 * j + 2 * q + h
                                            rank = r // rs
                                            addr = ((sl * np_ + p) * bm + t) * rsp + (r - rank * rs)
                                            assert 0 <= addr < size and r - rank * rs < rs
                                            key = (rank, addr)
                                            assert key not in seen, ("double push", key)
                                            seen[key] = (sl, p, r, t)
                                            banks.setdefault(rank, set()).add(addr % 32)
                                        for rk, bs in banks.items():
                                            n_l = sum(1 for lane in range(32) if (64 * c + 16 * w + (lane >> 2) + 8 * half) // rs == rk)
                                            assert len(bs) == n_l, ("bank conflict", math, bm, splits, rk)
            assert len(seen) == splits * np_ * 128 * bm
            for rank in range(splits):
                rlo = rank * rs
                rn = max(0, min(128, rlo + rs) - rlo)
                for ri in range(rn):
                    for t in range(bm):
                        got = sorted(seen[(rank, (s2 * bm + t) * rsp + ri)] for s2 in range(splits * np_))
                        want = sorted((sl, p, rlo + ri, t) for sl in range(splits) for p in range(np_))
                        assert got == want, (rank, ri, t)
    return True


def swizzled_tile_u16(xt, kc):
    """The TMA destination bytes of x tile kc as uint16 [BM * 64]: token t's unit u at t*128 + ((u ^ (t & 7)) << 4)."""
    bm = xt.shape[0]
    xb = ((xt.contiguous().view(torch.int32).to(torch.int64) >> 16) & 0xFFFF)
    buf = torch.zeros(bm * 64, dtype=torch.int64)
    for t in range(bm):
        for u in range(8):
            off = (t * 128 + ((u ^ (t & 7)) << 4)) // 2
            buf[off:off + 8] = xb[t, kc * 64 + u * 8: kc * 64 + u * 8 + 8]
    return buf


def match_any(vals):
    return [sum(1 << j for j in range(32) if vals[j] == vals[i]) for i in range(32)]


def popc(v):
    return bin(v & 0xFFFFFFFF).count("1")


def ffs(v):
    return (v & -v).bit_length()


def pre_ring_port(recr, xt, kc0, nk, S=None, cache=None, ckey=None):
    """recr int64 [128][8] (the row block's records), xt fp32 [BM][K] (rows >= M zeros). Returns corr [BM][129],
    scale[128], and per warp (list, bucketed list, ends): the kernel's list build, match_any bucketing, per-lane
    scan, scatter, and per-chunk reads from the swizzled tile bytes."""
    bm = xt.shape[0]
    cs = 129
    corr = torch.full((bm, cs), float("nan"))
    scale = [None] * 128
    rwr = (128 + PRE_WARPS - 1) // PRE_WARPS
    out_l = []
    tiles = {}
    for pw in range(PRE_WARPS):
        wlo, whi = pw * rwr, min(128, pw * rwr + rwr)
        corr[:, wlo:whi] = 0.0
        ends = [0] * nk
        lst = []
        ne = 0
        for r0 in range(wlo, whi, 32):
            ms, rows = [], []
            for lane in range(32):
                ri = r0 + lane
                v = ri < whi
                q0 = [int(recr[ri, j]) & M32 for j in range(4)] if v else [0] * 4
                base, cnt = q0[0] & 0xFF, min((q0[0] >> 8) & 0xFF, 7)
                q1 = [int(recr[ri, 4 + j]) & M32 for j in range(4)] if cnt > 3 else [0] * 4
                if v:
                    scale[ri] = 2.0 ** (base - 112)
                words = [(q0[e + 1] if e < 3 else q1[e - 3]) for e in range(7)]
                sel = [wd for e, wd in enumerate(words) if e < cnt and 0 <= ((wd & 0xFFFF) >> 6) - kc0 < nk]
                ms.append(len(sel))
                rows.append((ri | (base << 8), sel))
            for lane in range(32):
                tag, sel = rows[lane]
                lst += [(wd, tag) for wd in sel]
            ne += sum(ms)
        assert len(lst) == ne <= ESC_CAP
        chs = [((d[0] & 0xFFFF) >> 6) - kc0 for d in lst]
        for b in range(0, ne, 32):                       # counts
            vals = [chs[b + l] if b + l < ne else -1 for l in range(32)]
            grp = match_any(vals)
            for l in range(32):
                if vals[l] >= 0 and l == ffs(grp[l]) - 1:
                    ends[vals[l]] += popc(grp[l])
        assert nk <= 256
        tot = [sum(ends[c] for c in range(8 * l, min(nk, 8 * l + 8))) for l in range(32)]   # per-lane scan
        run_l = [sum(tot[:l]) for l in range(32)]
        for l in range(32):
            run = run_l[l]
            for u in range(8):
                c = 8 * l + u
                if c < nk:
                    v_ = ends[c]
                    ends[c] = run
                    run += v_
        buck = [None] * ne
        for b in range(0, ne, 32):                       # scatter
            vals = [chs[b + l] if b + l < ne else -1 for l in range(32)]
            grp = match_any(vals)
            pos = [ends[vals[l]] + popc(grp[l] & ((1 << l) - 1)) if vals[l] >= 0 else 0 for l in range(32)]
            for l in range(32):
                if vals[l] >= 0:
                    buck[pos[l]] = lst[b + l]
            for l in range(32):
                if vals[l] >= 0 and l == ffs(grp[l]) - 1:
                    ends[vals[l]] += popc(grp[l])
        assert buck == sorted(lst, key=lambda d: ((d[0] & 0xFFFF) >> 6)), "bucketing must be a stable sort by chunk"
        for i in range(nk):                              # per chunk
            tk = (ckey, kc0 + i)
            store = tiles if cache is None else cache
            if tk not in store:
                store[tk] = swizzled_tile_u16(xt, kc0 + i)
            buf = store[tk]
            lo, hi = (ends[i - 1] if i > 0 else 0), ends[i]
            for e in range(lo, hi):
                dx, dy = buck[e]
                kk = (dx & 0xFFFF) - (kc0 + i) * 64
                assert 0 <= kk < 64
                t = torch.arange(bm)
                addr = t * 128 + ((((kk >> 3) ^ (t & 7)) & 7) << 4) + (kk & 7) * 2
                xv = bf16_of(buf[addr // 2])
                ri, base = dy & 0xFF, dy >> 8
                c = corr[:, ri]
                c = fma32(-xv, torch.tensor(2.0 ** (base - 127)), c)
                c = fma32(xv, bf16_of(torch.tensor([(dx >> 16) & 0xFFFF]))[0], c)
                corr[:, ri] = c
        out_l.append((lst, buck, ends))
    return corr, scale, out_l


def pre_ring_reference(recr, xt, kc0, nk):
    """Direct chain per row: in-range escapes ordered by (chunk, record order); x read straight from x."""
    bm = xt.shape[0]
    ref = torch.zeros(bm, 128)
    for ri in range(128):
        w0 = int(recr[ri, 0]) & M32
        base, cnt = w0 & 0xFF, min((w0 >> 8) & 0xFF, 7)
        ds = [int(recr[ri, 1 + e]) & M32 for e in range(cnt)]
        ds = [d for d in ds if 0 <= ((d & 0xFFFF) >> 6) - kc0 < nk]
        ds.sort(key=lambda d: (d & 0xFFFF) >> 6)
        c = torch.zeros(bm)
        for d in ds:
            xv = xt[:, d & 0xFFFF]
            c = fma32(-xv, torch.tensor(2.0 ** (base - 127)), c)
            c = fma32(xv, bf16_of(torch.tensor([(d >> 16) & 0xFFFF]))[0], c)
        ref[:, ri] = c
    return ref


def ring_checks():
    """pre_ring port vs the direct chain, bit for bit: BM 32/64/128, slices of 1..256 chunks at offsets, random and
    all-7 rows (lists up to 301), escapes in and out of the slice; the bucketing equals a stable sort by chunk."""
    rng = random.Random(9)
    gen = torch.Generator().manual_seed(9)
    n = 0
    for bm, k, kc0, nk, mode in ((32, 2048, 0, 32, "random"), (32, 2048, 13, 19, "full"), (32, 2048, 0, 32, "full"), (64, 1024, 4, 7, "random"),
                                 (128, 1024, 0, 16, "random"), (32, 16384, 0, 256, "random"), (32, 4096, 31, 33, "full")):
        recr = torch.zeros(128, 8, dtype=torch.int64)
        for ri in range(128):
            cnt = 7 if mode == "full" else rng.choice([0, 0, 1, 1, 2, 3, 7])
            base = rng.randrange(96, 120)
            recr[ri, 0] = base | (cnt << 8)
            ks = rng.sample(range(k), 7)
            for e in range(7):
                tb = rng.randrange(0x3C00, 0x4400) | (rng.randrange(2) << 15)
                recr[ri, 1 + e] = ks[e] | (tb << 16)
        xt = torch.randn(bm, k, generator=gen).to(torch.bfloat16).float()
        xt[bm - 3:] = 0.0
        corr, scale, _ = pre_ring_port(recr, xt, kc0, nk)
        ref = pre_ring_reference(recr, xt, kc0, nk)
        assert not torch.isnan(corr[:, :128]).any()
        assert torch.equal(corr[:, :128], ref), (bm, kc0, nk, float((corr[:, :128] - ref).abs().max()))
        assert all(sc is not None for sc in scale)
        n += 1
    return n


def pre_checks():
    """Random records (counts 0..7, all-7 rows to fill the per-warp list to 301), rn = 1..128 incl. every rank share
    of S = 1..8, BM 32/64/128, U 4/8: each (token, row) correction equals the direct chain bit for bit, every entry
    of the rank's rows is written, every row gets its scale, no list overflow."""
    rng = random.Random(5)
    gen = torch.Generator().manual_seed(5)
    k = 512
    rns = sorted({max(0, min(128, sl * -(-128 // S) + -(-128 // S)) - sl * -(-128 // S)) for S in range(1, 9)
                  for sl in range(S)} | {1, 2, 3, 4, 31, 33, 97, 128})
    n = 0
    for rn in rns:
        if rn == 0:
            continue
        for bm, u_n in ((32, 4), (64, 4), (128, 8)):
            for mode in ("random", "full"):
                recr = torch.zeros(rn, 8, dtype=torch.int64)
                for ri in range(rn):
                    cnt = 7 if mode == "full" else rng.choice([0, 0, 0, 1, 1, 2, 3, 4, 7])
                    base = rng.randrange(96, 120)
                    recr[ri, 0] = base | (cnt << 8)
                    ks = rng.sample(range(k), 7)
                    for e in range(7):
                        tb = rng.randrange(0x3C00, 0x4400) | (rng.randrange(2) << 15)
                        recr[ri, 1 + e] = ks[e] | (tb << 16)
                xt = torch.randn(bm, k, generator=gen).to(torch.bfloat16).float()
                xt[bm - 5:] = 0.0      # tokens >= M
                corr, scale, lens = pre_port(recr, xt, rn, rn + 1)
                ref = pre_reference(recr, xt, rn)
                assert not torch.isnan(corr[:, :rn]).any(), (rn, bm, "unwritten correction")
                assert torch.equal(corr[:, :rn], ref), (rn, bm, mode, float((corr[:, :rn] - ref).abs().max()))
                assert all(scale[ri] is not None for ri in range(rn))
                n += 1
    return n


def math4_protocol(S, nk, seed, np_=2, rel_off=None, pw_n=0, pw_nowait=False):
    """Producer (thread 0: chunk i -> slot i % S after the slot's previous phase was released; full barrier count 1)
    and NP pairs (pair p: chunks p, p + NP, ...; full wait (i / S) & 1; after its step for chunk i, release chunk
    i - NP; both warpgroups of a pair arrive: empty count 2) in random interleavings. Checks: each chunk consumed once,
    by pair i % NP, only after it landed and before its slot is refilled; no deadlock. pw_n > 0 (DIAG 512): pw_n
    producer warps also consume every chunk in order (wait full, read, then arrive: empty count 2 + pw_n)."""
    rng = random.Random(seed)
    full_ph = [0] * S      # completed phases
    empty_ph = [0] * S
    empty_cnt = [0] * S
    slot_chunk = [None] * S
    reading = [set() for _ in range(S)]
    consumed = {}
    prod_i = 0
    pairs = [[p, 0] for p in range(np_)]   # next chunk, step state
    pws = [[0, False] for _ in range(pw_n)]  # next chunk, mid-read
    target = 2 + pw_n
    steps = 0
    while True:
        steps += 1
        if steps > 100000:
            raise AssertionError("livelock")
        opts = []
        if prod_i < nk:
            s, rnd = prod_i % S, prod_i // S
            # try_wait.parity((rnd - 1) & 1) passes once the barrier's completed-phase count has the other parity
            if rnd == 0 or (empty_ph[s] & 1) != ((rnd - 1) & 1):
                opts.append(("prod",))
        for p in range(np_):
            i = pairs[p][0]
            if i < nk and (full_ph[i % S] & 1) != ((i // S) & 1):   # parity wait, as the hardware does it
                opts.append(("cons", p))
        for wq in range(pw_n):
            i, mid = pws[wq]
            if mid:
                opts.append(("pw_end", wq))
            elif i < nk and (pw_nowait or (full_ph[i % S] & 1) != ((i // S) & 1)):
                opts.append(("pw_start", wq))
        if not opts:
            assert prod_i >= nk and all(pr[0] >= nk for pr in pairs) and all(w_[0] >= nk for w_ in pws), (S, nk, "deadlock")
            break
        o = rng.choice(opts)
        if o[0] == "pw_start":
            wq = o[1]
            i = pws[wq][0]
            assert slot_chunk[i % S] == i and full_ph[i % S] == i // S + 1, f"warp {wq} read chunk {i} from a stale phase"
            reading[i % S].add(("w", wq, i))
            pws[wq][1] = True
            continue
        if o[0] == "pw_end":
            wq = o[1]
            i = pws[wq][0]
            reading[i % S].discard(("w", wq, i))
            empty_cnt[i % S] += 1
            if empty_cnt[i % S] == target:
                empty_cnt[i % S] = 0
                empty_ph[i % S] += 1
            pws[wq] = [i + 1, False]
            continue
        if o[0] == "prod":
            s = prod_i % S
            assert not reading[s], f"slot {s} refilled while chunk {slot_chunk[s]} is read"
            assert prod_i < S or empty_ph[s] == prod_i // S, f"slot {s}: producer passed a stale empty phase"
            slot_chunk[s] = prod_i
            full_ph[s] += 1
            prod_i += 1
        else:
            p = o[1]
            i = pairs[p][0]
            assert i % np_ == p and i not in consumed
            assert slot_chunk[i % S] == i and full_ph[i % S] == i // S + 1, f"pair {p} read chunk {i} from a stale phase"
            consumed[i] = p
            reading[i % S].add(i)
            prev = i - (np_ if rel_off is None else rel_off)
            if prev >= 0:   # after wait_group<1>: the pair's previous chunk is done; both warpgroups arrive
                reading[prev % S].discard(prev)
                for _wg in range(2):
                    empty_cnt[prev % S] += 1
                    if empty_cnt[prev % S] == target:
                        empty_cnt[prev % S] = 0
                        empty_ph[prev % S] += 1
            pairs[p][0] = i + np_
    assert sorted(consumed) == list(range(nk))


for S_ in (4, 6, 8):
    for nk_ in range(1, 23):
        for sd in range(20):
            math4_protocol(S_, nk_, sd)
            math4_protocol(S_, nk_, sd, np_=1)
            math4_protocol(S_, nk_, sd, pw_n=3)          # DIAG 512: producer warps 1-3 as ring consumers
            math4_protocol(S_, nk_, sd, np_=1, pw_n=3)
bad_pw = False
for sd in range(300):
    try:
        math4_protocol(4, 17, sd, pw_n=3, pw_nowait=True)   # planted: producer warps read without the full wait
    except AssertionError:
        bad_pw = True
        break
assert bad_pw, "producer warps reading without the full wait must be caught"
for S_ in (5, 7):  # an odd ring is safe too: each pair and the producer go in order, so a slot's completed-phase
    for sd in range(20):  # count is k or k + 1 when a pair waits for phase k (no parity aliasing); kernel keeps S even
        math4_protocol(S_, 17, sd)
bad_caught = False
for sd in range(200):
    try:
        math4_protocol(4, 17, sd, rel_off=1)  # planted: release the chunk just before (the other pair's), not i - NP
    except AssertionError:
        bad_caught = True
        break
assert bad_caught, "releasing the other pair's chunk must be caught"
print("ok 4b: MATH 4 / MATH 2 chunk schedule with hardware parity waits: every chunk consumed once by its pair after "
      "it landed, no refill while read, no stale phase, no deadlock (S 4-8, 1-22 chunks, random orders), also with the "
      "3 producer warps as ring consumers (empty count 5, DIAG 512); planted wrong release / missing full wait caught")
stages = layout_checks()
spec_checks()
print("ok 5e: v9 specializations: folded row shares == runtime ones (S 1..8, red 0/1/2, every rank); dispatch lists "
      "specialized instances first and matches red / splits")
print("ok 4c: v6 shared-memory layout: regions ordered, aligned, inside 226 KB (112 KB at CPS 2) less the alignment "
      "pad, epilogue tiles inside the ring, MATH 4 ring even; every lab config accepted; ring stages (smax 8, red 0) "
      "red 0 BM 128 S1..8: " + " ".join(str(stages[(128, 2, sp, 0)]) for sp in range(1, 9)) + "; red 2 S2..8 BM 32 MATH 4: "
      + " ".join(str(stages[(32, 4, sp, 2)]) for sp in range(2, 9)) + ", BM 64 MATH 4: "
      + " ".join(str(stages[(64, 4, sp, 2)]) for sp in range(2, 9)) + ", BM 128 (0 = rejected): "
      + " ".join(str(stages[(128, 2, sp, 2)]) for sp in range(2, 9)))
push_checks()
print("ok 4e: red 2 push map: every (slice, pair, row, token) lands once in its owner rank's receive area, each rank's "
      "finalize reads exactly its rows' S x NP partials, conflict-free warp stores (RSP = 4 mod 8), S 2..8, MATH 2/4")
nring = ring_checks()
print(f"ok 4f: pre_ring port ({nring} cases: BM 32/64/128, slices of 7..256 chunks, random and all-7 rows): bucketing "
      "== stable sort by chunk, corrections from the swizzled tile bytes == the direct chain bit for bit")
npre = pre_checks()
print(f"ok 4d: pre_corr port ({npre} cases: every rank share of S 1..8, BM 32/64/128, random and all-7 escape rows): "
      "each correction == the direct per-row chain bit for bit, all entries and scales written, list <= ESC_CAP")
finalize_coverage()
print("ok 4: v5 finalize map (rank row ranges x thread (row, token lane)) covers every (row, token) once for "
      "S = 1..8, BM 32/64/128 (256 math threads, FU 8) and BM 32/64 (512, FU 4, MATH 4); slices are summed in rank order (deterministic)")

gen = torch.Generator().manual_seed(0)
LAYERS = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pooler", 4096, 8192), ("unpooler", 8192, 4096)]
CASES = [(32, 32, {"qkv": 7, "o": 2, "pooler": 4, "unpooler": 2}),
         (128, 128, {"qkv": 7, "o": 2, "pooler": 4, "unpooler": 2}),
         (256, 128, {"qkv": 4, "o": 1, "pooler": 2, "unpooler": 1})]
if QUICK:
    LAYERS = [("o", 1024, 1024), ("qkv-small", 256, 2048)]
    CASES = [(32, 32, {"o": 2, "qkv-small": 7}), (200, 128, {"o": 1, "qkv-small": 3})]
lines = []
for m, bm, splits in CASES:
    for name, n, k in LAYERS:
        e, r, ne = emulate_layer(name, n, k, m, bm, splits[name], gen)
        if m <= 32 or QUICK:
            e2, r2, _ = emulate_layer(name, n, k, m, bm, splits[name], torch.Generator().manual_seed(len(lines)), ring=True)
            lines.append(f"{name} m{m} bm{bm} S{splits[name]} ring (DIAG 512): err {e2:.1e} row {r2:.1e}")
        lines.append(f"{name} m{m} bm{bm} S{splits[name]}: err {e:.1e} row {r:.1e} ({ne} escapes)")
print("ok 1-3: decode == W * 2^(112 - base) exactly (escapes 2^-15), x tiles read back exactly, split-K GEMM + "
      "F12M epilogue vs FP32 x @ W^T:\n  " + "\n  ".join(lines))

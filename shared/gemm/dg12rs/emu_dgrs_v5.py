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
    "        for (int s2 = 0; s2 < splits; ++s2) {  // rank order: deterministic",
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
    '        const float scale = sScale[ri];',
    '                  __float2bfloat16_rn(fmaf(sum[u], scale, sCorr[(t0 + TL * u) * L.cs + ri]));',
    '  const int nr = glob ? 1 : splits;',
    '  L.rs = (128 + nr - 1) / nr;',
    '  L.cs = L.rs + 1;',
    '  const int stage = bm * 128 + W_CHUNK;',
    '  const int fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8;',
    '  const int corr = (bm * L.cs * 4 + 15) / 16 * 16;',
    '  const int avail = (cps == 1 ? SMEM_DYN1 : SMEM_DYN2) - 1024 - fixed - corr;',
    '  if (math == 4) S &= ~1;  // even: slot s only ever holds chunks of parity s % 2, i.e. of pair s % 2',
    '  L.bars = S * stage;',
    '  L.scale = L.bars + 16 * SMAX;',
    '  L.esc = L.scale + 128 * 4;',
    '  L.corr = L.esc + PRE_WARPS * ESC_CAP * 8;',
    '  L.end = L.corr + corr;',
    '  return L.S >= 4 && (math / 2) * bm * 128 * 4 <= L.bars;',
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
    '    const uint16_t* xr = xh + static_cast<size_t>(live ? tok0 + t : 0) * K;',
    '    float* cr = sCorr + t * cs;',
    '        xv[u] = (live && b + u < ne) ? __uint_as_float(static_cast<uint32_t>(xr[list[b + u].x & 0xFFFFu]) << 16) : 0.f;',
    '          float* cp = cr + (d.y & 0xFFu);',
    '          c = fmaf(-xv[u], f12m_wdec(d.y >> 8), c);',
    '          c = fmaf(xv[u], __uint_as_float(d.x & 0xFFFF0000u), c);',
    '      pre_corr<BM, (MATH == 4 ? 4 : 8)>(rec, reinterpret_cast<const uint16_t*>(x), sScale, sEsc, sCorr, rb * 128 + rlo,',
    '                                        rn, L.cs, tt * BM, M, K, (tid >> 5) - 1, tid & 31);',
    '    } else if (tid >= 32 && (DIAG & 49) == 0) {',
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
    assert me.count("bar_sync_na(2, THREADS);") == 1 and math.count("bar_sync_na(") == 1, "math: one sync on barrier 2"
    assert math.count("bar_arrive_na(") == 0
    assert pe.count("cluster_sync();") == 2 and prod.count("cluster_sync();") == 2
    assert "      if (splits > 1 && !glob) {  // the math warpgroups' two cluster barriers\n" in pe
    assert me.count("cluster_sync();") == 2 and math.count("cluster_sync();") == 2
    assert "      } else if (splits > 1) {\n        cluster_sync();" in me      # glob false, splits > 1
    assert "      if (splits > 1 && !glob) cluster_sync();  // peers may still read this CTA's tile\n" in me
    assert "  constexpr int THREADS = 128 + 128 * MATH;\n" in k and "  constexpr int TM = 128 * MATH;" in k
    assert k.count("named_sync(1, TM);") == 5 and "named_sync(1, 128" not in k   # barrier 1: math threads only
    assert k.count("__syncthreads();") == 1   # the barrier-init sync, before the role split
    return True


barrier_match()
print("ok 5b: no-timeout barriers match: producer 1 arrive on bar 2 + 2 cluster barriers, math 1 sync on bar 2 + "
      "2 cluster barriers (splits > 1, red 0), bar 1 math-only, one __syncthreads before the role split")

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


def emulate_layer(name, n, k, m, bm, splits, gen):
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
                corr, scale, _ = pre_port(rec64[rows], xt, rn, rs + 1, 4)
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
SMEM_DYN1, SMEM_DYN2, SMAX, PRE_WARPS, ESC_CAP, W_CHUNK = 226 * 1024, 112 * 1024, 12, 3, 304, 12288


def layout(bm, cps, math, splits, glob, smax):
    nr = 1 if glob else splits
    rs = (128 + nr - 1) // nr
    cs = rs + 1
    stage = bm * 128 + W_CHUNK
    fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8
    corr = (bm * cs * 4 + 15) // 16 * 16
    avail = (SMEM_DYN1 if cps == 1 else SMEM_DYN2) - 1024 - fixed - corr
    S = min(avail // stage, smax, SMAX)
    if math == 4:
        S &= ~1
    bars = S * stage
    sc = bars + 16 * SMAX
    esc = sc + 128 * 4
    co = esc + PRE_WARPS * ESC_CAP * 8
    L = dict(S=S, rs=rs, cs=cs, bars=bars, scale=sc, esc=esc, corr=co, end=co + corr)
    L["ok"] = S >= 4 and (math // 2) * bm * 128 * 4 <= bars
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
            for red in (0, 1):
                for smax in (4, 6, 8, 12):
                    L = layout(bm, cps, math, splits, red and splits > 1, smax)
                    if cps == 1 and smax >= 8:
                        assert L["ok"], (bm, cps, math, splits, red, smax, L)
                    if not L["ok"]:
                        continue
                    assert L["S"] <= smax and (math == 2 or L["S"] % 2 == 0)
                    assert L["bars"] % 1024 == 0 and L["scale"] % 16 == 0 and L["esc"] % 8 == 0 and L["corr"] % 16 == 0
                    assert L["bars"] + 16 * SMAX <= L["scale"] < L["esc"] < L["corr"] < L["end"]
                    assert L["end"] <= (SMEM_DYN1 if cps == 1 else SMEM_DYN2) - 1024, L
                    assert L["corr"] + 4 * bm * L["cs"] <= L["end"]
                    if cps == 1 and smax == 8 and red == 0:
                        stages[(bm, math, splits)] = L["S"]
    return stages


# ---- v5 pre_corr port (producer warps 1-3): the kernel's row split, warp scan, list order and per-escape fma chain
def fma32(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def pre_port(recr, xt, rn, cs, u_n):
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
        for b in range(0, ne, u_n):            # lane = token: all BM / 32 rounds at once (disjoint tokens)
            for u in range(u_n):
                if b + u < ne:
                    dx, dy = lst[b + u]
                    ri, base = dy & 0xFF, dy >> 8
                    xv = xt[:, dx & 0xFFFF]
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
                corr, scale, lens = pre_port(recr, xt, rn, rn + 1, u_n)
                ref = pre_reference(recr, xt, rn)
                assert not torch.isnan(corr[:, :rn]).any(), (rn, bm, "unwritten correction")
                assert torch.equal(corr[:, :rn], ref), (rn, bm, mode, float((corr[:, :rn] - ref).abs().max()))
                assert all(scale[ri] is not None for ri in range(rn))
                n += 1
    return n


def math4_protocol(S, nk, seed, np_=2, rel_off=None):
    """Producer (thread 0: chunk i -> slot i % S after the slot's previous phase was released; full barrier count 1)
    and NP pairs (pair p: chunks p, p + NP, ...; full wait (i / S) & 1; after its step for chunk i, release chunk
    i - NP; both warpgroups of a pair arrive: empty count 2) in random interleavings. Checks: each chunk consumed once,
    by pair i % NP, only after it landed and before its slot is refilled; no deadlock."""
    rng = random.Random(seed)
    full_ph = [0] * S      # completed phases
    empty_ph = [0] * S
    empty_cnt = [0] * S
    slot_chunk = [None] * S
    reading = [set() for _ in range(S)]
    consumed = {}
    prod_i = 0
    pairs = [[p, 0] for p in range(np_)]   # next chunk, step state
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
        if not opts:
            assert prod_i >= nk and all(pr[0] >= nk for pr in pairs), (S, nk, "deadlock")
            break
        o = rng.choice(opts)
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
                    if empty_cnt[prev % S] == 2:
                        empty_cnt[prev % S] = 0
                        empty_ph[prev % S] += 1
            pairs[p][0] = i + np_
    assert sorted(consumed) == list(range(nk))


for S_ in (4, 6, 8):
    for nk_ in range(1, 23):
        for sd in range(20):
            math4_protocol(S_, nk_, sd)
            math4_protocol(S_, nk_, sd, np_=1)
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
      "it landed, no refill while read, no stale phase, no deadlock (S 4-8, 1-22 chunks, random orders); a planted "
      "wrong release is caught")
stages = layout_checks()
print("ok 4c: v5 shared-memory layout: regions ordered, aligned, inside 226 KB (112 KB at CPS 2) less the alignment "
      "pad, epilogue tiles inside the ring, MATH 4 ring even; every lab config accepted; ring stages (smax 8, red 0) "
      "BM 128 S1..8: " + " ".join(str(stages[(128, 2, sp)]) for sp in range(1, 9)) + ", BM 64 MATH 4: "
      + " ".join(str(stages[(64, 4, sp)]) for sp in range(1, 9)))
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
        lines.append(f"{name} m{m} bm{bm} S{splits[name]}: err {e:.1e} row {r:.1e} ({ne} escapes)")
print("ok 1-3: decode == W * 2^(112 - base) exactly (escapes 2^-15), x tiles read back exactly, split-K GEMM + "
      "F12M epilogue vs FP32 x @ W^T:\n  " + "\n  ".join(lines))

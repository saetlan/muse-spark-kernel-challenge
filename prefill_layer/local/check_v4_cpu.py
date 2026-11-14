"""CPU checks of v4's R14 path (Hopper W4A16 CUDA kernels), at the real sizes (K = I = H = 4096), with
the kernels' own index formulas re-implemented in PyTorch (CUDA can't run here):

1. r14_pack + the consumer's fragment loads + dec<R> + mul.rn.bf16x2 + the wgmma A-fragment layout
   (row = c*64 + w*16 + g + 8*((i>>1)&1), k = ks*16 + 2q + (i&1) + 8*(i>>2)): every weight element of
   every (block, chunk) comes back EXACTLY as code * scale * 2^-62, for gate/up-interleaved GEMM1 and
   plain GEMM2 rows, with all 127 E4M3 scales present.
2. route_kernel (ported): segments 16-aligned, tiles 128/64/32/16 inside their segment, every local
   route at a unique row of its expert's segment.
3. The gather's 128B swizzle + the producer's bulk-copy offsets + the wgmma descriptor's address math
   (start + ks*32, SBO 1024, XOR of address bits [4:6] with [7:9]): every B element read back equals the
   token's activation.
4. The whole path (route, gather, GEMM1 with SwiGLU epilogue and swizzled staging, GEMM2 epilogue,
   v2's combine) vs a dense FP32 reference from nvfp4.dequantize, at K = I = H = 4096, with experts
   that need 128/64/32/16-row tiles and a token without local routes.
5. The CUDA/Triton source lines these ports mirror are pinned (an edit fails until the port follows).

Run: .venv/bin/python prefill_layer/local/check_v4_cpu.py
"""

import sys
import types
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR))
import nvfp4  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_ports import KC, H, I, bf16_from_bits, dec, hmul2_lane, decode_blob, route, gather, read_b, gemm  # noqa: E402

EMU = {}


def _jit(fn=None, **kw):
    def wrap(f):
        class K:
            def __getitem__(self, grid):
                return lambda *a, **k: EMU[f.__name__](grid, *a, **k)
        return K()
    return wrap(fn) if fn is not None else wrap


class _Any(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})


for mod in ("sgl_kernel", "sgl_kernel.flash_attn", "sgl_kernel.scalar_type", "sglang", "sglang.kernels",
            "sglang.kernels.ops", "sglang.kernels.ops.activation", "sglang.kernels.ops.activation.activation",
            "sglang.kernels.ops.kvcache", "sglang.kernels.ops.kvcache.kvcache", "sglang.kernels.ops.moe",
            "sglang.kernels.ops.moe.moe_wna16_marlin", "sglang.kernels.ops.muse_spark_v12",
            "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing",
            "sglang.kernels.ops.muse_spark_v12.fused_norm_gate", "sglang.kernels.ops.muse_spark_v12.fused_norm_residual",
            "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
            "sglang.kernels.ops.muse_spark_v12.fused_qk_norm", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.moe", "sglang.srt.layers.moe.fused_moe_triton", "sglang.srt.layers.muse_spark_layers",
            "sglang.srt.layers.quantization", "sglang.srt.layers.quantization.marlin_utils",
            "sglang.srt.layers.quantization.marlin_utils_fp4", "sglang.srt.layers.quantization.unquant"):
    sys.modules[mod] = _Any(mod)
triton = types.ModuleType("triton")
triton.jit = _jit
tl = _Any("triton.language")
triton.language = tl
sys.modules["triton"], sys.modules["triton.language"] = triton, tl

SRC = (DIR / "submission.py").read_text()
ns = {"__name__": "v4"}
exec(compile(SRC, "submission.py", "exec"), ns)
CUDA = ns["r14_source"]()

H = I = 4096
KC = 64
gen = torch.Generator().manual_seed(0)
all_scales = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)


# ------------------------------------------------------------------------- bf16 / decode helpers








# --------------------------------------------------------------------------------- 1. pack + decode
E = 3
w13_q = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen)
w13_sf = all_scales[torch.randint(0, 127, (E, 2 * I, H // 16), generator=gen)]
w2_q = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen)
w2_sf = all_scales[torch.randint(0, 127, (E, H, I // 16), generator=gen)]
blob13 = ns["r14_pack"](w13_q, w13_sf, True)
blob2 = ns["r14_pack"](w2_q, w2_sf, False)
assert blob13.shape == (E, 64, 64, 5120) and blob2.shape == (E, 32, 64, 5120), (blob13.shape, blob2.shape)
order13 = ns["_r14_rows_gemm1"]()
assert sorted(order13.tolist()) == list(range(2 * I)), "GEMM1 row order is not a permutation"
A13, A2 = {}, {}
for e in range(E):
    A13[e] = decode_blob(blob13[e])
    want13 = (nvfp4.dequantize(w13_q[e], w13_sf[e], 1.0) * 2.0**-62)[order13]
    assert torch.equal(A13[e], want13), f"GEMM1 decode mismatch, expert {e}"
    A2[e] = decode_blob(blob2[e])
    assert torch.equal(A2[e], nvfp4.dequantize(w2_q[e], w2_sf[e], 1.0) * 2.0**-62), f"GEMM2 decode mismatch, expert {e}"
print("ok 1: r14_pack + fragment loads + dec<R> + bf16x2 scale multiply + A layout = code*scale*2^-62 exactly "
      "(GEMM1 gate/up interleaved, GEMM2 plain; K = 4096; all E4M3 scales)")


# ------------------------------------------------------------------------------- 2. route kernel


# ----------------------------------------------------------------------------- the test routing
# 160 tokens; local experts 0..2: e0 = 149 routes (tiles 128, 16, 16), e1 = 106 (64, 32, 16), e2 = 32 (32);
# token 7 has no local route; every token's 8 experts are distinct.
T = 160
ids = torch.empty((T, 8), dtype=torch.int32)
for t in range(T):
    loc = []
    if t < 150 and t != 7:
        loc.append(0)
    if t % 3 != 0 and t != 7:
        loc.append(1)
    if t % 5 == 0 and t != 7:
        loc.append(2)
    ids[t] = torch.tensor(loc + list(range(100, 100 + 8 - len(loc))), dtype=torch.int32)
local_ids = torch.where(ids < 32, ids, torch.full_like(ids, -1))
slots, max_pieces, rows = ns["_r14_sizes"](T)
pos, pieces, start, cnt = route(local_ids.view(-1), max_pieces)
for (e, r, v, n) in pieces:
    assert r % 16 == 0 and start[e] <= r and r + n <= start[e + 1] and 0 < v <= n and n in (16, 32, 64, 128)
for e in range(32):
    rows_e = sorted(int(p) for p, i in zip(pos.tolist(), local_ids.view(-1).tolist()) if i == e)
    assert rows_e == list(range(start[e], start[e] + cnt[e])), e
    covered = sum(v for (ee, r, v, n) in pieces if ee == e)
    assert covered == cnt[e]
assert start[32] + ns["R14_BT"] <= rows
# Worst case of the tile count: per expert floor(cnt / 128) + up to 4 remainder tiles (64, 32, 16, 16).
for tt in (1, 40, 300, 753, 3012, 16384):
    sl, mp, rw = ns["_r14_sizes"](tt)
    for trial in range(200):
        gg = torch.Generator().manual_seed(trial)
        cuts = torch.sort(torch.randint(0, sl + 1, (31,), generator=gg)).values.tolist()
        counts = [b - a for a, b in zip([0] + cuts, cuts + [sl])]
        if trial % 3 == 0:  # many 127-route experts: 4 remainder tiles each
            counts = [min(127, sl // 32)] * 32
        npieces, padded = 0, 0
        for cnt_e in counts:
            rem = cnt_e
            while rem > 0:
                n = 128 if rem >= 128 else 64 if rem >= 64 else 32 if rem >= 32 else 16
                rem -= min(rem, n)
                npieces += 1
            padded += -(-cnt_e // 16) * 16
        assert npieces <= mp and padded + ns["R14_BT"] <= rw, (tt, trial, npieces, mp, padded, rw)
print(f"ok 2: route port: counts {cnt[:3]}, pieces {[(p[1], p[2], p[3]) for p in pieces]}, all rows inside their segments")

# ----------------------------------------------------------------- 3. gather swizzle + descriptor reads
x = (torch.randn(T, H, generator=gen) * 0.5).to(torch.bfloat16)






a_sw = gather(x, pos, rows)
flat_ids = local_ids.view(-1)
for (e, r, v, n) in pieces:
    for kc in (0, 17, 63):
        b = read_b(a_sw, r, n, kc, rows)
        for t in range(v):
            slot = int((pos == r + t).nonzero()[0])
            assert torch.equal(b[t], x[slot // 8, kc * 64:(kc + 1) * 64]), (e, r, t, kc)
assert sorted(set(p[3] for p in pieces)) == [16, 32, 64, 128], pieces
print("ok 3: gather swizzle + bulk-copy offsets + descriptor address math (SW128, SBO 1024, +32 B per k16) "
      "read back every token's activations")


# ------------------------------------------------------------------------------ 4. the whole path


gs13 = torch.rand(E, generator=gen) * 1e-3 + 1e-4
gs2 = torch.rand(E, generator=gen) * 1e-3 + 1e-4
c1, c2 = ns["r14_epilogue_scales"]({"w13_gs": gs13, "w2_gs": gs2})
act_sw = torch.zeros(KC, rows, 64, dtype=torch.bfloat16)
for e in range(E):
    gemm(A13[e], a_sw, e, 64, True, float(c1[e]), act_sw, rows, pieces)
y2 = torch.zeros(rows, H, dtype=torch.bfloat16)
for e in range(E):
    gemm(A2[e], act_sw, e, 32, False, float(c2[e]), y2, rows, pieces)
post_w = (1 + 0.1 * torch.randn(I, generator=gen)).to(torch.bfloat16)
w = torch.softmax(torch.randn(T, 8, generator=gen), -1)
out = torch.zeros(T, I)
for t in range(T):
    for k in range(8):
        e = int(local_ids[t, k])
        if e >= 0:
            yy = y2[int(pos[t * 8 + k])].float() * gs2[e] * post_w.float()
            out[t] += yy * torch.rsqrt((yy * yy).sum() / I + 1e-8) * w[t, k]
ref = torch.zeros(T, I)
for e in range(E):
    W13 = nvfp4.dequantize(w13_q[e], w13_sf[e], 1.0) * gs13[e]
    W2 = nvfp4.dequantize(w2_q[e], w2_sf[e], 1.0) * gs2[e]
    tok, slot = (local_ids == e).nonzero(as_tuple=True)
    h = x[tok].float() @ W13.T
    yv = ((torch.nn.functional.silu(h[:, :I]) * h[:, I:]) @ W2.T) * post_w.float()
    yv = yv * torch.rsqrt(yv.pow(2).mean(-1, keepdim=True) + 1e-8)
    ref.index_add_(0, tok, yv * w[tok, slot].unsqueeze(-1))
live = ref.norm(dim=-1) > 0
assert int((~live).sum()) >= 1 and float(out[~live].abs().max()) == 0.0
err = float((out - ref).norm() / ref.norm())
row = float(((out - ref).norm(dim=-1)[live] / ref.norm(dim=-1)[live]).max())
assert torch.isfinite(out).all() and err < 1e-2 and row < 5e-2, (err, row)
print(f"ok 4: whole R14 path (tiles {sorted(set(p[3] for p in pieces))}) vs dense FP32 at K=I=H=4096: "
      f"rel err {err:.2e}, worst row {row:.2e}")

# ------------------------------------------------------------------------------- 5. pinned lines
PINNED = [
    "  const uint32_t x = w << (4 * R);", "  return (x & 0x80008000u) | ((x >> 6) & 0x01C001C0u);",
    "  const uint32_t sg = __byte_perm(sw, 0u, 0x1010u);", "  const uint32_t sg8 = __byte_perm(sw, 0u, 0x3232u);",
    "  r0 = hmul2(dec<0>(wd), sg);", "  r1 = hmul2(dec<1>(wd), sg8);", "  r2 = hmul2(dec<2>(wd), sg);",
    "  r3 = hmul2(dec<3>(wd), sg8);",
    "  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |",
    "         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);",
    "    wgmma##NN(acc, a[0], a[1], a[2], a[3], desc, first ? 0 : 1);                                    \\",
    "    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \\",
    "    wgmma##NN(acc, a[8], a[9], a[10], a[11], desc + 4, 1);                                          \\",
    "    wgmma##NN(acc, a[12], a[13], a[14], a[15], desc + 6, 1);                                        \\",
    "  dec_k16(wv.x, sv.x, a[0], a[1], a[2], a[3]);", "  dec_k16(wv.y, sv.y, a[4], a[5], a[6], a[7]);",
    "  dec_k16(wv.z, sv.z, a[8], a[9], a[10], a[11]);", "  dec_k16(wv.w, sv.w, a[12], a[13], a[14], a[15]);",
    "  const int woff = ((c * 4 + w) * 32 + lane) * 16;", "  const int soff = W_BYTES + ((c * 4 + w) * 8 + g) * 16;",
    "  const int c = (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;",
    "    const int col = c * 32 + w * 8 + g;", "        const int t = 8 * j + 2 * q + h;",
    "        const float gt = acc[4 * j + h] * cs;", "        const float up = acc[4 * j + 2 + h] * cs;",
    "        const float v = gt / (1.f + expf(-gt)) * up;",
    "        *reinterpret_cast<__nv_bfloat16*>(stage + t * 128 + ((((col >> 3) ^ (t & 7))) << 4) + (col & 7) * 2) =",
    "    const int rows = (valid + 7) & ~7;",
    "    uint4* dst = reinterpret_cast<uint4*>(out + (static_cast<size_t>(nb) * P + row0) * 64);",
    "    const int r = c * 64 + w * 16 + g;",
    "        *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + r * 2) = __float2bfloat16_rn(acc[4 * j + h] * cs);",
    "        *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + (r + 8) * 2) = __float2bfloat16_rn(acc[4 * j + 2 + h] * cs);",
    "      reinterpret_cast<uint4*>(out + static_cast<size_t>(row0 + t) * K + nb * 128)[u] = src[i];",
    "      const uint8_t* wsrc = wblob + static_cast<size_t>(e * NB + nb) * KC * WS_BYTES;",
    "        bulk_g2s(smem_u32(sW + s * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);",
    "        bulk_g2s(smem_u32(sB + s * B_BYTES), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * s);",
    "      start[e + 1] = start[e] + ((cnt[e] + SEG - 1) / SEG) * SEG;",
    "        const int n = rem >= 128 ? 128 : rem >= 64 ? 64 : rem >= 32 ? 32 : 16;",
    "        const int v = rem < n ? rem : n;", "        if (np < max_pieces) pieces[np] = make_int4(e, r, v, n);",
    "    pos[i] = e >= 0 ? atomicAdd(&cur[e], 1) : -1;",
]
missing = [ln for ln in PINNED if (CUDA.count(ln + "\n") < 1)]
GATHER_LINES = ["    dst = kc.to(tl.int64) * P * 64 + p64 * 64 + ((u ^ (p & 7)) * 8 + offs % 8)",
                "    kc = offs // 64", "    u = (offs // 8) % 8"]
missing += [ln for ln in GATHER_LINES if SRC.count(ln + "\n") < 1]
assert not missing, f"kernel source changed; update the ports: {missing[:3]}"
print("ok 5: CUDA/Triton lines mirrored by the ports are unchanged")

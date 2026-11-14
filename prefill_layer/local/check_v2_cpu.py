"""CPU check of v2's grouped expert path (_experts_gmm): orchestration (sort, counts, offsets, inverse
permutation, gather, skipped experts) plus a PyTorch port of each Triton kernel's math, line for line,
against a dense FP32 reference built with nvfp4.dequantize. Also checks the E2M1 formula for all 16
codes and the nibble interleave over the full row, and every E4M3 scale incl. subnormals.

Run: .venv/bin/python prefill_layer/local/check_v2_cpu.py
"""

import sys
import types
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR))
import nvfp4  # noqa: E402

EMU = {}


def _jit(fn=None, **kw):
    def wrap(f):
        class K:
            __name__ = f.__name__

            def __getitem__(self, grid):
                return lambda *a, **k: EMU[f.__name__](grid, *a, **k)

            def __call__(self, *a, **k):
                return EMU[f.__name__](None, *a, **k)
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

ns = {"__name__": "v2"}
exec(compile((DIR / "submission.py").read_text(), "submission.py", "exec"), ns)


# --- PyTorch ports of the Triton kernels (same formulas) ---
def e2m1_value(code):
    sign = (code >> 3) & 1
    exp = (code >> 1) & 3
    man = (code & 1).float()
    normal = (1.0 + 0.5 * man) * torch.where(exp == 3, 4.0, torch.where(exp == 2, 2.0, 1.0))
    mag = torch.where(exp == 0, 0.5 * man, normal)
    return torch.where(sign == 1, -mag, mag)


def emu_dequant(grid, q, sf, counts, out, N, KB, BN, BKB, num_warps):
    assert grid == (q.shape[0], N // BN, KB // BKB) and N % BN == 0 and KB % BKB == 0
    for e in range(q.shape[0]):
        if int(counts[e]) == 0:
            continue
        qi = q[e].to(torch.int32)
        s = sf[e].float().repeat_interleave(8, dim=-1)  # one scale per packed byte (cols // 8)
        lo, hi = e2m1_value(qi & 15) * s, e2m1_value(qi >> 4) * s
        out[e] = torch.stack((lo, hi), dim=-1).reshape(N, 2 * KB).to(torch.bfloat16)  # tl.join + reshape


SEEN = {}


def emu_gather(grid, x, perm, offs, out, H, TOPK, NE, num_warps):
    n = int(offs[NE - 1])
    SEEN["n_local"], SEEN["offs"] = n, offs.clone()
    assert grid == (out.shape[0],)
    out[:n] = x[perm[:n] // TOPK]


def emu_act(grid, h, keys, gs, offs, out, I, NE, num_warps):
    n = int(offs[NE - 1])
    g = gs[keys[:n].long()].unsqueeze(-1)
    gate, up = h[:n, :I].float() * g, h[:n, I:2 * I].float() * g
    out[:n] = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def emu_combine(grid, y, ids, inv, gs, s, w, out, eps, H, TOPK, num_warps):
    tokens = grid[0]
    wf = w.float()
    for t in range(tokens):
        acc = torch.zeros(H)
        for k in range(TOPK):
            e = int(ids.view(-1)[t * TOPK + k])
            if e >= 0:
                row = int(inv[t * TOPK + k])
                x = y[row].float() * gs[e]
                yy = x * wf
                yy = yy * torch.rsqrt((yy * yy).sum() / H + eps)
                acc += yy * s.view(-1)[t * TOPK + k]
        out[t] = acc.to(torch.bfloat16)


EMU.update(_nvfp4_dequant_kernel=emu_dequant, _gather_rows_kernel=emu_gather, _gmm_act_kernel=emu_act,
           _gmm_combine_kernel=emu_combine)


def grouped_mm(a, b, offs, out_dtype):
    out = torch.full((a.shape[0], b.shape[-1]), float("nan"), dtype=out_dtype)  # rows past offs stay garbage
    start = 0
    for g, end in enumerate(offs.tolist()):
        out[start:end] = (a[start:end].float() @ b[g].float()).to(out_dtype)
        start = end
    return out


torch._grouped_mm = grouped_mm
torch.cuda.synchronize = lambda *a, **k: None

# 0. The emulation above mirrors these exact kernel lines; if the kernel source changes, this fails
#    and the emulation must be updated with it (the Triton code itself only runs on the GPU).
SRC = (DIR / "submission.py").read_text()
KERNEL_LINES = [
    "    sign = (code >> 3) & 1", "    exp = (code >> 1) & 3", "    man = (code & 1).to(tl.float32)",
    "    normal = (1.0 + 0.5 * man) * tl.where(exp == 3, 4.0, tl.where(exp == 2, 2.0, 1.0))",
    "    mag = tl.where(exp == 0, 0.5 * man, normal)", "    return tl.where(sign == 1, -mag, mag)",
    "    if tl.load(counts_ptr + e) == 0:",
    "    q = tl.load(q_ptr + e64 * (N * KB) + rows[:, None] * KB + cols[None, :]).to(tl.int32)",
    "    sf = tl.load(sf_ptr + e64 * (N * (KB // 8)) + rows[:, None] * (KB // 8) + (cols // 8)[None, :]).to(tl.float32)",
    "    lo = _e2m1_value(q & 15) * sf", "    hi = _e2m1_value(q >> 4) * sf",
    "    vals = tl.reshape(tl.join(lo, hi), (BN, 2 * BKB))",
    "    out_cols = tl.program_id(2) * (2 * BKB) + tl.arange(0, 2 * BKB)",
    "    tl.store(out_ptr + e64 * (N * 2 * KB) + rows[:, None] * (2 * KB) + out_cols[None, :], vals.to(tl.bfloat16))",
    "    if r >= tl.load(offs_ptr + NE - 1):",
    "    tok = (tl.load(perm_ptr + r) // TOPK).to(tl.int64)",
    "    tl.store(out_ptr + r.to(tl.int64) * H + cols, tl.load(x_ptr + tok * H + cols))",
    "    gs = tl.load(gs_ptr + tl.load(keys_ptr + r))",
    "    gate = tl.load(h_ptr + base + cols).to(tl.float32) * gs",
    "    up = tl.load(h_ptr + base + I + cols).to(tl.float32) * gs",
    "    tl.store(out_ptr + r.to(tl.int64) * I + cols, (gate * tl.sigmoid(gate) * up).to(tl.bfloat16))",
    "        if expert >= 0:", "            row = tl.load(inv_ptr + tok * TOPK + k)",
    "            x = tl.load(y_ptr + row * H + offs).to(tl.float32) * tl.load(gs_ptr + expert)",
    "            y = x * w", "            var = tl.sum(y * y, axis=0) / H", "            y = y * tl.rsqrt(var + eps)",
    "            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)",
]
missing = [line for line in KERNEL_LINES if SRC.count(line + "\n") < 1]
assert not missing, f"kernel source changed; update the emulation: {missing[:3]}"

# 1. E2M1 formula for all 16 codes.
codes = torch.arange(16, dtype=torch.int32)
assert torch.equal(e2m1_value(codes), nvfp4.unpack_e2m1(torch.arange(16, dtype=torch.uint8))[0::2]), "e2m1 table"

# 2. Small layer: K = I = 64 (full rows), 32 local experts, every E4M3 scale incl. subnormals and zero.
K = I = 64
ns.update(D_EXPERT=I)
E, gen = 32, torch.Generator().manual_seed(0)
all_scales = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)  # 0x00..0x7E


def bank_weights(n, k):
    q = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=gen)
    idx = torch.randint(0, all_scales.numel(), (E, n, k // 16), generator=gen)
    sf = all_scales[idx]
    gs = torch.rand(E, generator=gen) * 1e-3 + 1e-4
    return q, sf, gs


w13_q, w13_sf, w13_gs = bank_weights(2 * I, K)
w2_q, w2_sf, w2_gs = bank_weights(K, I)
post_w = (1 + 0.1 * torch.randn(I, generator=gen)).to(torch.bfloat16)
expert_map = torch.full((256,), -1, dtype=torch.int32)
expert_map[:E] = torch.arange(E, dtype=torch.int32)
state = {"bank": {"expert_map": expert_map}, "post_expert_w": post_w}
gmm = {"w13_q": w13_q, "w13_sf": w13_sf, "w2_q": w2_q, "w2_sf": w2_sf, "w13_gs": w13_gs, "w2_gs": w2_gs,
       "w13": torch.zeros(E, 2 * I, K, dtype=torch.bfloat16), "w2": torch.zeros(E, K, I, dtype=torch.bfloat16),
       "ok": True}
state["gmm"] = gmm

ns["DEQ_BLOCK_N"], ns["DEQ_BLOCK_KB"] = 32, 8  # kernel blocks that fit these small rows
# Exactness of the decode: BF16 scratch == FP32 dequantize without the global scale, every expert.
ones = torch.ones(E + 1, dtype=torch.int32)
ns["_dequant"](w13_q, w13_sf, ones, gmm["w13"])
ref13 = nvfp4.dequantize(w13_q, w13_sf, 1.0)
assert torch.equal(gmm["w13"].float(), ref13), "dequant not exact"

T = 50
x = (torch.randn(T, K, generator=gen) * 0.5).to(torch.bfloat16)
ids, w = ns["_gmm_selftest_routing"](T, "cpu", seed=3)
ids[0] = torch.arange(200, 208, dtype=torch.int32)  # a token with no local route
ids[(ids >= 28) & (ids < 32)] += 100  # experts 28-31 get no route (ids stay distinct per token)
untouched = set(range(E)) - set(ids[ids < E].tolist())
gmm["w13"].fill_(7.0)
gmm["w2"].fill_(7.0)
out = ns["_experts_gmm"](x, w, ids, state).float()

# Experts without routes are not decoded (their scratch keeps the sentinel).
for e in untouched:
    assert bool((gmm["w13"][e] == 7.0).all()), f"expert {e} decoded without routes"

# Dense FP32 reference (SGLang semantics: gs inside the weights, SiLU-and-mul, weight-first post norm).
W13 = nvfp4.dequantize(w13_q, w13_sf, 1.0) * w13_gs.view(E, 1, 1)
W2 = nvfp4.dequantize(w2_q, w2_sf, 1.0) * w2_gs.view(E, 1, 1)
ref = torch.zeros(T, I)
for t in range(T):
    for k in range(8):
        e = int(ids[t, k])
        if e < E:
            h = x[t].float() @ W13[e].T
            a = torch.nn.functional.silu(h[:I]) * h[I:]
            y = (a @ W2[e].T) * post_w.float()
            y = y * torch.rsqrt(y.pow(2).mean() + 1e-8)
            ref[t] += y * w[t, k]
err = float((out - ref).norm() / ref.norm())
assert torch.isfinite(out).all()
assert float(out[0].abs().max()) == 0.0, "token without local routes must be zero"
assert err < 2e-2, err
assert len(untouched) >= 4
local_counts = torch.bincount(ids[ids < E].long(), minlength=E)
assert SEEN["n_local"] == int((ids < E).sum()), "grouped rows must be exactly the local routes"
assert torch.equal(SEEN["offs"].long(), local_counts.cumsum(0)), "group offsets must be the per-expert cumsum"
print(f"ok: v2 grouped path vs dense FP32 reference rel err {err:.2e} (T={T}, {len(untouched)} untouched experts); "
      f"decode exact for all E4M3 scales")

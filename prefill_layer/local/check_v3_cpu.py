"""CPU checks of v3's fused expert path (R3), at the real sizes (K = I = H = 4096, 32 local experts).

1. _r3_dequant's asm, parsed from submission.py and interpreted bit for bit (PTX interpreter copied from
   moe/local/check_m2.py), on every byte value at every pack position against every E4M3 scale; and
   the asm text equals M2's (moe/m2/m2.py), which ran correctly on the H200.
2. r3_repack + the kernel's tile addressing (both scale paths), for BK 128 and 256 over the full
   K = 4096: every weight element recovered exactly (= code * scale * 2^-62).
3. _r3_tiles (real code) on skewed counts: every local row in exactly one tile of its own expert.
4. The whole _experts_r3 (real routing, tables, gather and orchestration; GEMM kernels emulated with
   the kernel's math; v2's combine emulated as in check_v2_cpu) vs a dense FP32 reference.
5. The kernel source lines the emulation mirrors are pinned.

Run: .venv/bin/python prefill_layer/local/check_v3_cpu.py
"""

import ast
import sys
import types
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1]
ROOT = DIR.parent
sys.path.insert(0, str(DIR))
import nvfp4  # noqa: E402

EMU = {}


def _jit(fn=None, **kw):
    def wrap(f):
        class K:
            __name__ = f.__name__

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
ns = {"__name__": "v3"}
exec(compile(SRC, "submission.py", "exec"), ns)


# ------------------------------------------------------------- 1. decode asm (from check_m2.py)
def _u32(x):
    return x & 0xFFFFFFFF


def asm_text(source_text, fn_name):
    tree = ast.parse(source_text)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "inline_asm_elementwise")
    return next(k.value.value for k in call.keywords if k.arg == "asm")


def _bf16_pair(reg32):
    lo16 = (reg32 & 0xFFFF).to(torch.int32).to(torch.int16)
    hi16 = ((reg32 >> 16) & 0xFFFF).to(torch.int32).to(torch.int16)
    return torch.stack((lo16, hi16), -1).view(torch.bfloat16)


def _pair_bits(bf2):
    bits = bf2.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return bits[:, 0] | (bits[:, 1] << 16)


def run_ptx(asm, regs):
    def val(tok):
        tok = tok.strip()
        return regs[tok] if tok in regs else int(tok, 0)

    for line in asm.replace("{", ";").replace("}", ";").split(";"):
        line = line.strip()
        if not line or line.startswith(".reg"):
            continue
        op, args = line.split(None, 1)
        a = [t.strip() for t in args.split(",")]
        if op == "prmt.b32":
            src = val(a[1]) | (val(a[2]) << 32)
            ctl = val(a[3])
            out = 0
            for i in range(4):
                sel = (ctl >> (4 * i)) & 0x7
                out = out | (((src >> (8 * sel)) & 0xFF) << (8 * i))
            regs[a[0]] = out
        elif op == "and.b32":
            regs[a[0]] = val(a[1]) & val(a[2])
        elif op == "or.b32":
            regs[a[0]] = val(a[1]) | val(a[2])
        elif op == "shr.b32":
            regs[a[0]] = _u32(val(a[1])) >> val(a[2])
        elif op == "shl.b32":
            regs[a[0]] = _u32(val(a[1]) << val(a[2]))
        elif op == "mul.rn.bf16x2":
            prod = (_bf16_pair(val(a[1])).float() * _bf16_pair(val(a[2])).float()).to(torch.bfloat16)
            regs[a[0]] = _pair_bits(prod)
        else:
            raise SystemExit(f"run_ptx: unsupported instruction {op!r}")
    return regs


def emulate_dequant(asm, q4, s_lo4, s_hi4):
    b = q4.to(torch.int64)
    regs = {"$4": b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24),
            "$5": _pair_bits(s_lo4[:, 0:2]), "$6": _pair_bits(s_lo4[:, 2:4]),
            "$7": _pair_bits(s_hi4[:, 0:2]), "$8": _pair_bits(s_hi4[:, 2:4])}
    regs = run_ptx(asm, regs)
    return (torch.cat((_bf16_pair(regs["$0"]), _bf16_pair(regs["$1"])), -1),
            torch.cat((_bf16_pair(regs["$2"]), _bf16_pair(regs["$3"])), -1))


asm = asm_text(SRC, "_r3_dequant")
assert asm == asm_text((ROOT / "moe/m2/m2.py").read_text(), "_m2_dequant"), "decode asm differs from M2's"
scales = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
s_bf = (scales * 2.0**64).to(torch.bfloat16)
assert torch.equal(s_bf.float(), scales * 2.0**64)
byte = torch.arange(256, dtype=torch.int64)
q = torch.stack([(byte + 37 * i) % 256 for i in range(4)], -1).to(torch.uint8).repeat_interleave(127, 0)
si = torch.arange(127).repeat(256)
s_lo = torch.stack([s_bf[(si + 5 * i) % 127] for i in range(4)], -1)
s_hi = torch.stack([s_bf[(si + 11 * i + 3) % 127] for i in range(4)], -1)
lo, hi = emulate_dequant(asm, q, s_lo, s_hi)
codes = nvfp4.unpack_e2m1(q.reshape(-1)).view(-1, 4, 2)
assert torch.equal(lo.float(), codes[..., 0] * s_lo.float() * 2.0**-126), "decode asm (low nibbles)"
assert torch.equal(hi.float(), codes[..., 1] * s_hi.float() * 2.0**-126), "decode asm (high nibbles)"
print("ok 1: _r3_dequant asm = M2's; interpreted bit for bit = code * scale * 2^-62 for every byte, position, scale")


# ----------------------------------------------- 2. repack + the kernel's tile addressing at K = 4096
def decode_like_kernel(qrep, sf, bk, scale2d):
    """Rebuild [rows, K] from the repacked bytes exactly as _r3_tile addresses them, per BK tile."""
    rows, kh = qrep.shape
    k = 2 * kh
    hb, g = bk // 2, bk // 32
    out = torch.empty(rows, k)
    sfx = (sf.float() * 2.0**64).to(torch.bfloat16).float()
    offs_j = torch.arange(hb)
    for k0 in range(0, k, bk):
        if scale2d:
            qq = qrep[:, k0 // 2 + offs_j]
            slo = sfx[:, (k0 + offs_j) // 16]
            shi = sfx[:, (k0 + hb + offs_j) // 16]
        else:
            idx = (torch.arange(g)[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)  # offs_g*16 + offs_i
            qq = qrep[:, k0 // 2 + idx]
            slo = sfx[:, k0 // 16 + torch.arange(g)].repeat_interleave(16, -1)
            shi = sfx[:, k0 // 16 + g + torch.arange(g)].repeat_interleave(16, -1)
        c = nvfp4.unpack_e2m1(qq).view(rows, hb, 2)
        out[:, k0 + offs_j] = ((c[..., 0] * 2.0**-126).to(torch.bfloat16).float() * slo).to(torch.bfloat16).float()
        out[:, k0 + hb + offs_j] = ((c[..., 1] * 2.0**-126).to(torch.bfloat16).float() * shi).to(torch.bfloat16).float()
    return out


gen = torch.Generator().manual_seed(0)
all_scales = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)
rows_t = 64
qt = torch.randint(0, 256, (rows_t, 2048), dtype=torch.uint8, generator=gen)
sft = all_scales[torch.randint(0, 127, (rows_t, 256), generator=gen)]
want = nvfp4.dequantize(qt, sft, 1.0) * 2.0**-62
for bk in (128, 256):
    rep = ns["r3_repack"](qt, bk)
    for scale2d in (0, 1):
        assert torch.equal(decode_like_kernel(rep, sft, bk, scale2d), want), f"addressing bk={bk} scale2d={scale2d}"
print("ok 2: r3_repack + tile addressing exact over K = 4096 for BK 128/256, both scale paths, all E4M3 scales")

# ---------------------------------------------------------------------------- 3. tile tables
E = ns["LOCAL_EXPERTS"]
for seed in range(6):
    g2 = torch.Generator().manual_seed(seed)
    counts = torch.zeros(E + 1, dtype=torch.int32)
    hot = torch.randint(0, E, (6,), generator=g2)
    counts[:E] = torch.randint(0, 3, (E,), generator=g2).to(torch.int32) * torch.randint(0, 40, (E,), generator=g2).to(torch.int32)
    counts[hot] += torch.randint(50, 600, (6,), generator=g2).to(torch.int32)
    offs = counts[:E].cumsum(0, dtype=torch.int32)
    slots = int(offs[-1]) + int(torch.randint(0, 500, (1,), generator=g2))
    for bt in (16, 32, 64, 128):
        mt = ns["_r3_max_tiles"](slots, bt)
        tab = ns["_r3_tiles"](counts, offs, bt, mt)
        assert tab.shape == (3, mt) and tab.dtype == torch.int32
        seen = torch.zeros(slots, dtype=torch.int32)
        for t in range(mt):
            e, r0, r1 = (int(v) for v in tab[:, t])
            if e < 0:
                continue
            start, end = int(offs[e] - counts[e]), int(offs[e])
            assert r1 == end and start <= r0 < end, (seed, bt, t)
            rows = torch.arange(r0, min(r0 + bt, r1))
            seen[rows] += 1
        assert torch.equal(seen[: int(offs[-1])], torch.ones(int(offs[-1]), dtype=torch.int32)), (seed, bt)
        assert int(seen[int(offs[-1]):].sum()) == 0
print("ok 3: tile tables cover every local row exactly once, in its own expert's tiles (6 skewed routings x 4 BT)")


# --------------------------------------------------------------- 4. the whole fused path vs FP32
KERNEL_CALLS = []


def emu_r3_gemm(grid, a, q, sf, c, table, out, max_tiles, N, K, ROWS, GATED, SWAP, BT, BN, BK, SCALE2D,
                num_warps, num_stages):
    """The kernel's math, every valid tile at once over all N columns (grid = (N/BN) x tiles)."""
    assert grid == ((N // BN) * max_tiles,) and N % BN == 0 and K % BK == 0
    assert q.shape[1] == ROWS and q.shape[2] * 2 == K and ROWS == (2 * N if GATED else N)
    KERNEL_CALLS.append((GATED, BT, BN, BK))
    for t in range(max_tiles):
        e = int(table[0, t])
        if e < 0:
            continue
        r0, r1 = int(table[1, t]), int(table[2, t])
        rows = torch.arange(r0, r0 + BT)
        valid = rows < r1
        rows = rows[valid]
        w = decode_like_kernel(q[e], sf[e], BK, SCALE2D)  # [ROWS, K], code*scale*2^-62 in bf16
        x = a[rows].float()
        if GATED:
            g = (x @ w[:N].T) * float(c[e])
            u = (x @ w[N:].T) * float(c[e])
            res = g * torch.sigmoid(g) * u
        else:
            res = (x @ w.T) * float(c[e])
        out[rows] = res.to(torch.bfloat16)


def emu_gather(grid, x, perm, offs, out, H, TOPK, NE, num_warps):
    n = int(offs[NE - 1])
    out[:n] = x[perm[:n] // TOPK]


def emu_combine(grid, y, ids, inv, gs, s, w, out, eps, H, TOPK, num_warps):
    wf = w.float()
    for t in range(grid[0]):
        acc = torch.zeros(H)
        for k in range(TOPK):
            e = int(ids.view(-1)[t * TOPK + k])
            if e >= 0:
                xx = y[int(inv[t * TOPK + k])].float() * gs[e]
                yy = xx * wf
                yy = yy * torch.rsqrt((yy * yy).sum() / H + eps)
                acc += yy * s.view(-1)[t * TOPK + k]
        out[t] = acc.to(torch.bfloat16)


EMU.update(_r3_gemm_kernel=emu_r3_gemm, _gather_rows_kernel=emu_gather, _gmm_combine_kernel=emu_combine)
torch.cuda.synchronize = lambda *a, **k: None

H = I = 4096
w13_q = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen)
w13_sf = all_scales[torch.randint(0, 127, (E, 2 * I, H // 16), generator=gen)]
w2_q = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen)
w2_sf = all_scales[torch.randint(0, 127, (E, H, I // 16), generator=gen)]
w13_gs = torch.rand(E, generator=gen) * 1e-3 + 1e-4
w2_gs = torch.rand(E, generator=gen) * 1e-3 + 1e-4
post_w = (1 + 0.1 * torch.randn(I, generator=gen)).to(torch.bfloat16)
expert_map = torch.full((256,), -1, dtype=torch.int32)
expert_map[:E] = torch.arange(E, dtype=torch.int32)
state = {"bank": {"expert_map": expert_map}, "post_expert_w": post_w,
         "gmm": {"w13_q": w13_q, "w13_sf": w13_sf, "w2_q": w2_q, "w2_sf": w2_sf, "w13_gs": w13_gs, "w2_gs": w2_gs}}
assert ns["R3_CONFIGS"] == {} or isinstance(ns["R3_CONFIGS"], dict)
saved_cfgs, ns["R3_CONFIGS"] = ns["R3_CONFIGS"], {}
ns["_r3_prepare"](state, None)  # the real prepare code builds the epilogue scales (no configs: no GPU check)
ns["R3_CONFIGS"] = saved_cfgs

# 24 tokens; local routes: expert 3 x20 (2 tiles of 16), expert 7 x17, expert 0 x5, expert 31 x1.
T = 24
ids = torch.empty(T, 8, dtype=torch.int32)
for t in range(T):
    local = ([3] if t < 20 else []) + ([7] if t < 17 else []) + ([0] if t % 5 == 0 else []) + ([31] if t == 23 else [])
    ids[t] = torch.tensor(local + list(range(100, 100 + 8 - len(local))), dtype=torch.int32)
w = torch.softmax(torch.randn(T, 8, generator=gen), -1)
x = (torch.randn(T, H, generator=gen) * 0.5).to(torch.bfloat16)

cfg1 = (0, 16, 128, 256, 8, 3, 0)
cfg2 = (1, 16, 64, 128, 4, 3, 1)
ns["r3_add_bk"](state, 256)
ns["r3_add_bk"](state, 128)
out = ns["_experts_r3"](x, w, ids, state, cfg1, cfg2).float()
assert [c[0] for c in KERNEL_CALLS] == [True, False]

W13 = {e: nvfp4.dequantize(w13_q[e], w13_sf[e], 1.0) * w13_gs[e] for e in (0, 3, 7, 31)}
W2 = {e: nvfp4.dequantize(w2_q[e], w2_sf[e], 1.0) * w2_gs[e] for e in (0, 3, 7, 31)}
ref = torch.zeros(T, I)
for t in range(T):
    for k in range(8):
        e = int(ids[t, k])
        if e < E:
            h = x[t].float() @ W13[e].T
            act = torch.nn.functional.silu(h[:I]) * h[I:]
            y = (act @ W2[e].T) * post_w.float()
            ref[t] += y * torch.rsqrt(y.pow(2).mean() + 1e-8) * w[t, k]
err = float((out - ref).norm() / ref.norm())
live = ref.norm(dim=-1) > 0
assert int((~live).sum()) >= 2 and float(out[~live].abs().max()) == 0.0, "tokens without local routes must be 0"
row = float(((out - ref).norm(dim=-1)[live] / ref.norm(dim=-1)[live]).max())
assert torch.isfinite(out).all() and err < 1e-2 and row < 5e-2, (err, row)
print(f"ok 4: _experts_r3 (BK 256 swap0 / BK 128 swap1 scale2d) vs dense FP32 at K=I=H=4096: rel err {err:.2e}, "
      f"worst row {row:.2e}")

# ---------------------------------------------------------------------------- 5. pinned lines
KERNEL_LINES = [
    "    nb = pid // max_tiles", "    t = pid - nb * max_tiles", "    e = tl.load(table_ptr + t)",
    "    row0 = tl.load(table_ptr + max_tiles + t)", "    row_end = tl.load(table_ptr + 2 * max_tiles + t)",
    "    offs_n = nb * BN + tl.arange(0, BN)",
    "    w_rows = w_ptr + e64 * (ROWS * (K // 2)) + offs_n.to(tl.int64) * (K // 2)",
    "    s_rows = sf_ptr + e64 * (ROWS * (K // 16)) + offs_n.to(tl.int64) * (K // 16)",
    "    w_tile = w_rows[:, None, None] + (offs_g[:, None] * 16 + offs_i[None, :])[None, :, :]",
    "    s_tile = s_rows[:, None] + offs_g[None, :]", "    a_row = (row0 + tl.arange(0, BT)).to(tl.int64)",
    "    valid = a_row < row_end", "        a_tile = a_ptr + a_row[None, :] * K + offs_j[:, None]",
    "        a_tile = a_ptr + a_row[:, None] * K + offs_j[None, :]",
    "        a_lo = tl.load(a_tile + k0, mask=a_mask, other=0.0)",
    "        a_hi = tl.load(a_tile + (k0 + HB), mask=a_mask, other=0.0)",
    "            acc = tl.dot(lo, a_lo, acc)", "            acc = tl.dot(hi, a_hi, acc)",
    "            acc = tl.dot(a_lo, tl.trans(lo), acc)", "            acc = tl.dot(a_hi, tl.trans(hi), acc)",
    "            lo, hi = _r3_tile(w_rows + N * (K // 2), s_rows + N * (K // 16), w_tile + N * (K // 2),",
    "        g = acc * c_e", "        res = g * tl.sigmoid(g) * (acc_u * c_e)", "        res = acc * c_e",
    "    tl.store(out_ptr + rows * N + cols, res.to(tl.bfloat16), mask=a_mask)",
    "        q = tl.load(w_rows[:, None] + (k0 // 2 + offs_j)[None, :])",
    "        s_lo = _r3_scales(s_rows[:, None] + ((k0 + offs_j) // 16)[None, :])",
    "        s_hi = _r3_scales(s_rows[:, None] + ((k0 + HB + offs_j) // 16)[None, :])",
    "        s_lo = tl.broadcast_to(_r3_scales(s_tile + k0 // 16)[:, :, None], (BN, G, 16))",
    "        s_hi = tl.broadcast_to(_r3_scales(s_tile + (k0 // 16 + G))[:, :, None], (BN, G, 16))",
    "        lo, hi = _r3_dequant(tl.load(w_tile + k0 // 2), s_lo, s_hi)",
    "    return (tl.load(ptrs).to(tl.float32) * 18446744073709551616.0).to(tl.bfloat16)",
]
missing = [line for line in KERNEL_LINES if SRC.count(line + "\n") < 1]
assert not missing, f"kernel source changed; update the emulation: {missing[:3]}"
print("ok 5: kernel lines mirrored by the emulation are unchanged")

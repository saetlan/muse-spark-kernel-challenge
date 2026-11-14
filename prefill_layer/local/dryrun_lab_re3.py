"""CPU dry run of lab_re3.py's prepare(): the lab's real control flow (sweep, retime, e2e, dispatch,
dequant blocks) and the real v2/v3 host code, with every Triton kernel emulated (same math as
check_v2_cpu / check_v3_cpu), tiny sizes, Marlin replaced by a dense FP32 reference, and timings faked
so R3 "wins": the lab must then switch classes on and kernel_fn must use them.

Run: .venv/bin/python prefill_layer/local/dryrun_lab_re3.py
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
for name in ("synchronize", "empty_cache"):
    setattr(torch.cuda, name, lambda *a, **k: None)

src = (DIR / "lab_re3.py").read_text().replace('"cuda"', '"cpu"')
ns = {"__name__": "lab_re3"}
exec(compile(src, "lab_re3.py", "exec"), ns)


# ---- kernel emulations (math as in check_v2_cpu / check_v3_cpu) ----
def e2m1_value(code):
    sign, exp, man = (code >> 3) & 1, (code >> 1) & 3, (code & 1).float()
    normal = (1.0 + 0.5 * man) * torch.where(exp == 3, 4.0, torch.where(exp == 2, 2.0, 1.0))
    mag = torch.where(exp == 0, 0.5 * man, normal)
    return torch.where(sign == 1, -mag, mag)


def emu_dequant(grid, q, sf, counts, out, N, KB, BN, BKB, num_warps):
    assert grid == (q.shape[0], N // BN, KB // BKB)
    for e in range(q.shape[0]):
        if int(counts[e]):
            qi, s = q[e].to(torch.int32), sf[e].float().repeat_interleave(8, -1)
            out[e] = torch.stack((e2m1_value(qi & 15) * s, e2m1_value(qi >> 4) * s), -1).reshape(N, 2 * KB).to(torch.bfloat16)


def emu_gather(grid, x, perm, offs, out, H, TOPK, NE, num_warps):
    n = int(offs[NE - 1])
    out[:n] = x[perm[:n] // TOPK]


def emu_act(grid, h, keys, gs, offs, out, I, NE, num_warps):
    n = int(offs[NE - 1])
    g = gs[keys[:n].long()].unsqueeze(-1)
    gate, up = h[:n, :I].float() * g, h[:n, I:2 * I].float() * g
    out[:n] = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def emu_combine(grid, y, ids, inv, gs, s, w, out, eps, H, TOPK, num_warps):
    wf = w.float()
    for t in range(grid[0]):
        acc = torch.zeros(H)
        for k in range(TOPK):
            e = int(ids.view(-1)[t * TOPK + k])
            if e >= 0:
                yy = y[int(inv[t * TOPK + k])].float() * gs[e] * wf
                acc += yy * torch.rsqrt((yy * yy).sum() / H + eps) * s.view(-1)[t * TOPK + k]
        out[t] = acc.to(torch.bfloat16)


def decode_rep(qrep, sf, bk):
    rows, kh = qrep.shape
    hb = bk // 2
    sfx = (sf.float() * 2.0**64).to(torch.bfloat16).float()
    out = torch.empty(rows, 2 * kh)
    j = torch.arange(hb)
    for k0 in range(0, 2 * kh, bk):
        c = nvfp4.unpack_e2m1(qrep[:, k0 // 2 + j]).view(rows, hb, 2)
        out[:, k0 + j] = ((c[..., 0] * 2.0**-126).to(torch.bfloat16).float() * sfx[:, (k0 + j) // 16]).to(torch.bfloat16).float()
        out[:, k0 + hb + j] = ((c[..., 1] * 2.0**-126).to(torch.bfloat16).float() * sfx[:, (k0 + hb + j) // 16]).to(torch.bfloat16).float()
    return out


class Compiled:
    n_regs, n_spills = 128, 0


R3_CALLS = [0]


def emu_r3(grid, a, q, sf, c, table, out, max_tiles, N, K, ROWS, GATED, SWAP, BT, BN, BK, SCALE2D, num_warps, num_stages):
    assert grid == ((N // BN) * max_tiles,) and K % BK == 0 and N % BN == 0
    R3_CALLS[0] += 1
    for t in range(max_tiles):
        e = int(table[0, t])
        if e < 0:
            continue
        rows = torch.arange(int(table[1, t]), int(table[1, t]) + BT)
        rows = rows[rows < int(table[2, t])]
        wd = decode_rep(q[e], sf[e], BK)
        x = a[rows].float()
        if GATED:
            g, u = (x @ wd[:N].T) * float(c[e]), (x @ wd[N:].T) * float(c[e])
            out[rows] = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
        else:
            out[rows] = ((x @ wd.T) * float(c[e])).to(torch.bfloat16)
    return Compiled()


def grouped_mm(a, b, offs, out_dtype):
    out, start = torch.full((a.shape[0], b.shape[-1]), float("nan"), dtype=out_dtype), 0
    for g, end in enumerate(offs.tolist()):
        out[start:end] = (a[start:end].float() @ b[g].float()).to(out_dtype)
        start = end
    return out


EMU.update(_nvfp4_dequant_kernel=emu_dequant, _gather_rows_kernel=emu_gather, _gmm_act_kernel=emu_act,
           _gmm_combine_kernel=emu_combine, _r3_gemm_kernel=emu_r3)
torch._grouped_mm = grouped_mm

# ---- tiny problem: I = H = 64, 32 local experts ----
I = H = 64
E = 32
gen = torch.Generator().manual_seed(0)
scales = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)
weights = {
    "w13": torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen),
    "w13_sf": scales[torch.randint(40, 70, (E, 2 * I, H // 16), generator=gen)],
    "w13_gs": torch.rand(E, generator=gen) * 1e-2 + 1e-3,
    "w2": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen),
    "w2_sf": scales[torch.randint(40, 70, (E, H, I // 16), generator=gen)],
    "w2_gs": torch.rand(E, generator=gen) * 1e-2 + 1e-3,
    "router": torch.randn(256, 64, generator=gen), "router_bias": torch.randn(256, generator=gen) * 0.01,
}
expert_map = torch.full((256,), -1, dtype=torch.int32)
expert_map[:E] = torch.arange(E, dtype=torch.int32)
post_w = (1 + 0.1 * torch.randn(I, generator=gen)).to(torch.bfloat16)


def dense_reference(x, w, ids, state):
    out = torch.zeros(x.shape[0], I)
    for e in range(E):
        hit = ids == e
        if not bool(hit.any()):
            continue
        W13 = nvfp4.dequantize(weights["w13"][e], weights["w13_sf"][e], 1.0) * weights["w13_gs"][e]
        W2 = nvfp4.dequantize(weights["w2"][e], weights["w2_sf"][e], 1.0) * weights["w2_gs"][e]
        tok, slot = hit.nonzero(as_tuple=True)
        h = x[tok].float() @ W13.T
        y = ((torch.nn.functional.silu(h[:, :I]) * h[:, I:]) @ W2.T) * post_w.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * w[tok, slot].unsqueeze(-1))
    return out.to(torch.bfloat16)


def fake_submission_prepare(w):
    state = {"bank": {"expert_map": expert_map}, "post_expert_w": post_w}
    ns["_gmm_prepare"](state, w)
    ns["_r3_prepare"](state, w)
    return state


def time_graph(self, fn, reps=20, cold=True):
    before = R3_CALLS[0]
    fn()
    return 1.0 if R3_CALLS[0] > before else 2.0  # R3 "wins" everywhere


ns.update(HIDDEN=64, D_EXPERT=I, _experts=dense_reference, _submission_prepare=fake_submission_prepare,
          GMM_SELFTEST_TOKENS=40, RE3_SWEEP_TOKENS=60, RE3_CLASS_TOKENS={"t1": 30, "t4": 60, "ft": 90},
          RE3_TEST_TOKENS={"t1": (12,), "t4": (45,), "ft": ()}, GMM_FROM_TOKENS=80, R3_CLASSES=((40, "t1"), (80, "t4")),
          RE3_BKS=(32, 64), DEQ_BLOCK_N=32, DEQ_BLOCK_KB=8,
          RE3_G1=[(0, 16, 32, 64, 4, 3, 0), (1, 32, 64, 32, 8, 3, 1), (0, 16, 64, 64, 4, 2, 0)],
          RE3_G2=[(0, 16, 32, 64, 4, 3, 0), (1, 32, 32, 32, 8, 3, 0)])
ns["Lab"].time_graph = time_graph
lines = []
ns["log"] = lambda m: lines.append(m)
state = ns["prepare"](weights)
text = "\n".join(lines)
print(text)
assert "FAILED" not in text and "WRONG" not in text, "lab section failed in the dry run"
assert state["gmm"]["ok"], "v2 self-test (vs the dense reference) should pass"
assert sorted(state["r3"]["cfg"]) == ["ft", "t1", "t4"], state["r3"]["cfg"]
assert "R3_CONFIGS = {" in text and "dequant all 32" in text
# kernel_fn's dispatch now picks R3 for every class.
before = R3_CALLS[0]
x = (torch.randn(30, I, generator=gen) * 0.5).to(torch.bfloat16)
ids, w = ns["synthetic_routing"](30, seed=1)
y = ns["_moe_v3"](x, w, ids, state, 30)
assert R3_CALLS[0] == before + 2
err = float((y.float() - dense_reference(x, w, ids, state).float()).norm() / dense_reference(x, w, ids, state).float().norm())
assert err < 1e-2, err
print(f"dryrun_lab_re3: ok (dispatch uses R3 after the lab, err {err:.1e})")

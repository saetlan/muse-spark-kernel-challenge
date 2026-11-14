"""CPU dry run of lab_re19.py's prepare(): the lab's real control flow (build report, unit tests with their
references and error localizer, end-to-end checks, dispatch, parts timing) and v4's real host code
(r14_pack, _r14_sizes, _experts_r14, _moe_v4, r14_check), with the CUDA extension replaced by fast dense
mocks (the kernels' own formulas are checked in check_v4_cpu.py), Triton kernels emulated, Marlin replaced
by a dense FP32 reference, 3 local experts at the real sizes and small token counts. Timings are faked so
R14 "wins": the lab must then switch classes on and _moe_v4 must use them.

Run: .venv/bin/python prefill_layer/local/dryrun_lab_re19.py
"""

import sys
import types
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvfp4  # noqa: E402
import r14_ports  # noqa: E402

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

src = (DIR / "lab_re19.py").read_text().replace('"cuda"', '"cpu"')
ns = {"__name__": "lab_re19"}
exec(compile(src, "lab_re19.py", "exec"), ns)

E, I, H = 3, 4096, 4096
gen = torch.Generator().manual_seed(0)
scales = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)
weights = {
    "w13": torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen),
    "w13_sf": scales[torch.randint(40, 70, (E, 2 * I, H // 16), generator=gen)],
    "w13_gs": torch.rand(E, generator=gen) * 1e-2 + 1e-3,
    "w2": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen),
    "w2_sf": scales[torch.randint(40, 70, (E, H, I // 16), generator=gen)],
    "w2_gs": torch.rand(E, generator=gen) * 1e-2 + 1e-3,
}
W13 = {e: nvfp4.dequantize(weights["w13"][e], weights["w13_sf"][e], 1.0) for e in range(E)}
W2 = {e: nvfp4.dequantize(weights["w2"][e], weights["w2_sf"][e], 1.0) for e in range(E)}
expert_map = torch.full((256,), -1, dtype=torch.int32)
expert_map[:E] = torch.arange(E, dtype=torch.int32)
post_w = (1 + 0.1 * torch.randn(I, generator=gen)).to(torch.bfloat16)


def routing(tokens, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(tokens, 8, dtype=torch.int32)
    for t in range(tokens):
        loc = [e for e in range(E) if bool(torch.rand(1, generator=g) < 0.6)]
        ids[t] = torch.tensor(loc + list(range(100, 100 + 8 - len(loc))), dtype=torch.int32)
    w = torch.softmax(torch.randn(tokens, 8, generator=g), -1)
    return ids, w


def dense(x, w, ids, state):
    out = torch.zeros(x.shape[0], I)
    for e in range(E):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if len(tok) == 0:
            continue
        h = x[tok].float() @ (W13[e] * weights["w13_gs"][e]).T
        y = ((torch.nn.functional.silu(h[:, :I]) * h[:, I:]) @ (W2[e] * weights["w2_gs"][e]).T) * post_w.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * w[tok, slot].unsqueeze(-1))
    return out.to(torch.bfloat16)


# ---- Triton emulations ----
def emu_r14_gather(grid, x, pos, out, P, H, TOPK, num_warps):
    out.copy_(r14_ports.gather(x, pos, P)) if out.shape[1] == P else None


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


def e2m1_value(code):
    sign, exp, man = (code >> 3) & 1, (code >> 1) & 3, (code & 1).float()
    normal = (1.0 + 0.5 * man) * torch.where(exp == 3, 4.0, torch.where(exp == 2, 2.0, 1.0))
    mag = torch.where(exp == 0, 0.5 * man, normal)
    return torch.where(sign == 1, -mag, mag)


def emu_dequant(grid, q, sf, counts, out, N, KB, BN, BKB, num_warps):
    for e in range(q.shape[0]):
        if int(counts[e]):
            qi, s = q[e].to(torch.int32), sf[e].float().repeat_interleave(8, -1)
            out[e] = torch.stack((e2m1_value(qi & 15) * s, e2m1_value(qi >> 4) * s), -1).reshape(N, 2 * KB).to(torch.bfloat16)


def emu_gather_rows(grid, x, perm, offs, out, H, TOPK, NE, num_warps):
    n = int(offs[NE - 1])
    out[:n] = x[perm[:n] // TOPK]


def emu_act(grid, h, keys, gs, offs, out, I, NE, num_warps):
    n = int(offs[NE - 1])
    g = gs[keys[:n].long()].unsqueeze(-1)
    gate, up = h[:n, :I].float() * g, h[:n, I:2 * I].float() * g
    out[:n] = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def grouped_mm(a, b, offs, out_dtype):
    out, start = torch.full((a.shape[0], b.shape[-1]), float("nan"), dtype=out_dtype), 0
    for g, end in enumerate(offs.tolist()):
        out[start:end] = (a[start:end].float() @ b[g].float()).to(out_dtype)
        start = end
    return out


EMU.update(_r14_gather_kernel=emu_r14_gather, _gmm_combine_kernel=emu_combine, _nvfp4_dequant_kernel=emu_dequant,
           _gather_rows_kernel=emu_gather_rows, _gmm_act_kernel=emu_act)
torch._grouped_mm = grouped_mm

R14_CALLS = [0]
DIAG_CALLS = [0]
FAIL_CL4 = [False]
PROBE = [0]
LAST_DIAG = [None]


class FakeExt:
    def r14_route(self, ids, pos, pieces, n_pieces, max_pieces, mode):
        p, pcs, _, _ = r14_ports.route(ids, max_pieces, mode)
        pos.copy_(p)
        for i, pc in enumerate(pcs):
            pieces[i] = torch.tensor(pc, dtype=torch.int32)
        n_pieces[0] = len(pcs)

    def r14_gemm(self, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, max_pieces, depth, smax, gated, cl,
                 diag):
        assert depth == 2 and smax >= 2 and cl == 1 and diag in (0, 3, 14, 20, 21, 24, 25)
        assert diag not in (20, 21, 24, 25) or wblob is not None
        if FAIL_CL4[0] and cl == 4:
            raise RuntimeError("r14_gemm launch failed: cluster out of resources (simulated)")
        if diag in (3, 21, 25):
            DIAG_CALLS[0] += 1
            assert err.numel() >= 64
            err[2:64].view(torch.int64).add_(torch.arange(31, dtype=torch.int64) + 1)
            return
        if diag in (1, 2, 18):
            DIAG_CALLS[0] += 1
            out.fill_(float("nan"))  # garbage, like the real diagnostic kernels
            err.add_(1)  # and possibly flags: the lab must clear them
            return
        R14_CALLS[0] += 1
        LAST_DIAG[0] = diag
        unsw = ns["re4_unswizzle"](bsrc, P)  # [P, K]
        for e_, row0, valid, n in pieces[:int(n_pieces[0])].tolist():
            xs = unsw[row0:row0 + valid].float()
            if gated:
                gs = float(cscale[e_]) / 2.0**62
                h = xs @ (W13[e_] * gs).T
                act = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(torch.bfloat16)
                p = torch.arange(row0, row0 + valid).view(-1, 1)
                col = torch.arange(I).view(1, -1)
                out[col // 64, p, (((col % 64) // 8) ^ (p & 7)) * 8 + col % 8] = act
            else:
                out[row0:row0 + valid] = (xs @ W2[e_].T).to(torch.bfloat16)

    def r14_attrs(self):
        return torch.tensor([168, 0, 160, 0, 168, 0, 160, 0, 168, 0, 160, 0, 66, 66, 32, 32])


def fake_submission_prepare(w):
    state = {"bank": {"expert_map": expert_map}, "post_expert_w": post_w}
    ns["_gmm_prepare"](state, w)
    ns["_r14_prepare"](state, w)
    return state


def time_graph(self, fn, reps=20, cold=True):
    before = R14_CALLS[0]
    LAST_DIAG[0] = None
    fn()
    if R14_CALLS[0] == before:
        return 2.0
    return 0.5 if LAST_DIAG[0] == 24 else 1.0  # R55 "wins": dispatch must pick it, kernel_fn must run it


def fake_build(probe=False):
    PROBE[0] += probe
    return FakeExt()


ns.update(_experts=dense, _submission_prepare=fake_submission_prepare, r14_build=fake_build,
          GMM_SELFTEST_TOKENS=24, RE4_UNIT_TOKENS=(16, 24), RE4_SMALL=16,
          RE4_CLASS_TOKENS={"t1": 24, "t4": 28, "ft": 32}, RE4_TEST_TOKENS={"t1": (20,), "t4": (26,), "ft": ()},
          R14_CLASSES=((26, "t1"), (30, "t4")), DEQ_BLOCK_N=32, DEQ_BLOCK_KB=128,
          re4_routing=lambda weights, tokens, kind, seed=None: routing(tokens, tokens + (0 if kind == "real" else 999)),
          _gmm_selftest_routing=lambda tokens, device, seed=0: routing(tokens, 500 + seed))
ns["Lab"].time_graph = time_graph
lines = []
ns["log"] = lambda m: lines.append(m)
state = ns["prepare"](weights)
text = "\n".join(lines)
print(text)
assert "FAILED" not in text and "WRONG" not in text, "lab section failed in the dry run"
assert "units" not in text or "gemm1 locate" not in text
assert state["r14"]["cfg"] == {"t1": (2, 8, 1, 1, 24), "t4": (2, 8, 1, 1, 24)}, state["r14"]["cfg"]  # ft: grouped
assert "dispatch ft: stays grouped" in text
assert all(len(c) in (4, 5) for c in state["r14"]["cfg"].values()), state["r14"]["cfg"]
assert DIAG_CALLS[0] > 0 and PROBE[0] == 0, (DIAG_CALLS, PROBE)
assert text.count("prof T=") == 24 and "per chunk: issue" in text and "decoder per chunk" in text, "profile lines missing"
assert text.count("r55 g1") == 4 and text.count("r53 g2") == 4, "profile kinds missing"
assert int(state["r14"]["err"].item()) == 0, "diagnostic flags leaked into the error counter"
assert "===== summary =====" in text and text.rindex("R14_CONFIGS") > text.index("===== summary =====")
before = R14_CALLS[0]
x = (torch.randn(24, I, generator=gen) * 0.5).to(torch.bfloat16)
ids, w = routing(24, 7)
LAST_DIAG[0] = None
y = ns["_moe_v4"](x, w, ids, state, 24)
assert R14_CALLS[0] == before + 2 and LAST_DIAG[0] == 24 and int(state["r14"]["err"].item()) == 0
err = float((y.float() - dense(x, w, ids, state).float()).norm() / dense(x, w, ids, state).float().norm())
assert err < 1e-2, err
print(f"dryrun_lab_re19: ok (dispatch uses R14 after the lab, err {err:.1e})")

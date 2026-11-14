"""CPU dry run of proj/lab_e1.py: the whole lab and kernel_fn, with Triton replaced by PyTorch
emulations of _swap_kernel/_gemm_kernel that follow the launch exactly (grid, split-K ranges,
modes, atomics vs stores, output dtype), SGLang's GEMM by F.linear, the image kernels by a planted
failure, and time_graph by a fake timer. Also checks that an empty CHOICE is baseline.py's path.

Run: .venv/bin/python proj/local/dryrun_lab_e1.py   (after make_lab.py has built proj/lab_e1.py)
"""

import hashlib
import importlib.util
import tempfile
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
# One scratch directory per tree, so mutation_test.py can run several sandboxes at once.
SCRATCH = Path(tempfile.gettempdir()) / ("okc_proj_dryrun_" + hashlib.sha1(str(PROJ).encode()).hexdigest()[:12])


class _Kernel:
    def __init__(self, fn):
        self.name = fn.__name__

    def __getitem__(self, grid):
        return lambda *a, **k: EMULATE[self.name](grid, *a, **k)


def _jit(fn=None, **_):
    return _Kernel(fn) if fn is not None else (lambda f: _Kernel(f))


def _partial(xs, ws, mode):
    if mode == 0:
        return xs.float() @ ws.float().T
    if mode == 1:
        hi = ws.to(torch.bfloat16)
        lo = (ws - hi.float()).to(torch.bfloat16)
        return xs.float() @ hi.float().T + xs.float() @ lo.float().T
    return (xs.double() @ ws.double().T).float()


def _emulate(grid, x, w, out, M, N, kps, BM, BN, BK, MODE, ATOMIC, OUT_BF16, swap):
    assert kps % BK == 0, (kps, BK)
    tiles_n = grid[0] if swap else grid[1]
    splits = grid[1] if swap else grid[2]
    assert tiles_n * BN >= N and (tiles_n - 1) * BN < N, (tiles_n, BN, N)
    assert splits * kps == x.shape[1], (splits, kps, x.shape)
    if swap:
        assert M <= BM, (M, BM)
    else:
        assert grid[0] * BM >= M
    assert out.dtype == (torch.float32 if (ATOMIC or not OUT_BF16) else torch.bfloat16)
    for s in range(splits):
        part = _partial(x[:, s * kps:(s + 1) * kps], w[:N, s * kps:(s + 1) * kps], MODE)
        if ATOMIC:
            out += part
        else:
            assert splits == 1
            out.copy_(part.to(out.dtype))


def _swap(grid, w, x, out, M, N, kps, sw, sx, so, BM, BN, BK, MODE, ATOMIC, OUT_BF16, num_warps, num_stages):
    assert sw == w.stride(0) and sx == x.stride(0) and so == out.stride(0)
    _emulate(grid, x, w, out, M, N, kps, BM, BN, BK, MODE, ATOMIC, OUT_BF16, True)


def _gemm(grid, x, w, out, M, N, kps, sx, sw, so, BM, BN, BK, MODE, ATOMIC, OUT_BF16, num_warps, num_stages):
    assert sw == w.stride(0) and sx == x.stride(0) and so == out.stride(0)
    _emulate(grid, x, w, out, M, N, kps, BM, BN, BK, MODE, ATOMIC, OUT_BF16, False)


def _fma(grid, w, x, out, N, sw, sx, so, M, BN, BK, ATOMIC, OUT_BF16, num_warps):
    assert sw == w.stride(0) and sx == x.stride(0) and so == out.stride(0)
    assert M == x.shape[0] and BK & (BK - 1) == 0, (M, BK)
    tiles_n, splits = grid
    assert tiles_n * BN >= N and (tiles_n - 1) * BN < N, (tiles_n, BN, N)
    assert splits * BK == x.shape[1], (splits, BK, x.shape)
    assert out.dtype == (torch.float32 if (ATOMIC or not OUT_BF16) else torch.bfloat16)
    assert ATOMIC == (splits > 1)
    for s in range(splits):
        part = (x[:, s * BK:(s + 1) * BK].double() @ w[:N, s * BK:(s + 1) * BK].double().T).float()
        if ATOMIC:
            out += part
        else:
            out.copy_(part.to(out.dtype))


EMULATE = {"_swap_kernel": _swap, "_gemm_kernel": _gemm, "_fma_kernel": _fma}


def install_stubs():
    triton = types.ModuleType("triton")
    triton.jit = _jit
    triton.cdiv = lambda a, b: -(-a // b)
    triton.__version__ = "stub"
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    triton.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = triton, tl
    names = ["sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.quantization",
             "sglang.srt.layers.quantization.unquant", "sglang.kernels", "sglang.kernels.ops",
             "sglang.kernels.ops.gemm", "sglang.kernels.ops.gemm.tiny_gemm", "sglang.kernels.ops.gemm.hopper_bf16_gemv"]
    for n in names:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["sglang.srt.layers.quantization.unquant"]._bf16_gemm_dispatch_impl = lambda x, w, b: F.linear(x, w, b)
    tiny = sys.modules["sglang.kernels.ops.gemm.tiny_gemm"]
    tiny.can_use_tiny_gemm = lambda n, k, max_m=16: True

    def planted(*a, **k):
        raise RuntimeError("planted image-kernel failure")
    tiny.tiny_gemm_bf16 = planted
    sys.modules["sglang.kernels.ops.gemm.hopper_bf16_gemv"].hopper_bf16_gemv = lambda x, w: F.linear(x, w)
    mm = torch.mm

    def mm_cpu(a, b, out_dtype=None):  # mm.dtype has no CPU kernel; same math in FP32
        return mm(a.float(), b.float()).to(out_dtype) if out_dtype is not None else mm(a, b)
    torch.mm = mm_cpu
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.get_device_name = lambda *a, **k: "cpu"
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=132)


def load(path, name):
    src = path.read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    SCRATCH.mkdir(exist_ok=True)
    p = SCRATCH / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights():
    gen = torch.Generator().manual_seed(5151)
    dims = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
            "router": (256, 8192)}
    return {k: (torch.randn((n, kk), generator=gen) / math.sqrt(kk)).to(torch.float32 if k == "router" else torch.bfloat16)
            for k, (n, kk) in dims.items()}


def main():
    install_stubs()
    w = weights()
    base = load(PROJ / "baseline.py", "baseline_cpu")
    sub = load(PROJ / "submission.py", "submission_cpu")
    bstate, sstate = base.prepare(w), sub.prepare(w)
    for layer, m in (("router", 8), ("router", 9), ("router", 256), ("unpooler", 5), ("unpooler", 130), ("o", 33)):
        n = 1024 if layer == "unpooler" and m <= 8 else w[layer].shape[0]
        x = torch.randn((m, w[layer].shape[1])).to(torch.bfloat16)
        data = {"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}
        a, b = base.kernel_fn(data, bstate), sub.kernel_fn(data, sstate)
        assert a.dtype == b.dtype and torch.equal(a, b), (layer, m)
    print("ok: an empty CHOICE gives baseline.py's outputs bit for bit")

    lab = load(PROJ / "lab_e1.py", "lab_e1_cpu")
    calls = {"timed": 0}

    def fake_time(self, fn, reps=20, cold=True):
        calls["timed"] += 1
        fn()
        return 10.0 + (calls["timed"] % 13)

    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_BUDGET_S = 1e9
    # Keep the CPU run short: two configs per shape, but every kind, mode and split path.
    full = lab.triton_candidates

    def few(layer, m, n, k):
        c = full(layer, m, n, k)
        atomic = [x for x in c if x["split"] > 1][:1]
        direct = [x for x in c if x["split"] == 1][:1]
        return direct + atomic
    lab.triton_candidates = few
    state = lab.prepare(w)
    assert len(lab.CHOICE) > 0, "dry run chose nothing"
    kinds = {c["kind"] for c in lab.CHOICE.values()}
    assert "tiny" not in kinds, "a failing kernel was chosen"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 7)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    modes = {(c["kind"], c["mode"], c["split"] > 1) for c in lab.CHOICE.values() if "mode" in c}
    print(f"ok: lab_e1.prepare dry run; {calls['timed']} timings; CHOICE {len(lab.CHOICE)} shapes; kinds {sorted(kinds)}; "
          f"(kind, mode, split) seen {sorted(modes)}")


if __name__ == "__main__":
    main()

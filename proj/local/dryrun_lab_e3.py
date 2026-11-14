"""CPU dry run of proj/lab_e3.py (v2 + lab/sweep.py + lab/e3.py) with Triton emulated
(dryrun_lab_e1's stubs), the three microbenchmark modules, FlashInfer, DeepGEMM and SGLang's router
kernel mocked, and a fake timer. Shapes are scaled down (same structure, smaller K and N) so the
FP64 gates run fast on the CPU; v2's own configs still fit them.

Checks: every lab section runs; the cluster microbenchmark's correctness check passes on a correct
reduction; a planted DeepGEMM native failure costs only a log line; a planted wrong FlashInfer
backend (cudnn) is rejected by the gate even though the timer makes it fastest; the lab moves the
shapes the timer favors to the right library; every final choice is correct; DeepGEMM's
deterministic flag is left off when no native kernel is chosen.

Run: .venv/bin/python proj/local/dryrun_lab_e3.py   (after make_lab.py has built proj/lab_e3.py)
"""

import contextlib
import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402

SMALL = {"qkv_gate": (288, 1024), "o": (1024, 128), "pooler": (512, 1024), "unpooler": (1024, 512),
         "router": (256, 4096)}
SLICE = 128  # unpooler rows at m <= 8 in the scaled shapes


def mock_modules(calls, planted=True):
    def ext_load_inline(name, functions, **kw):
        part = name.rsplit("_", 1)[-1]
        mod = types.SimpleNamespace()
        if part == "core":
            mod.mb_run_ldg = lambda *a: calls.__setitem__("ldg", calls.get("ldg", 0) + 1)
            mod.mb_run_empty = lambda *a: None
            mod.mb_run_int = lambda *a: None
        elif part == "bulk":
            mod.mb_run_bulk = lambda *a: calls.__setitem__("bulk", calls.get("bulk", 0) + 1)
            mod.mb_bulk_attrs = lambda: None
        else:
            def run_cluster(src, out, ctas, c, n_per):
                out.copy_(src.view(ctas // c, c, n_per).sum(1).flatten())
                calls["cluster"] = calls.get("cluster", 0) + 1
            mod.mb_run_cluster = run_cluster
        assert sorted(functions) == sorted(k for k in vars(mod)), (part, functions)
        return mod
    if planted:
        cpp = types.ModuleType("torch.utils.cpp_extension")
        cpp.load_inline = ext_load_inline
        sys.modules["torch.utils.cpp_extension"] = cpp

    fi = types.ModuleType("flashinfer")

    def mm_bf16(a, b, bias=None, pdl=False, out=None, out_dtype=torch.bfloat16, backend="cudnn"):
        calls["fi_" + backend] = calls.get("fi_" + backend, 0) + 1
        r = (a.double() @ b.double()).to(out_dtype)
        if planted and backend == "cudnn":  # planted: a wrong backend
            r = (r.float() * 1.05).to(out_dtype)
        return r
    fi.mm_bf16 = mm_bf16
    fi.autotune = lambda tune_mode=True, **kw: contextlib.nullcontext()
    sys.modules["flashinfer"] = fi

    dg = types.ModuleType("deep_gemm")
    dg.state = {"det": False}
    dg.use_deterministic_algorithms = lambda v: dg.state.__setitem__("det", bool(v))
    dg.get_deterministic_algorithms = lambda: dg.state["det"]

    def bf16_gemm_nt(a, b, out, c=None, compiled_dims="", alpha=None):
        if planted and dg.state["det"]:
            raise RuntimeError("planted DeepGEMM native failure")
        out.copy_((a.double() @ b.double().T).to(out.dtype))
    dg.bf16_gemm_nt = bf16_gemm_nt
    sys.modules["deep_gemm"] = dg

    rg = types.ModuleType("sglang.kernels.ops.muse_spark_v12.router_gemm")
    rg.router_gemm = lambda x, w_t, out=None: (x.double() @ w_t.double()).float()
    sys.modules["sglang.kernels.ops.muse_spark_v12"] = types.ModuleType("sglang.kernels.ops.muse_spark_v12")
    sys.modules["sglang.kernels.ops.muse_spark_v12.router_gemm"] = rg
    return dg


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: torch.nn.functional.linear(x, w)
    calls = {}
    dg = mock_modules(calls)
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=132, L2_cache_size=60 << 20)
    gen = torch.Generator().manual_seed(5151)
    w = {k: (torch.randn((n, kk), generator=gen) / math.sqrt(kk)).to(torch.float32 if k == "router" else torch.bfloat16)
         for k, (n, kk) in SMALL.items()}
    lab = d.load(d.PROJ / "lab_e3.py", "lab_e3_cpu")
    lab.FULL.clear()
    lab.FULL.update(SMALL)
    lab.lab_n = lambda layer, m: SLICE if layer == "unpooler" and m <= 8 else SMALL[layer][0]
    lab.MB_SIZES = (("8.4MB", 1024 * 4096 * 2), ("67.1MB", 4096 * 8192 * 2))
    v2_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        # Planted cudnn always "fastest" (must be rejected); otherwise FlashInfer cuBLASLt wins at
        # m <= 32 and DeepGEMM's cuBLASLt at m >= 128, so both library paths must work.
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        x = next((v for v in cells if isinstance(v, torch.Tensor) and v.dim() == 2), None)
        if c is None or x is None:
            return 20.0
        m = x.shape[0]
        if c["kind"] == "fi_mm":
            return 1.0 if c["backend"] == "cudnn" else 5.0 if (c["backend"] == "cublaslt" and m <= 32) else 30.0
        if c["kind"] == "dg" and not c["native"]:
            return 4.0 if m >= 128 else 30.0
        return 30.0 if c["kind"] in ("dg", "sgl_router") else 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(w)

    assert calls.get("ldg", 0) > 10 and calls.get("bulk", 0) > 0 and calls.get("cluster", 0) == 6, calls
    assert calls.get("fi_cudnn", 0) > 0 and calls.get("fi_cublaslt", 0) > 0, calls
    new = {k: c for k, c in lab.CHOICE.items() if v2_choice.get(k) != c}
    bf16_shapes = {(layer, m) for layer in ("unpooler", "o", "qkv_gate", "pooler") for m in (1, 8, 16, 32, 128, 256)}
    assert set(new) == bf16_shapes, sorted(set(new) ^ bf16_shapes)
    want = {k: ({"kind": "fi_mm", "backend": "cublaslt"} if k[1] <= 32 else {"kind": "dg", "native": False}) for k in new}
    assert new == want, sorted((k, new[k]) for k in new if new[k] != want[k])
    assert not dg.state["det"], "DeepGEMM deterministic mode left on"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: lab_e3 dry run; microbenchmarks ran ({calls.get('ldg')} ldg, {calls.get('bulk')} bulk, "
          f"{calls.get('cluster')} cluster calls); planted wrong backend rejected, planted DeepGEMM failure contained; "
          f"{len(new)} shapes moved to the fastest correct library (FlashInfer cuBLASLt at m <= 32, DeepGEMM cuBLASLt at m >= 128)")


if __name__ == "__main__":
    main()

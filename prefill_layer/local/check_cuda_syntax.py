"""Syntax- and type-check R14's CUDA source (v4) on the Mac with Apple clang (no nvcc here); a copy of
moe/local/check_cuda_syntax.py adapted to prefill (stub headers in prefill_layer/local/cuda_stubs).
It parses host and sm_90a device code; it can't check the PTX inside asm strings.

Run: .venv/bin/python prefill_layer/local/check_cuda_syntax.py
"""

import subprocess
import sys
import tempfile
import types
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
DIR = LOCAL.parent


class _Any(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})


def cuda_source():
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
    triton.jit = lambda fn=None, **k: (fn if fn is not None else (lambda f: f))
    tl = _Any("triton.language")
    triton.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = triton, tl
    ns = {"__name__": "v6"}
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DIR / "submission.py"   # optional file to check
    exec(compile(path.read_text(), path.name, "exec"), ns)
    return ns["r14_source"], ns["_R14_CPP"]


def main():
    source, cpp = cuda_source()
    import inspect
    has_probe = "probe" in inspect.signature(source).parameters  # v6+; v5's r14_source() has no probe build
    check(source(), cpp, "full")
    if has_probe:
        check(source(True), cpp, "probe")


def check(cuda, cpp, label):
    src = '#include "cuda_stub_prelude.h"\n' + cuda + "\n// host declarations as load_inline's cpp_sources\n" + cpp
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "r14.cu"
        cu.write_text(src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
               "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(LOCAL / "cuda_stubs"), str(cu)]
        res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or "error" in out:
        print(out[-6000:])
        raise SystemExit("check_cuda_syntax: clang found problems in R14's CUDA source")
    if out:
        print(out[-3000:])
    print(f"ok ({label}): R14 CUDA source parses for host and sm_90a device (clang -fsyntax-only, stub headers)")


if __name__ == "__main__":
    main()

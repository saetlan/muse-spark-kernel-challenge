"""Syntax- and type-check the CUDA strings in an attention file with Apple clang (no nvcc here).
Adapted from moe/local/check_cuda_syntax.py: clang's CUDA front end parses host and device code
(-fsyntax-only, sm_90a) against stub headers in attention/local/cuda_stubs/. It can't check PTX
inside asm strings or what nvcc's real headers declare; those only show on the GPU.

Run: .venv/bin/python attention/local/check_cuda_syntax.py <file.py> <VAR> [<VAR> ...]
VAR = module-level string holding CUDA source; "#include <torch/extension.h>" is replaced by
the stubs. Imports the file with a stub Triton (emulate.load_submission).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))


def main(path, names):
    if os.path.basename(path).startswith(("cuda_", "_combined")):   # plain modules with source strings
        import importlib.util
        spec = importlib.util.spec_from_file_location("cuda_src_under_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        from emulate import load_submission
        mod = load_submission(path)
    for name in names:
        src = getattr(mod, name).replace("#include <torch/extension.h>", "")
        src = src.replace("#include <ATen/cuda/CUDAContext.h>", "").replace("#include <cmath>", "")
        src = ('#include "cuda_stub_prelude.h"\n#include "ATen/cuda/CUDAContext.h"\n#include <math.h>\n' + src)
        with tempfile.TemporaryDirectory() as tmp:
            cu = Path(tmp) / f"{name}.cu"
            cu.write_text(src)
            cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
                   "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(LOCAL / "cuda_stubs"), str(cu)]
            res = subprocess.run(cmd, capture_output=True, text=True)
        out = (res.stdout + res.stderr).strip()
        if res.returncode != 0 or "error" in out:
            print(out[-6000:])
            raise SystemExit(f"check_cuda_syntax: clang found problems in {path}:{name}")
        if out:
            print(out[-3000:])
        print(f"ok: {Path(path).name}:{name} parses for host and sm_90a device (clang -fsyntax-only, stubs)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

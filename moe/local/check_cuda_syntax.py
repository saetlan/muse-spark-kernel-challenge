"""Syntax- and type-check M11's CUDA source on the Mac with Apple clang (no nvcc here).

clang's CUDA front end parses host and device code (-fsyntax-only) against small stub headers
(moe/local/cuda_stubs/), so C++ mistakes, template errors, undeclared names and asm constraint
syntax are caught. It can't check the PTX inside asm strings or what nvcc's real headers
declare; those only show on the GPU.

Run: .venv/bin/python moe/local/check_cuda_syntax.py [moe/submission_m11.py]
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
MOE = LOCAL.parent
sys.path.insert(0, str(LOCAL))


def cuda_source(path, nowg=False):
    """The source m11_build() compiles: m11_source() of the submission module, imported with stubs."""
    import check_m2
    import check_v0_lab as base
    check_m2.install_stubs()
    ext = type(sys)("torch.utils.cpp_extension")
    ext.load_inline = None
    sys.modules["torch.utils.cpp_extension"] = ext
    import importlib.util
    spec = importlib.util.spec_from_file_location("m11_under_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if nowg:  # m13_setup's fallback build: no M16/M17 instances, wgmma asm compiled out
        return mod.m13_source(m16_instances=())
    return mod.m13_source() if hasattr(mod, "m13_source") else mod.m11_source()


def main(path, nowg=False):
    src = '#include "cuda_stub_prelude.h"\n' + cuda_source(path, nowg)
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "m11.cu"
        cu.write_text(src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
               "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(LOCAL / "cuda_stubs"), str(cu)]
        if nowg:
            cmd.insert(-1, "-DM13_NO_WGMMA")
        res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or "error" in out:
        print(out[-6000:])
        raise SystemExit(f"check_cuda_syntax: clang found problems in {path}")
    if out:
        print(out[-3000:])
    print(f"ok: {Path(path).name}{' (fallback build without wgmma)' if nowg else ''} CUDA source parses for host and "
          "sm_90a device (clang -fsyntax-only, stub headers)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--nowg"]
    main(args[0] if args else MOE / "submission_m11.py", "--nowg" in sys.argv)

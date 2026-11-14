"""clang -fsyntax-only (host + sm_90a) of dgrs.py's CUDA, all token-tile widths, against decode_layer s stub headers.
Run: .venv/bin/python shared/gemm/dg12rs/check_dgrs_syntax.py"""
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUBS = ROOT / "decode_layer" / "local" / "cuda_stubs"
sys.modules.setdefault("torch.utils.cpp_extension", types.SimpleNamespace(load_inline=None))
ns = {"__name__": "dgrs_check"}
src = (HERE / "dgrs.py").read_text().replace("from torch.utils.cpp_extension import load_inline", "load_inline = None")
exec(compile(src, "dgrs.py", "exec"), ns)
EXTRA = ("#ifndef __grid_constant__\n#define __grid_constant__\n#endif\n"
         "__device__ int atomicExch(int* address, int val);\n"
         "__device__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift);\n"
         "__device__ float fmaf(float a, float b, float c);\n"
         "__device__ unsigned int __float_as_uint(float x);\n"
         "__host__ __device__ uint2 make_uint2(unsigned int x, unsigned int y);\n"
         "__device__ float __frcp_rn(float x);\n"
         "__device__ int __ffs(int x);\n"
         )
INSTS = ns["DGRS_INSTANCES"] + ((32, 2, 0), (64, 2, 0), (64, 1, 1), (64, 1, 2), (64, 1, 4), (64, 1, 8), (32, 1, 15),
         (32, 1, 0, 4), (64, 1, 0, 4), (32, 1, 16, 4), (64, 1, 16), (32, 1, 2, 4), (32, 1, 32, 4), (128, 1, 32), (32, 1, 64, 4), (32, 1, 80, 4), (32, 1, 96, 4), (64, 1, 192, 4), (128, 1, 64), (32, 1, 256, 4), (32, 1, 320, 4), (64, 1, 256, 4), (32, 1, 352, 4), (32, 1, 512, 4), (64, 1, 512, 4), (128, 1, 512), (32, 1, 576, 4), (32, 1, 528, 4), (32, 1, 512), (32, 1, 512, 4, 2, 2), (32, 1, 576, 4, 2, 2), (32, 1, 0, 4, 2, 2), (32, 1, 512, 4, 0, 3), (32, 1, 512, 4, 1, 4), (64, 1, 512, 4, 2, 2), (128, 1, 512, 2, 0, 4), (32, 1, 0, 4, 2, 2, 8, 32, 8192, 4096), (32, 1, 0, 4, 0, 3, 8, 32, 4096, 8192), (128, 1, 0, 2, 0, 2, 8, 128, 8192, 1024), (32, 1, 64, 4, 2, 2, 8, 32, 8192, 4096))
code = '#include "cuda_stub_prelude.h"\n' + EXTRA + ns["dgrs_source"](INSTS) + "\n" + ns["_DGRS_CPP"]
with tempfile.TemporaryDirectory() as tmp:
    cu = Path(tmp) / "dgrs.cu"
    cu.write_text(code)
    r = subprocess.run(["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
                        "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(HERE / "stubs"), str(cu)],
                       capture_output=True, text=True)
out = (r.stdout + r.stderr).strip()
if r.returncode != 0 or "error" in out:
    print(out[-5000:])
    raise SystemExit("check_dgrs_syntax: clang found problems")
if out:
    print(out[-1500:])
print("ok: dgrs.py CUDA (BM 32/64/128, CPS 2, DIAG 1/2/4/8/15/16/32/64/80/96/192/256/320/352/512/528/576, MATH 4, RED/SPL and v10 SMX/M/N/K specializations) parses for host and sm_90a device (clang -fsyntax-only, stub headers)")

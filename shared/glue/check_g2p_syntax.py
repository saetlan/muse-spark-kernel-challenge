"""clang -fsyntax-only (host + sm_90a device) of shared/glue/g2p.py's CUDA, every instance, against decode_layer's
stub headers (read-only use of decode_layer/local/cuda_stubs). Also checks the MoE variant (g2p_moe.py) if present.

Run: .venv/bin/python shared/glue/check_g2p_syntax.py
"""

import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUBS = ROOT / "decode_layer" / "local" / "cuda_stubs"


def load(path):
    ns = {"__name__": "g2p_check", "torch": types.SimpleNamespace(), "load_inline": None, "tempfile": tempfile,
          "M11_MAX_TOKENS": 320, "POST_NORM_EPS": 1e-8}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


# Real CUDA intrinsics decode_layer's stubs don't declare (signatures as in the CUDA C++ Programming Guide).
G2P_STUBS = "__device__ int atomicExch(int* address, int val);\n"


def check(cuda, cpp, label, extra=""):
    src = '#include "cuda_stub_prelude.h"\n' + G2P_STUBS + extra + cuda + "\n// host declarations\n" + cpp
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "g2p.cu"
        cu.write_text(src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
               "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(STUBS), str(cu)]
        res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or "error" in out:
        print(out[-6000:])
        raise SystemExit(f"check_g2p_syntax: clang found problems in {label}")
    if out:
        print(out[-2000:])
    print(f"ok: {label} parses for host and sm_90a device (clang -fsyntax-only, decode_layer's stub headers)")


if __name__ == "__main__":
    ns = load(HERE / "g2p.py")
    check(ns["g2p_source"](), ns["_G2P_CPP"], "g2p.py (decode_layer variant)")
    moe = HERE / "g2p_moe.py"
    if moe.exists():
        ns = load(moe)
        check(ns["g2p_moe_source"](), ns["_M11F_CPP"], "g2p_moe.py (MoE variant)", ns["G2P_MOE_TEST_PRELUDE"])
    sys.exit(0)

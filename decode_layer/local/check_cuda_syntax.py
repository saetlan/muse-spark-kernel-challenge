"""Syntax- and type-check decode_layer CUDA sources on the Mac with Apple clang (no nvcc here):
clang's CUDA front end parses host and device code (-fsyntax-only, sm_90a) against stub headers in
decode_layer/local/cuda_stubs/ (copied from moe/local/cuda_stubs and extended). It can't check the
PTX inside asm strings or what nvcc's real headers declare.

Run: .venv/bin/python decode_layer/local/check_cuda_syntax.py   (checks parts/x5.py's _X5_CUDA)
"""

import subprocess
import sys
import tempfile
import types
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
DECODE = LOCAL.parent


def x5_source():
    mod = types.ModuleType("x5_src")
    mod.load_inline = None
    exec(compile((DECODE / "parts" / "x5.py").read_text(), "parts/x5.py", "exec"), mod.__dict__)
    return mod.x5_source()


def check(name, src):
    src = '#include "cuda_stub_prelude.h"\n' + src
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / f"{name}.cu"
        cu.write_text(src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
               "-std=c++20", "-Wall", "-Wno-unused-variable", "-I", str(LOCAL / "cuda_stubs"), str(cu)]
        res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or "error" in out:
        print(out[-6000:])
        raise SystemExit(f"check_cuda_syntax: clang found problems in {name}")
    if out:
        print(out[-2000:])
    print(f"ok: {name} parses for host and sm_90a device (clang -fsyntax-only, stub headers)")


def part_module(name):
    sys.path.insert(0, str(LOCAL))
    import dryrun_d0
    if "triton" not in sys.modules:
        dryrun_d0.install_stubs()
    mod = types.ModuleType(f"{name}_src")
    mod.load_inline = None
    mod._v9_timed = lambda key, fn: fn  # parts that time their builds (v14, v16) reference the glue's wrapper
    import torch
    mod.torch = torch
    stub_tl = types.SimpleNamespace(constexpr=lambda x=None: x)
    mod.tl = stub_tl
    mod.triton = types.SimpleNamespace(jit=lambda fn=None, **k: (fn if fn else (lambda f: f)))
    exec(compile((DECODE / "parts" / f"{name}.py").read_text(), f"parts/{name}.py", "exec"), mod.__dict__)
    return mod


if __name__ == "__main__":
    check("x5", x5_source())
    check("v3_prefetch", part_module("v3").v3_source())
    m11 = part_module("m11_e3c")
    check("m11_e3c (v3 build: M11 instances, no M12)", m11.m11_source(m11.M11_INSTANCES, ()))
    m11pf = part_module("m11_pf")
    check("m11_pf (L8 next-task L2 prefetch: M11 instances, no M12)", m11pf.m11_source(m11pf.M11_INSTANCES, ()))
    check("l9 (L2 staging kernels)", part_module("l9").l9_source())
    m11pdl = part_module("m11_pdl")
    check("m11_pdl (M11 + MoE E15-E16 PDL GEMM2 + CUDA post-norm: M11 instances, no M12)",
          m11pdl.m11_source(m11pdl.M11_INSTANCES, ()))
    moe19 = part_module("moe19")  # MoE E19 subset; v16 builds only its two M19 instances, wgmma compiled out
    v16_src = moe19.moe_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(),
                                   m16_instances=(), m17_instances=(), m18_instances=(),
                                   m19_instances=((1, 2, 2, 2, 4), (0, 2, 2, 4, 4)))
    check("moe19 (v16, MoE E19 M19 GEMM1 and GEMM2 + M13 post-norm, -DM13_NO_WGMMA)", "#define M13_NO_WGMMA 1\n" + v16_src)
    v14 = part_module("v14")  # G2P (shared/glue/g2p.py): M11 GEMM2 + post-norm fused, its four instances
    v14.M11_MAX_TOKENS = 320
    check("v14 G2P fused GEMM2 + post-norm", v14.g2p_source())
    moe25 = part_module("moe25")  # MoE E25 subset; v17 builds 4 M11b + 2 M19 instances, CR / SR / G2P emptied, no wgmma
    moe25.m25_M11B_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4))
    moe25.m25_M13_M11_CR_INSTANCES = moe25.m25_M13_M11_SR_INSTANCES = moe25.m25_M11F_INSTANCES = ()
    v17_src = moe25.m25_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(), m16_instances=(),
                                   m17_instances=(), m18_instances=(), m19_instances=((1, 2, 2, 2, 4), (0, 2, 2, 4, 4)))
    check("moe25 (v17, MoE E25 M11b x4 + M19 x2 + M13 post-norm, -DM13_NO_WGMMA)", "#define M13_NO_WGMMA 1\n" + v17_src)
    moe36 = part_module("moe36")  # MoE E36 subset; v22 builds the same 4 M11b (with sel) + 4 M19 (cold + hot), no wgmma
    moe36.m36_M11B_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4))
    for name in ("M11B_OCC_INSTANCES", "M11B_EARLY_INSTANCES", "M11B_PF_INSTANCES", "M11B_NU_INSTANCES", "M11B_RECON_SHAPES",
                 "M13_M11_CR_INSTANCES", "M13_M11_SR_INSTANCES", "M11F_INSTANCES"):
        setattr(moe36, "m36_" + name, ())
    v22_src = moe36.m36_m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(), m16_instances=(),
                                   m17_instances=(), m18_instances=(), m19_instances=((1, 2, 2, 2, 4), (0, 2, 2, 4, 4),
                                                                                       (1, 2, 4, 2, 3), (0, 2, 4, 2, 4)))
    check("moe36 (v22, MoE E36 M11b x4 + M19 x4 + M13 post-norm, -DM13_NO_WGMMA)", "#define M13_NO_WGMMA 1\n" + v22_src)

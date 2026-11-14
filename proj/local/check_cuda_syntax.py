"""Syntax- and type-check proj's CUDA sources on the Mac with Apple clang (no nvcc here).

clang's CUDA front end parses host and device code (-fsyntax-only, sm_90a) against stub headers
(proj/local/cuda_stubs/, copied from moe/local/cuda_stubs/ and extended), so C++ mistakes, template
errors, undeclared names and asm constraint syntax are caught. It can't check the PTX inside asm
strings or what nvcc's real headers declare; those only show on the GPU.

Sources: every value of a module-level dict or string named *_SRC in the given lab/submission
files (read with ast, nothing imported).

Run: .venv/bin/python proj/local/check_cuda_syntax.py proj/lab/e3.py [more files]
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent


def sources(path):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.endswith("_SRC"):
            val = ast.literal_eval(node.value)
            items = val.items() if isinstance(val, dict) else [("", val)]
            for key, src in items:
                yield f"{node.targets[0].id}[{key}]" if key else node.targets[0].id, src


def check(name, src):
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "src.cu"
        cu.write_text('#include "cuda_stub_prelude.h"\n' + src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "-nocudainc", "-nocudalib", "-fsyntax-only",
               "-std=c++20", "-Wall", "-Wno-unused-variable", "-Wno-sign-compare", "-I", str(LOCAL / "cuda_stubs"), str(cu)]
        res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or "error" in out:
        print(out[-6000:])
        raise SystemExit(f"check_cuda_syntax: clang found problems in {name}")
    if out:
        print(out[-2000:])
    print(f"ok: {name} parses for host and sm_90a device (clang -fsyntax-only, stub headers)")


def generated_sources(path):
    """Modules that build their source from instances (p9.py): import and render a sample."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_under_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    insts = getattr(mod, "P9_CHECK_INSTANCES", ())
    if insts:
        yield "p9_source(P9_CHECK_INSTANCES)", mod.p9_source(insts)
    insts = getattr(mod, "F12V_CHECK_INSTANCES", ())
    if insts:
        yield "f12v_source(F12V_CHECK_INSTANCES)", mod.f12v_source(insts)
    insts = getattr(mod, "F12M_CHECK_INSTANCES", ())
    if insts:
        yield "f12m_source(F12M_CHECK_INSTANCES)", mod.f12m_source(insts)
    insts = getattr(mod, "F12K_CHECK_INSTANCES", ())
    if insts:
        yield "f12k_source(F12K_CHECK_INSTANCES)", mod.f12k_source(insts)
    insts = getattr(mod, "SX_CHECK_INSTANCES", ())
    if insts:
        yield "sx_source(SX_CHECK_INSTANCES, SXW_CHECK_INSTANCES)", mod.sx_source(insts, getattr(mod, "SXW_CHECK_INSTANCES", ()))
    insts = getattr(mod, "RG_CHECK_INSTANCES", ())
    if insts:
        yield "rg_source(RG_CHECK_INSTANCES)", mod.rg_source(insts)
    insts = getattr(mod, "WG_CHECK_INSTANCES", ())
    if insts:
        yield "wg_source(WG_CHECK_INSTANCES, WGF_CHECK_INSTANCES)", mod.wg_source(insts, getattr(mod, "WGF_CHECK_INSTANCES", ()))


if __name__ == "__main__":
    n = 0
    for path in sys.argv[1:]:
        found = list(sources(path))
        if not found or any("@@INSTANCES@@" in src for _, src in found):
            found = list(generated_sources(path))
        for name, src in found:
            check(f"{Path(path).name}:{name}", src)
            n += 1
    if not n:
        raise SystemExit("check_cuda_syntax: no *_SRC sources found")

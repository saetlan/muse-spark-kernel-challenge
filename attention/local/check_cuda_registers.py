"""Flags register arrays indexed with a lane-dependent value in our CUDA sources (AF20: such an
index forces the whole array into local memory; A5b's accumulator ran from local memory in L5).
Register arrays = local array declarations of float/uint32_t/uint4/uint2/int/long inside device
code (not __shared__); lane-dependent names = lane, g, t4, hl, warp, wib, tid. Any use
`name[...]` whose index mentions one of them fails.
Run: .venv/bin/python attention/local/check_cuda_registers.py [file.py:VAR ...]"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["cuda_dec2.py:CUDA_DEC2_SRC", "cuda_dec3.py:CUDA_DEC3_SRC", "cuda_fused.py:CUDA_FUSED_SRC", "cuda_ver.py:CUDA_VER_SRC",
           "cuda_dec.py:CUDA_DEC_SRC", "cuda_unified.py:CUDA_UD_SRC", "cuda_ver2.py:CUDA_VER2_SRC", "cuda_dec4.py:CUDA_DEC4_SRC", "cuda_unified2.py:CUDA_UD2_SRC", "cuda_dec3p.py:CUDA_DEC3P_SRC", "cuda_dec4p.py:CUDA_DEC4P_SRC", "cuda_ver3.py:CUDA_VER3_SRC", "cuda_gather.py:CUDA_GATHER_SRC", "cuda_gather.py:CUDA_BULK_SRC"]
LANE = re.compile(r"\b(lane|g|t4|hl|warp|wib|tid)\b")
DECL = re.compile(r"^\s*(?:const\s+)?(?:float|uint32_t|uint4|uint2|int|long|Chunk)\s+(\w+)\s*\[", re.M)


def source(spec):
    path, var = spec.split(":")
    full = path if os.path.isabs(path) else os.path.join(HERE, path)
    s = importlib.util.spec_from_file_location("src_" + var, full)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return getattr(m, var)


def index_exprs(src, name):
    """Yield (line number, index text) for every name[...] use, matching nested brackets."""
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\[", src):
        i, depth, start = m.end(), 1, m.end()
        while i < len(src) and depth:
            depth += {"[": 1, "]": -1}.get(src[i], 0)
            i += 1
        yield src.count("\n", 0, m.start()) + 1, src[start:i - 1]
        # chained dimensions: name[a][b] -> also check [b]
        while i < len(src) and src[i] == "[":
            j, depth, s2 = i + 1, 1, i + 1
            while j < len(src) and depth:
                depth += {"[": 1, "]": -1}.get(src[j], 0)
                j += 1
            yield src.count("\n", 0, m.start()) + 1, src[s2:j - 1]
            i = j


bad = 0
for spec in sys.argv[1:] or DEFAULT:
    src = source(spec)
    shared = set(re.findall(r"__shared__[^;]*?\b(\w+)\s*\[", src))
    names = sorted(set(DECL.findall(src)) - shared)
    for name in names:
        for line, idx in index_exprs(src, name):
            if LANE.search(idx):
                print(f"{spec}:{line}: register array `{name}` indexed with a lane-dependent value: [{idx}]")
                bad += 1
print("check_cuda_registers:", "ok" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)

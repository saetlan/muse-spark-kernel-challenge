"""Builds attention/submission_v4.py: submission.py (v2.2) byte for byte, then the CUDA sources
(cuda_dec2.py, cuda_dec3.py, cuda_dec4.py, cuda_unified.py, cuda_ver2.py) and the v4 dispatch block (v4_block.py).
Usage: .venv/bin/python attention/local/make_v4.py [output name, default submission_v4.py]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_dec3.py", "cuda_dec4.py", "cuda_unified.py", "cuda_ver2.py", "v4_block.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())
out = os.path.join(PROB, sys.argv[1] if len(sys.argv) > 1 else "submission_v4.py")
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes)")

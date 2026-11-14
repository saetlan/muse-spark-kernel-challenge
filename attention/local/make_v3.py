"""Builds attention/submission_v3.py: submission.py (v2.2) byte for byte, then the CUDA sources
(cuda_dec2.py, cuda_fused.py, cuda_ver.py) and the v3 dispatch block (v3_block.py).
Usage: .venv/bin/python attention/local/make_v3.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_fused.py", "cuda_ver.py", "v3_block.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())
out = os.path.join(PROB, "submission_v3.py")
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes)")

"""Builds attention/submission_v5.py: v4 (submission.py + CUDA sources + v4_block.py, as make_v4.py)
plus the PDL Triton chain (pdl_block.py, make_pdl.py) and the v5 dispatch block (v5_block.py).
Usage: .venv/bin/python attention/local/make_v5.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_dec3.py", "cuda_dec4.py", "cuda_unified.py", "cuda_ver2.py", "v4_block.py",
             "pdl_block.py", "v5_block.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())
out = os.path.join(PROB, "submission_v5.py")
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes)")

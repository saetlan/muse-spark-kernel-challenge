"""Build proj/submission_v2.py: v1's choices plus PE2's five _fma_kernel winners (job 1d761e3043c74f50).

Run: .venv/bin/python proj/local/make_v2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_v1  # noqa: E402


def fma(bn, split, warps):
    return {"kind": "fma", "BN": bn, "split": split, "warps": warps}


CHOICE = dict(make_v1.CHOICE)
CHOICE.update({
    ("o", 1): fma(4, 1, 8),
    ("pooler", 1): fma(2, 1, 8),
    ("qkv_gate", 1): fma(2, 1, 8),
    ("router", 1): fma(2, 1, 8),
    ("router", 8): fma(16, 64, 8),
})

if __name__ == "__main__":
    make_v1.build(CHOICE, "submission_v2.py", "the PE1 and PE2 labs (jobs eb62bbc5e6d147fa, 1d761e3043c74f50)")

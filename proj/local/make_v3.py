"""Build proj/submission_v3.py: v2's choices plus PE3's four library winners (job 9b4420a835d84b81).

Run: .venv/bin/python proj/local/make_v3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_v1  # noqa: E402
import make_v2  # noqa: E402

CHOICE = dict(make_v2.CHOICE)
CHOICE.update({
    ("qkv_gate", 8): {"kind": "fi_mm", "backend": "auto"},
    ("unpooler", 8): {"kind": "fi_mm", "backend": "tinygemm"},
    ("pooler", 128): {"kind": "dg", "native": True},
    ("qkv_gate", 128): {"kind": "fi_mm", "backend": "cublaslt"},
})

if __name__ == "__main__":
    make_v1.build(CHOICE, "submission_v3.py",
                  "the PE1, PE2 and PE3 labs (jobs eb62bbc5e6d147fa, 1d761e3043c74f50, 9b4420a835d84b81)")

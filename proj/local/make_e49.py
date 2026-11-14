"""Build proj/lab_e49.py (PE49: padded F12M config re-sweep at m = 1 / 8 / 16): proj/submission_v13.py byte for
byte + lab/common.py + lab/sweep.py + lab/_e47_targets.py + lab/e47.py.

Run: .venv/bin/python proj/local/make_e47.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v13.py"


def build(out=PROJ / "lab_e49.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("ace3d9b0"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "_e49_targets.py", PROJ / "lab" / "e49.py"],
                   out, base=BASE)


if __name__ == "__main__":
    build()

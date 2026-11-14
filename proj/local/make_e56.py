"""Build proj/lab_e56.py (PE56: unpooler m32 on padded F12M): proj/submission_v15.py byte for
byte + lab/common.py + lab/sweep.py + lab/_e47_targets.py + lab/e47.py.

Run: .venv/bin/python proj/local/make_e47.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v15.py"


def build(out=PROJ / "lab_e56.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("20c1d624"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "_e56_targets.py", PROJ / "lab" / "e56.py"],
                   out, base=BASE)


if __name__ == "__main__":
    build()

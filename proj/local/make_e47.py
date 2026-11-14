"""Build proj/lab_e47.py (PE47: padded F12M on o m1 / m16 / m32, qkv / pooler / unpooler m32): proj/submission_v12.py byte for
byte + lab/common.py + lab/sweep.py + lab/_e47_targets.py + lab/e47.py.

Run: .venv/bin/python proj/local/make_e47.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v12.py"


def build(out=PROJ / "lab_e47.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("5ec03ce1"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "_e47_targets.py", PROJ / "lab" / "e47.py"],
                   out, base=BASE)


if __name__ == "__main__":
    build()

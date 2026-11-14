"""Build proj/lab_e46.py (PE46: F12M helper warp for the escape corrections): proj/submission_v12.py byte for byte + lab/common.py
+ lab/sweep.py + lab/f12mh.py + lab/e46.py.

Run: .venv/bin/python proj/local/make_e46.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v12.py"


def build(out=PROJ / "lab_e46.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("5ec03ce1"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "f12mh.py", PROJ / "lab" / "e46.py"],
                   out, base=BASE)


if __name__ == "__main__":
    build()

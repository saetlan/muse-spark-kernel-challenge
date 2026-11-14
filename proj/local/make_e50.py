"""Build proj/lab_e50.py (PE50: F12M epilogue with the escape reads first): scored/submission_v11b.py byte for byte + lab/common.py +
lab/sweep.py + lab/e44.py.

Run: .venv/bin/python proj/local/make_e44.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v13.py"


def build(out=PROJ / "lab_e50.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("ace3d9b0"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "e50.py"], out, base=BASE)


if __name__ == "__main__":
    build()

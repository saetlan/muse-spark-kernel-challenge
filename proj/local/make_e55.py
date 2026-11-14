"""Build proj/lab_e55.py (PE55: capped helper on the unpooler slice): proj/submission_v14.py byte for byte + lab/common.py (f12mh_source is already in v13)
+ lab/sweep.py + lab/f12mh.py + lab/e46.py.

Run: .venv/bin/python proj/local/make_e46.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v14.py"


def build(out=PROJ / "lab_e55.py"):
    sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert sha.startswith("b8a10be7"), sha[:16]
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "e55.py"],
                   out, base=BASE)


if __name__ == "__main__":
    build()

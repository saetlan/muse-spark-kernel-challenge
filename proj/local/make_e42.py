"""Build proj/lab_e42.py (PE42, P43 F12K = stream-K F12M): scored/submission_v11b.py (the board entry) byte for byte as the
timed path + lab/common.py + proj/f12m/f12k.py (code only) + lab/sweep.py + lab/f12k_lab.py + lab/e42.py.

Run: .venv/bin/python proj/local/make_e42.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "scored/submission_v11b.py"


def build():
    base_sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert base_sha.startswith("cfe20913"), f"{BASE} is not v11b: {base_sha[:16]}"
    sec = body(PROJ / "f12m" / "f12k.py")
    assert "F12K_INSTANCES = ()" in sec and "def f12k_build(" in sec and "def f12k_pieces(" in sec
    sec_path = PROJ / "lab" / "_f12k_section.py"
    sec_path.write_text(sec)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "f12k_lab.py",
                    PROJ / "lab" / "e42.py"], PROJ / "lab_e42.py", base=BASE)
    print("base", BASE, "sha256", base_sha[:16], "| f12k.py sha256", hashlib.sha256((PROJ / "f12m" / "f12k.py").read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    build()

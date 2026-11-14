"""Build proj/lab_e30.py (PE30): v10 + lab/common.py + lab/sweep.py + the load_inline import + lab/f12mlv.py + lab/e30.py.

Run: .venv/bin/python proj/local/make_e30.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def build():
    sec = PROJ / "lab" / "_f12mlv_section.py"
    sec.write_text("from torch.utils.cpp_extension import load_inline\n\n" + (PROJ / "lab" / "f12mlv.py").read_text())
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", sec, PROJ / "lab" / "e30.py"],
                   PROJ / "lab_e30.py", base="scored/submission_v10.py")


if __name__ == "__main__":
    build()

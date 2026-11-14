"""Build proj/lab_e27.py (PE27): v10 + DG12 section (with DIAG) + DG12_DIAG + lab/common.py + lab/sweep.py + lab/e27.py.

Run: .venv/bin/python proj/local/make_e27.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dg"))
import make_e24  # noqa: E402
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def build():
    diag = {("unpooler", 128): [dict(make_e24.cfg(128, 64), DIAG=d) for d in (0, 1, 2, 3, 4, 7)],
            ("pooler", 128): [dict(make_e24.cfg(64, 64), DIAG=d) for d in (0, 1, 3, 4)]}
    section = ("import os\n\n" + body(PROJ / "dg" / "dg12_layout.py") + "\n\n" + body(PROJ / "dg" / "dg12.py")
               + f"\n\nDG12_DIAG = {diag!r}\n")
    sec_path = PROJ / "lab" / "_dg12_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e27.py"],
                   PROJ / "lab_e27.py", base="submission_v10.py")
    for key, cs in diag.items():
        print(key, [f"{c['BM']}x{c['BN']}s{c['ST']}d{c['DIAG']}" for c in cs])


if __name__ == "__main__":
    build()

"""Build proj/lab_e22.py (PE22): v10 (proj/submission_v10.py) + the DGX section (proj/dg/dgx.py's code, `import os`, and
DGX_TARGETS = DeepGEMM's heuristic config per target shape from proj/dg/dg_configs.py) + lab/common.py + lab/sweep.py +
lab/e22.py.

Run: .venv/bin/python proj/local/make_e22.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dg"))
import dg_configs  # noqa: E402
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
TARGETS = [("pooler", 128), ("qkv_gate", 128), ("unpooler", 128), ("o", 128), ("pooler", 256), ("pooler", 32)]


def build():
    targets = {}
    for layer, m in TARGETS:
        n, k = dg_configs.SHAPES[layer]
        c = dg_configs.dg_best(m, n, k)
        targets[(layer, m)] = {key: c[key] for key in ("BM", "BN", "ST", "SWD", "NMATH", "MC", "MN")}
    section = ("import os\n\n" + body(PROJ / "dg" / "dgx.py") + f"\n\nDGX_TARGETS = {targets!r}\n")
    sec_path = PROJ / "lab" / "_dgx_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e22.py"],
                   PROJ / "lab_e22.py", base="submission_v10.py")
    for key, c in targets.items():
        print(key, c)


if __name__ == "__main__":
    build()

"""Build proj/lab_e21.py (PE21): v10's choices rebuilt with the TG-capable F12M (make_v9x: v7a's choices + PE19's x-ring
winner via make_v7, then the HL section with PE18-PE20's router winners -> proj/submission_v10b.py, v10's kernels),
F12M_INSTANCES extended by f12m_configs.TG_LAB and F12M_TG_CANDIDATES set (proj/submission_v10b_tglab.py), then
lab/common.py + lab/sweep.py + lab/e21.py.

Run: .venv/bin/python proj/local/make_e21.py
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f12m"))
import f12m_configs  # noqa: E402
import make_lab  # noqa: E402
import make_v9x  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def build():
    make_v9x.build(make_v9x.XD_WON, make_v9x.HL_WON, out="submission_v10b.py")
    src = (PROJ / "submission_v10b.py").read_text()
    found = re.findall(r"\nF12M_INSTANCES = (\(.*?\))\n", src)
    assert len(found) == 1, "expected one F12M_INSTANCES line"
    cur = ast.literal_eval(found[0])
    new = tuple(dict.fromkeys(tuple(cur) + f12m_configs.TG_LAB))
    src = src.replace(f"\nF12M_INSTANCES = {found[0]}\n", f"\nF12M_INSTANCES = {new!r}\n", 1)
    cands = {key: [dict(c) for c in v] for key, v in f12m_configs.TG_CANDIDATES.items()}
    src = src.rstrip() + f"\n\n# PE21 token-group candidates (lab only)\nF12M_TG_CANDIDATES = {cands!r}\n"
    (PROJ / "submission_v10b_tglab.py").write_text(src)
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "e21.py"],
                   PROJ / "lab_e21.py", base="submission_v10b_tglab.py")
    print(f"F12M instances: {len(cur)} chosen + {len(new) - len(cur)} token-group")


if __name__ == "__main__":
    build()

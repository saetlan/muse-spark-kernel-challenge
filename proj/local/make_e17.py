"""Build proj/lab_e17.py (PE17): v7a's choices rebuilt with the CS-capable F12M section
(proj/submission_v7b.py: the same kernels, CS = 1), F12M_INSTANCES extended by f12m_configs.CS_LAB and
F12M_CS_CANDIDATES set (proj/submission_v7b_cslab.py), then lab/common.py + lab/sweep.py + lab/e17.py.

Run: .venv/bin/python proj/local/make_e17.py
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f12m"))
import chosen  # noqa: E402
import f12m_configs  # noqa: E402
import make_lab  # noqa: E402
import make_v7  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def build():
    make_v7.build({**chosen.V6, **chosen.PE10}, "submission_v7b.py")
    src = (PROJ / "submission_v7b.py").read_text()
    found = re.findall(r"\nF12M_INSTANCES = (\(.*?\))\n", src)
    assert len(found) == 1, "expected one F12M_INSTANCES line"
    cur = ast.literal_eval(found[0])
    new = tuple(dict.fromkeys(tuple(cur) + f12m_configs.CS_LAB))
    src = src.replace(f"\nF12M_INSTANCES = {found[0]}\n", f"\nF12M_INSTANCES = {new!r}\n", 1)
    cands = {key: [dict(c) for c in v] for key, v in f12m_configs.CS_CANDIDATES.items()}
    src = src.rstrip() + f"\n\n# PE17 split-K candidates (lab only)\nF12M_CS_CANDIDATES = {cands!r}\n"
    (PROJ / "submission_v7b_cslab.py").write_text(src)
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "e17.py"],
                   PROJ / "lab_e17.py", base="submission_v7b_cslab.py")
    print(f"F12M instances: {len(cur)} chosen + {len(new) - len(cur)} split-K")


if __name__ == "__main__":
    build()

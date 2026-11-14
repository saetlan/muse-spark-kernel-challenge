"""Build proj/lab_e19.py (PE19): v7a's choices rebuilt with the CS/XD-capable F12M (proj/submission_v7b.py), the HL
section with PE18's router winners (proj/submission_v8b.py: v8's kernels), F12M_INSTANCES extended by
f12m_configs.XD_LAB and F12M_XD_CANDIDATES set (proj/submission_v8b_xdlab.py), then lab/common.py +
lab/sweep.py + lab/e19.py.

Run: .venv/bin/python proj/local/make_e19.py
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f12m"))
import chosen  # noqa: E402
import f12m_configs  # noqa: E402
import make_hl  # noqa: E402
import make_lab  # noqa: E402
import make_v7  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def build():
    make_v7.build({**chosen.V6, **chosen.PE10}, "submission_v7b.py")
    make_hl.build(str(PROJ / "submission_v8b.py"), "submission_v7b.py", make_hl.HL_CHOSEN)
    src = (PROJ / "submission_v8b.py").read_text()
    found = re.findall(r"\nF12M_INSTANCES = (\(.*?\))\n", src)
    assert len(found) == 1, "expected one F12M_INSTANCES line"
    cur = ast.literal_eval(found[0])
    new = tuple(dict.fromkeys(tuple(cur) + f12m_configs.XD_LAB))
    src = src.replace(f"\nF12M_INSTANCES = {found[0]}\n", f"\nF12M_INSTANCES = {new!r}\n", 1)
    cands = {key: [dict(c) for c in v] for key, v in f12m_configs.XD_CANDIDATES.items()}
    src = src.rstrip() + f"\n\n# PE19 x-ring candidates (lab only)\nF12M_XD_CANDIDATES = {cands!r}\n"
    (PROJ / "submission_v8b_xdlab.py").write_text(src)
    make_lab.build([PROJ / "lab" / "common.py", PROJ / "lab" / "sweep.py", PROJ / "lab" / "e19.py"],
                   PROJ / "lab_e19.py", base="submission_v8b_xdlab.py")
    print(f"F12M instances: {len(cur)} chosen + {len(new) - len(cur)} x-ring")


if __name__ == "__main__":
    build()

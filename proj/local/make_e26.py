"""Build proj/lab_e26.py (PE26): v10 + DG12 section (batched decoder) + DG12_TARGETS + lab/common.py + lab/sweep.py +
lab/e26.py. Targets: PE24/PE25's (DeepGEMM's block shape + a smaller-BN alternative) plus BM 128 at m = 128/256 so
each weight tile is decoded once (PF43 b).

Run: .venv/bin/python proj/local/make_e26.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dg"))
import dg_configs  # noqa: E402
import make_e24  # noqa: E402
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def targets():
    t = make_e24.targets()
    extra = {("pooler", 128): [(128, 64)], ("unpooler", 128): [(128, 64), (128, 128)], ("qkv_gate", 128): [(128, 48)],
             ("o", 128): [(128, 64)], ("pooler", 256): [(128, 64), (128, 128)], ("qkv_gate", 256): [(128, 48)]}
    for key, shapes in extra.items():
        for bm, bn in shapes:
            c = make_e24.cfg(bm, bn)
            if c not in t[key] and dg_configs.SHAPES[key[0]][0] % bn == 0:
                t[key].append(c)
    return t


def build():
    t = targets()
    section = ("import os\n\n" + body(PROJ / "dg" / "dg12_layout.py") + "\n\n" + body(PROJ / "dg" / "dg12.py")
               + f"\n\nDG12_TARGETS = {t!r}\n")
    sec_path = PROJ / "lab" / "_dg12_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e26.py"],
                   PROJ / "lab_e26.py", base="submission_v10.py")
    n_inst = len({(c["BM"], c["BN"], c["ST"], dg_configs.SHAPES[l][0], dg_configs.SHAPES[l][1]) for (l, m), cs in t.items() for c in cs})
    for key, cs in t.items():
        print(key, [f"{c['BM']}x{c['BN']}s{c['ST']}" for c in cs])
    print("instances:", n_inst)


if __name__ == "__main__":
    build()

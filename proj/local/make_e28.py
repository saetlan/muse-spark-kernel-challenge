"""Build proj/lab_e28.py (PE28): v10 + DG12 section (prefetched escape epilogue, batched decoder, DIAG) + DG12_TARGETS +
DG12_DIAG + lab/common.py + lab/sweep.py + lab/e28.py. Targets: PE26's best per shape plus BM 128 x BN 32 at pooler
m128/256; diagnosis: unpooler m128 128x64 at DIAG 0/1/8/9 and pooler m128 64x64 at DIAG 0/8/9.

Run: .venv/bin/python proj/local/make_e28.py
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
C = make_e24.cfg


def targets():
    return {
        ("pooler", 128): [C(64, 64), C(128, 64), C(128, 32)],
        ("unpooler", 128): [C(128, 64), C(128, 128)],
        ("qkv_gate", 128): [C(64, 48), C(128, 48)],
        ("o", 128): [C(128, 64), C(128, 128)],
        ("pooler", 256): [C(128, 64), C(128, 32), C(128, 128)],
        ("unpooler", 256): [C(128, 128), C(128, 64)],
        ("qkv_gate", 256): [C(128, 48), C(64, 96)],
        ("o", 256): [C(128, 128), C(128, 64)],
        ("pooler", 32): [C(32, 32)],
        ("unpooler", 32): [C(32, 64)],
        ("qkv_gate", 32): [C(32, 32)],
        ("o", 32): [C(32, 64)],
    }


def diag():
    return {("unpooler", 128): [dict(C(128, 64), DIAG=d) for d in (0, 1, 8, 9)],
            ("pooler", 128): [dict(C(64, 64), DIAG=d) for d in (0, 8, 9)]}


def build():
    t, dg = targets(), diag()
    for (layer, m), cs in t.items():
        assert all(dg_configs.SHAPES[layer][0] % c["BN"] == 0 for c in cs), (layer, m)
    section = ("import os\n\n" + body(PROJ / "dg" / "dg12_layout.py") + "\n\n" + body(PROJ / "dg" / "dg12.py")
               + f"\n\nDG12_TARGETS = {t!r}\nDG12_DIAG = {dg!r}\n")
    sec_path = PROJ / "lab" / "_dg12_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e28.py"],
                   PROJ / "lab_e28.py", base="submission_v10.py")
    insts = {(c["BM"], c["BN"], c["ST"], dg_configs.SHAPES[l][0], c.get("DIAG", 0)) for d in (t, dg) for (l, m), cs in d.items() for c in cs}
    print("instances:", len(insts))


if __name__ == "__main__":
    build()

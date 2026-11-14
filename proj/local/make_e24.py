"""Build proj/lab_e24.py (PE24): v10 + DG12 section (proj/dg/dg12_layout.py and proj/dg/dg12.py, `import os`,
DG12_TARGETS) + lab/common.py + lab/sweep.py + lab/e24.py. Targets: DeepGEMM's block shape per (layer, m) for
m = 32/128/256 (BN changed to divide N where DeepGEMM's doesn't), stages recomputed for DG12's raw planes, plus a
smaller-BN alternative at m = 128/256.

Run: .venv/bin/python proj/local/make_e24.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dg"))
import dg12  # noqa: E402
import dg_configs  # noqa: E402
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def cfg(bm, bn):
    return dict(BM=bm, BN=bn, ST=dg12.dg12_stages(bm, bn), SWD=dg_configs.storage(bm, bn, 64)["SWD"],
                NMATH=128 if bm <= 64 else 256)


def targets():
    out = {}
    for m in (128, 256, 32):
        for layer in ("pooler", "unpooler", "qkv_gate", "o"):
            n, k = dg_configs.SHAPES[layer]
            c = dg_configs.dg_best(m, n, k)
            bm, bn = c["BM"], c["BN"]
            if n % bn:
                bn = max(b for b in range(16, bn + 1, 16) if n % b == 0)
            cands = [cfg(bm, bn)]
            if m >= 128 and bn >= 64 and n % (bn // 2) == 0:
                cands.append(cfg(bm, bn // 2))
            if layer == "qkv_gate" and m == 256:
                cands.append(cfg(64, 96))
            out[(layer, m)] = cands
    return out


def build():
    t = targets()
    section = ("import os\n\n" + body(PROJ / "dg" / "dg12_layout.py") + "\n\n" + body(PROJ / "dg" / "dg12.py")
               + f"\n\nDG12_TARGETS = {t!r}\n")
    sec_path = PROJ / "lab" / "_dg12_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e24.py"],
                   PROJ / "lab_e24.py", base="submission_v10.py")
    n_inst = len({(c["BM"], c["BN"], c["ST"], dg_configs.SHAPES[l][0], dg_configs.SHAPES[l][1]) for (l, m), cs in t.items() for c in cs})
    for key, cs in t.items():
        print(key, [f"{c['BM']}x{c['BN']}s{c['ST']}" for c in cs])
    print("instances:", n_inst)


if __name__ == "__main__":
    build()

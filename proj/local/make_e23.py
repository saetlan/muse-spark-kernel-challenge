"""Build proj/lab_e23.py (PE23 recon): v10 + DGX section (proj/dg/dgx.py with launch variants) + DGX_TARGETS
{(layer, m): [config, ...]} + lab/common.py + lab/sweep.py + lab/e23.py. Pooler m128: DeepGEMM's heuristic config at
LV 0 / 1 / 2 plus three other block shapes at LV 0; qkv/unpooler/o m128 and pooler m256: the heuristic at LV 0.

Run: .venv/bin/python proj/local/make_e23.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dg"))
import dg_configs  # noqa: E402
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def cfg(bm, bn, lv):
    st = dg_configs.storage(bm, bn, 64)
    pl = dg_configs.pipeline(bm, bn, 64)
    return dict(BM=bm, BN=bn, ST=pl["ST"], SWD=st["SWD"], NMATH=128 if bm <= 64 else 256, MC=1, MN=1, LV=lv)


def heuristic(layer, m, lv):
    n, k = dg_configs.SHAPES[layer]
    c = dg_configs.dg_best(m, n, k)
    return cfg(c["BM"], c["BN"], lv)


def build():
    targets = {
        ("pooler", 128): [heuristic("pooler", 128, 0), heuristic("pooler", 128, 1), heuristic("pooler", 128, 2),
                          cfg(128, 32, 0), cfg(64, 32, 0), cfg(64, 128, 0)],
        ("qkv_gate", 128): [heuristic("qkv_gate", 128, 0)],
        ("unpooler", 128): [heuristic("unpooler", 128, 0)],
        ("o", 128): [heuristic("o", 128, 0)],
        ("pooler", 256): [heuristic("pooler", 256, 0)],
    }
    section = ("import os\n\n" + body(PROJ / "dg" / "dgx.py") + f"\n\nDGX_TARGETS = {targets!r}\n")
    sec_path = PROJ / "lab" / "_dgx_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e23.py"],
                   PROJ / "lab_e23.py", base="submission_v10.py")
    for key, cs in targets.items():
        print(key, [f"{c['BM']}x{c['BN']}s{c['ST']}l{c['LV']}" for c in cs])


if __name__ == "__main__":
    build()

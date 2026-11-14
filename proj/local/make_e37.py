"""Build proj/lab_e37.py (PE37): v10 + lab/common.py + the DGRS v8 section (the pinned copy proj/dg/pinned/dgrs_v8.py, sha256
c80ffca8..., code only, with its load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs8b.py + lab/e37.py. DGRS configs:
unpooler m32 and o m32, BM32 MATH 4: S 1, S 2 red 0, S 2 red 2 (the d0 entries of the DIAG runs).

Run: .venv/bin/python proj/local/make_e37.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ / "dg" / "pinned" / "dgrs_v8.py"
SHAPE = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def c(bm, s, red=0, math=2, diag=0):
    return dict(BM=bm, S=s, CPS=1, RED=red if s > 1 else 0, MATH=math, DIAG=diag)


def targets():
    t = {
        ("unpooler", 32): [c(32, 1, 0, 4), c(32, 2, 0, 4), c(32, 2, 2, 4)],
        ("o", 32): [c(32, 1, 0, 4), c(32, 2, 0, 4), c(32, 2, 2, 4)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * -(-m // cc["BM"]) * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 and cc["S"] <= 8 and (cc["S"] - 1) * per < kc, (layer, m, cc, ctas)
            assert cc["MATH"] == 2 or cc["BM"] <= 64, (layer, m, cc)
            assert cc["DIAG"] in (0, 256) and (cc["DIAG"] == 0 or (cc["RED"] == 2 and cc["S"] > 1)), (layer, m, cc)
        assert len({tuple(sorted(cc.items())) for cc in cs}) == len(cs), (layer, m)
    assert all(cc in t[(layer, 32)] for layer in ("unpooler", "o") for cc in (c(32, 1, 0, 4), c(32, 2, 0, 4), c(32, 2, 2, 4)))
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    dg = DGRS.read_text()
    assert sha.startswith("c80ffca870818bc9"), f"pinned v8 changed: {sha[:16]}"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs8b.py",
                    PROJ / "lab" / "e37.py"], PROJ / "lab_e37.py", base="scored/submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

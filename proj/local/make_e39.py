"""Build proj/lab_e39.py (PE39): v10 + lab/common.py + the DGRS v9 section (the pinned copy proj/dg/pinned/dgrs_v9.py, sha256
c8b76b59..., code only, with its load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs9.py + lab/e39.py. DGRS configs:
specialized d0 instances (RED / SPL fixed at compile time) on every m32 / m128 / m256 shape: BM32 MATH 4 at m32, BM64 MATH 4 or
BM128 MATH 2 at m128 / m256; red 2 only where it fits (not BM128 S2), CTAs <= 132.

Run: .venv/bin/python proj/local/make_e39.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ / "dg" / "pinned" / "dgrs_v9.py"
SHAPE = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def c(s, red, bm=32, math=4):
    return dict(BM=bm, S=s, CPS=1, RED=red if s > 1 else 0, MATH=math, DIAG=0, SPEC=True)


def targets():
    t = {
        ("unpooler", 32): [c(2, 2), c(2, 0)],
        ("pooler", 32): [c(3, 0), c(3, 2), c(2, 2), c(4, 1)],
        ("qkv_gate", 32): [c(4, 0), c(4, 2), c(7, 1)],
        ("o", 32): [c(2, 2), c(2, 0)],
        ("unpooler", 128): [c(1, 0, 64), c(2, 0, 128, 2)],
        ("pooler", 128): [c(2, 0, 64), c(2, 2, 64)],
        ("qkv_gate", 128): [c(2, 0, 64), c(4, 0, 128, 2)],
        ("o", 128): [c(1, 0, 64), c(2, 0, 128, 2)],
        ("unpooler", 256): [c(1, 0, 128, 2)],
        ("pooler", 256): [c(2, 0, 128, 2), c(1, 0, 64)],
        ("qkv_gate", 256): [c(2, 0, 128, 2), c(1, 0, 64)],
        ("o", 256): [c(1, 0, 128, 2)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * -(-m // cc["BM"]) * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 and cc["S"] <= 8 and (cc["S"] - 1) * per < kc, (layer, m, cc, ctas)
            assert cc["MATH"] == 2 or cc["BM"] <= 64, (layer, m, cc)
            assert not (cc["BM"] == 128 and cc["RED"] == 2), (layer, m, cc)   # red 2 BM128 S2 does not fit (v6)
        assert len({tuple(sorted(cc.items())) for cc in cs}) == len(cs), (layer, m)
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    assert sha.startswith("c8b76b59502806b8"), f"pinned v9 changed: {sha[:16]}"
    dg = DGRS.read_text()
    assert ('functions=["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"]' in dg
            and 'name="dgrs_ext"' in dg), "v9's exported functions changed: update DGRS_FUNCS in lab/dgrs9.py"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# proj/dg/pinned/dgrs_v9.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs9.py",
                    PROJ / "lab" / "e39.py"], PROJ / "lab_e39.py", base="scored/submission_v10.py")
    print("dgrs v9 sha256", sha[:16])


if __name__ == "__main__":
    build()

"""Build proj/lab_e32.py (PE32): v10 + lab/common.py + the DGRS v4 section (shared/gemm/dg12rs/dgrs.py, code only, with its
load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/hlp.py + lab/dgrs4.py + lab/e32.py. DGRS configs (BM, S, CPS, RED,
MATH): PE31's best per shape with math 2, plus math 4 where it fits (BM <= 64, CPS 1, CTAs <= 132).

Run: .venv/bin/python proj/local/make_e32.py   (records the sha256 of the dgrs.py it copied)
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ.parent / "shared" / "gemm" / "dg12rs" / "dgrs.py"
SHAPE = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def c(bm, s, red=0, math=2):
    return dict(BM=bm, S=s, CPS=1, RED=red if s > 1 else 0, MATH=math)


def targets():
    t = {
        ("qkv_gate", 32): [c(32, 4, 0, 2), c(32, 4, 0, 4), c(32, 7, 1, 4)],
        ("o", 32): [c(32, 2, 0, 2), c(32, 2, 0, 4), c(32, 1, 0, 4)],
        ("pooler", 32): [c(32, 4, 1, 2), c(32, 4, 1, 4), c(32, 4, 0, 4), c(32, 2, 0, 4)],
        ("unpooler", 32): [c(32, 2, 0, 2), c(32, 2, 0, 4), c(32, 1, 0, 4)],
        ("qkv_gate", 128): [c(128, 4, 0, 2), c(64, 3, 1, 4), c(64, 2, 0, 4)],
        ("o", 128): [c(128, 2, 0, 2), c(64, 1, 0, 4)],
        ("pooler", 128): [c(64, 2, 0, 2), c(64, 2, 0, 4)],
        ("unpooler", 128): [c(128, 2, 0, 2), c(64, 1, 0, 4)],
        ("qkv_gate", 256): [c(128, 2, 0, 2), c(64, 1, 0, 4)],
        ("o", 256): [c(128, 1, 0, 2)],
        ("pooler", 256): [c(128, 2, 0, 2), c(64, 1, 0, 4)],
        ("unpooler", 256): [c(128, 1, 0, 2)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * -(-m // cc["BM"]) * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 and cc["S"] <= 8 and (cc["S"] - 1) * per < kc, (layer, m, cc, ctas)
            assert cc["MATH"] == 2 or cc["BM"] <= 64, (layer, m, cc)
        assert len({tuple(sorted(cc.items())) for cc in cs}) == len(cs), (layer, m)
    assert c(32, 2, 0, 2) in t[("unpooler", 32)] and c(32, 2, 0, 4) in t[("unpooler", 32)]   # DIAG d0 entries
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    dg = DGRS.read_text()
    assert "dgrs_max_clusters" in dg and "int64_t red, int64_t math" in dg, "dgrs.py is not v4"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "hlp.py",
                    PROJ / "lab" / "dgrs4.py", PROJ / "lab" / "e32.py"], PROJ / "lab_e32.py", base="scored/submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

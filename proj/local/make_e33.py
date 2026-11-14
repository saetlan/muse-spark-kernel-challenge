"""Build proj/lab_e33.py (PE33): v10 + lab/common.py + the DGRS v5 section (shared/gemm/dg12rs/dgrs.py, code only, with its
load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs5.py + lab/e33.py. DGRS configs (BM, S, CPS, RED, MATH): PE32's
sweep with math 4 back in (BM <= 64, CTAs <= 132), red 0 with S <= 2 except pooler m32 S3 and qkv S4 (clusters fit one
wave per PE32's occupancy: S4 30, S2 66).

Run: .venv/bin/python proj/local/make_e33.py   (records the sha256 of the dgrs.py it copied)
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
        ("qkv_gate", 32): [c(32, 4, 0, 2), c(32, 4, 0, 4), c(32, 7, 1, 4), c(32, 7, 1, 2)],
        ("o", 32): [c(32, 2, 0, 2), c(32, 2, 0, 4), c(32, 1, 0, 4)],
        ("pooler", 32): [c(32, 2, 0, 4), c(32, 3, 0, 4), c(32, 4, 1, 4), c(32, 4, 1, 2)],
        ("unpooler", 32): [c(32, 2, 0, 2), c(32, 2, 0, 4), c(32, 1, 0, 4)],
        ("qkv_gate", 128): [c(128, 4, 0, 2), c(64, 2, 0, 4), c(64, 3, 1, 4)],
        ("o", 128): [c(128, 2, 0, 2), c(64, 1, 0, 4), c(64, 1, 0, 2)],
        ("pooler", 128): [c(64, 2, 0, 2), c(64, 2, 0, 4)],
        ("unpooler", 128): [c(128, 2, 0, 2), c(64, 1, 0, 4), c(64, 1, 0, 2)],
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
    assert all(c(32, 2, 0, mm) in t[(layer, 32)] for layer in ("unpooler", "o") for mm in (2, 4))   # DIAG d0 entries
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    dg = DGRS.read_text()
    assert "dgrs_stages" in dg and "pre_corr" in dg and "int64_t red, int64_t math" in dg, "dgrs.py is not v5"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs5.py",
                    PROJ / "lab" / "e33.py"], PROJ / "lab_e33.py", base="scored/submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

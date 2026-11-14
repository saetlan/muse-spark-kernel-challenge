"""Build proj/lab_e34.py (PE34): v10 + lab/common.py + the DGRS v6 section (shared/gemm/dg12rs/dgrs.py, code only, with its
load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs6.py + lab/e34.py. DGRS configs (BM, S, CPS, RED, MATH): the m32
shapes with red 2 (push) next to PE33's best red 0 / red 1 (MATH 4), qkv S7 red 2 (one wave only if 18 clusters of 7 fit:
logged), plus pooler m128 BM64 S2 red 0 / red 2 and unpooler m128 BM64 S1.

Run: .venv/bin/python proj/local/make_e34.py   (records the sha256 of the dgrs.py it copied)
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
        ("qkv_gate", 32): [c(32, 4, 0, 4), c(32, 4, 2, 4), c(32, 7, 2, 4), c(32, 7, 1, 4)],
        ("o", 32): [c(32, 2, 0, 4), c(32, 2, 2, 4)],
        ("pooler", 32): [c(32, 3, 0, 4), c(32, 3, 2, 4), c(32, 2, 2, 4), c(32, 4, 1, 4), c(32, 4, 2, 4)],
        ("unpooler", 32): [c(32, 2, 0, 4), c(32, 2, 2, 4)],
        ("pooler", 128): [c(64, 2, 0, 4), c(64, 2, 2, 4)],
        ("unpooler", 128): [c(64, 1, 0, 4)],
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
    assert all(c(32, 2, r, 4) in t[(layer, 32)] for layer in ("unpooler", "o") for r in (0, 2))   # DIAG d0 entries
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    dg = DGRS.read_text()
    assert "st.shared::cluster" in dg and "cl_arrive_relaxed" in dg and "red 0 (pull), 1 (global), 2 (push)" in dg, "dgrs.py is not v6"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs6.py",
                    PROJ / "lab" / "e34.py"], PROJ / "lab_e34.py", base="scored/submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

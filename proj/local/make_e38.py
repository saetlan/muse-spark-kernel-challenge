"""Build proj/lab_e38.py (PE38): v10 + lab/common.py + the DGRS v9 section (the pinned copy proj/dg/pinned/dgrs_v9.py, sha256
c8b76b59..., code only, with its load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs9.py + lab/e38.py. DGRS configs
(BM32 MATH 4; DIAG 0 / 512; SPEC = built in the specialized module with RED / SPL fixed): unpooler m32 S2 r2 (d0, d512, d512
spec), pooler m32 S3 r0 / r2 (d512, generic and spec), o m32 S2 r2 (d512, generic and spec), qkv m32 S4 r2 (d512, generic and
spec).

Run: .venv/bin/python proj/local/make_e38.py
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


def c(s, red, diag=512, spec=False):
    return dict(BM=32, S=s, CPS=1, RED=red if s > 1 else 0, MATH=4, DIAG=diag, SPEC=spec)


def targets():
    t = {
        ("unpooler", 32): [c(2, 2, 0), c(2, 2), c(2, 2, spec=True)],
        ("pooler", 32): [c(3, 0), c(3, 2), c(3, 0, spec=True), c(3, 2, spec=True)],
        ("o", 32): [c(2, 2), c(2, 2, spec=True)],
        ("qkv_gate", 32): [c(4, 2), c(4, 2, spec=True)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 and cc["S"] <= 8 and (cc["S"] - 1) * per < kc and per <= 256, (layer, m, cc, ctas)
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
                    PROJ / "lab" / "e38.py"], PROJ / "lab_e38.py", base="scored/submission_v10.py")
    print("dgrs v9 sha256", sha[:16])


if __name__ == "__main__":
    build()

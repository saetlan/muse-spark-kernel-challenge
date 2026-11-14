"""Build proj/lab_e41.py (PE41, ring depth): v10 + lab/common.py + the DGRS v10 section (the pinned copy proj/dg/pinned/dgrs_v10.py,
sha256 f4ff43fa..., code only, with its load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs10b.py + lab/e41.py. DGRS
configs (BM32 d0 m32, all fully specialized): unpooler S2 r2 at 8 / 10 stages, pooler S4 r1 at 8 / 10 and S3 r0 at 8 / 12, qkv S5
r2 at 8 / 10 and S4 r2 at 10, o S2 r2 MATH 4 at 8 and MATH 2 at 8.

Run: .venv/bin/python proj/local/make_e41.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ / "dg" / "pinned" / "dgrs_v10.py"
SHAPE = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def c(layer, s, red, mod, smx=8, math=4):
    n, k = SHAPE[layer]
    out = dict(BM=32, S=s, CPS=1, RED=red if s > 1 else 0, MATH=math, DIAG=0, MOD=mod)
    if mod == "full":
        out.update(M=32, N=n, K=k, SMX=smx)
    return out


def both(layer, s, red):
    return [c(layer, s, red, "spec"), c(layer, s, red, "full")]


def targets():
    t = {
        ("unpooler", 32): [c("unpooler", 2, 2, "full", 8), c("unpooler", 2, 2, "full", 10)],
        ("pooler", 32): [c("pooler", 4, 1, "full", 8), c("pooler", 4, 1, "full", 10), c("pooler", 3, 0, "full", 8),
                         c("pooler", 3, 0, "full", 12)],
        ("qkv_gate", 32): [c("qkv_gate", 5, 2, "full", 8), c("qkv_gate", 5, 2, "full", 10), c("qkv_gate", 4, 2, "full", 10)],
        ("o", 32): [c("o", 2, 2, "full", 8), c("o", 2, 2, "full", 8, 2)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * -(-m // cc["BM"]) * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 and cc["S"] <= 8 and (cc["S"] - 1) * per < kc, (layer, m, cc, ctas)
            assert cc.get("SMX", 8) in (8, 10, 12), (layer, m, cc)
            assert cc["MATH"] == 2 or cc["BM"] <= 64, (layer, m, cc)
            assert not (cc["BM"] == 128 and cc["RED"] == 2), (layer, m, cc)   # red 2 BM128 S2 does not fit (v6)
        assert len({tuple(sorted(cc.items())) for cc in cs}) == len(cs), (layer, m)
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    assert sha.startswith("f4ff43fafffd5af2"), f"pinned v10 changed: {sha[:16]}"
    dg = DGRS.read_text()
    assert ('functions=["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_attrs_x", "dgrs_max_clusters", "dgrs_stages", '
            '"dgrs_stages_d"]' in dg), "v10's exported functions changed: update DGRS_FUNCS in lab/dgrs10.py"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# proj/dg/pinned/dgrs_v10.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs10b.py",
                    PROJ / "lab" / "e41.py"], PROJ / "lab_e41.py", base="scored/submission_v10.py")
    print("dgrs v10 sha256", sha[:16])


if __name__ == "__main__":
    build()

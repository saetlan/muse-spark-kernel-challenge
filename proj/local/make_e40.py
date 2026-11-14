"""Build proj/lab_e40.py (PE40): v10 + lab/common.py + the DGRS v10 section (the pinned copy proj/dg/pinned/dgrs_v10.py, sha256
f4ff43fa..., code only, with its load_inline import) + DGRS_TARGETS + lab/sweep.py + lab/dgrs10.py + lab/e40.py. DGRS configs
(BM32 MATH 4 d0, m32 only): each of PE39's best RED / SPL instances (MOD spec) next to its fully specialized twin (MOD full: SMX
8, M 32, N, K fixed): unpooler S2 r2, pooler S3 r2 / S4 r1 / S3 r0, qkv S4 r2 / S4 r0 / S6 r2 / S5 r2 (108 / 90 CTAs), o S2 r2.

Run: .venv/bin/python proj/local/make_e40.py
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


def c(layer, s, red, mod):
    n, k = SHAPE[layer]
    out = dict(BM=32, S=s, CPS=1, RED=red if s > 1 else 0, MATH=4, DIAG=0, MOD=mod)
    if mod == "full":
        out.update(M=32, N=n, K=k)
    return out


def both(layer, s, red):
    return [c(layer, s, red, "spec"), c(layer, s, red, "full")]


def targets():
    t = {
        ("unpooler", 32): both("unpooler", 2, 2),
        ("pooler", 32): both("pooler", 3, 2) + both("pooler", 4, 1) + both("pooler", 3, 0),
        ("qkv_gate", 32): both("qkv_gate", 4, 2) + both("qkv_gate", 4, 0) + both("qkv_gate", 6, 2) + both("qkv_gate", 5, 2),
        ("o", 32): both("o", 2, 2),
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
    assert sha.startswith("f4ff43fafffd5af2"), f"pinned v10 changed: {sha[:16]}"
    dg = DGRS.read_text()
    assert ('functions=["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_attrs_x", "dgrs_max_clusters", "dgrs_stages", '
            '"dgrs_stages_d"]' in dg), "v10's exported functions changed: update DGRS_FUNCS in lab/dgrs10.py"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# proj/dg/pinned/dgrs_v10.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "dgrs10.py",
                    PROJ / "lab" / "e40.py"], PROJ / "lab_e40.py", base="scored/submission_v10.py")
    print("dgrs v10 sha256", sha[:16])


if __name__ == "__main__":
    build()

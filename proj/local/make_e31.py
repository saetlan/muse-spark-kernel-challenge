"""Build proj/lab_e31.py (PE31): v10 + the DGRS v3 section (shared/gemm/dg12rs/dgrs.py by muse_prefill_layer, code only,
with its load_inline import) + DGRS_TARGETS + lab/common.py + lab/sweep.py + lab/e31.py. Configs (BM, S, CPS, RED) per
shape; CTAs = N/128 x tiles x S <= 132 x CPS; RED only matters for S > 1 (0 = cluster DSMEM, 1 = global last-arriver).

Run: .venv/bin/python proj/local/make_e31.py   (records the sha256 of the dgrs.py it copied)
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


def c(bm, s, cps=1, red=0):
    return dict(BM=bm, S=s, CPS=cps, RED=red if s > 1 else 0)


def targets():
    t = {
        ("qkv_gate", 32): [c(32, 7, 1, 1), c(32, 4, 1, 0), c(32, 4, 1, 1)],
        ("o", 32): [c(32, 2, 1, 0), c(32, 2, 1, 1), c(32, 1), c(32, 4, 2, 0)],
        ("pooler", 32): [c(32, 4, 1, 0), c(32, 4, 1, 1), c(32, 8, 2, 0), c(32, 8, 2, 1)],
        ("unpooler", 32): [c(32, 2, 1, 0), c(32, 2, 1, 1), c(32, 4, 2, 0), c(32, 4, 2, 1)],
        ("qkv_gate", 128): [c(128, 7, 1, 1), c(128, 4, 1, 0), c(64, 3, 1, 1), c(64, 7, 2, 1), c(64, 4, 2, 0)],
        ("o", 128): [c(128, 2, 1, 0), c(128, 1), c(64, 1), c(64, 2, 2, 0)],
        ("pooler", 128): [c(128, 4, 1, 0), c(128, 4, 1, 1), c(64, 2, 1, 0), c(64, 4, 2, 0)],
        ("unpooler", 128): [c(128, 2, 1, 0), c(128, 2, 1, 1), c(64, 1), c(64, 2, 2, 0)],
        ("qkv_gate", 256): [c(128, 3, 1, 1), c(128, 2, 1, 0), c(64, 3, 2, 1)],
        ("o", 256): [c(128, 1), c(64, 1, 2)],
        ("pooler", 256): [c(128, 2, 1, 0), c(128, 2, 1, 1), c(64, 2, 2, 0)],
        ("unpooler", 256): [c(128, 1), c(64, 1, 2)],
    }
    for (layer, m), cs in t.items():
        n, k = SHAPE[layer]
        kc = k // 64
        for cc in cs:
            ctas = n // 128 * -(-m // cc["BM"]) * cc["S"]
            per = -(-kc // cc["S"])
            assert ctas <= 132 * cc["CPS"] and cc["S"] <= 8 and (cc["S"] - 1) * per < kc, (layer, m, cc, ctas)
            assert cc["CPS"] == 1 or cc["BM"] <= 64, (layer, m, cc)
        assert len({tuple(sorted(cc.items())) for cc in cs}) == len(cs), (layer, m)
    assert c(32, 2, 1, 0) in t[("unpooler", 32)]   # the DIAG section's d0 is this sweep entry
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e31.py"],
                   PROJ / "lab_e31.py", base="scored/submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

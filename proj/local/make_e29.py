"""Build proj/lab_e29.py (PE29): v10 + the DGRS section (shared/gemm/dg12rs/dgrs.py by muse_prefill_layer, code only, with its
load_inline import) + DGRS_TARGETS + lab/common.py + lab/sweep.py + lab/e29.py. Targets (CTAs = N/128 x tiles x S <= 132):
m = 32/128: BM 32 or 128 with the suggested splits (qkv 7, o 2, pooler 4, unpooler 2) and a smaller one; m = 128 also
BM 64 (2 tiles); m = 256: BM 128, 2 tiles (qkv 3, o 1, pooler 2, unpooler 1).

Run: .venv/bin/python proj/local/make_e29.py   (records the sha256 of the dgrs.py it copied)
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ.parent / "shared" / "gemm" / "dg12rs" / "dgrs.py"
N_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192}


def c(bm, s):
    return dict(BM=bm, S=s)


def targets():
    t = {
        ("qkv_gate", 32): [c(32, 7), c(32, 3)], ("o", 32): [c(32, 2), c(32, 1)],
        ("pooler", 32): [c(32, 4), c(32, 2)], ("unpooler", 32): [c(32, 2), c(32, 1)],
        ("qkv_gate", 128): [c(128, 7), c(64, 3)], ("o", 128): [c(128, 2), c(64, 1)],
        ("pooler", 128): [c(128, 4), c(64, 2)], ("unpooler", 128): [c(128, 2), c(64, 1)],
        ("qkv_gate", 256): [c(128, 3)], ("o", 256): [c(128, 1)],
        ("pooler", 256): [c(128, 2)], ("unpooler", 256): [c(128, 1)],
    }
    for (layer, m), cs in t.items():
        for cc in cs:
            ctas = N_ROWS[layer] // 128 * -(-m // cc["BM"]) * cc["S"]
            assert ctas <= 132, (layer, m, cc, ctas)
    return t


def build():
    sha = hashlib.sha256(DGRS.read_bytes()).hexdigest()
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(DGRS)
               + f"\n\n# shared/gemm/dg12rs/dgrs.py sha256 {sha[:16]}\nDGRS_TARGETS = {targets()!r}\n")
    sec_path = PROJ / "lab" / "_dgrs_section.py"
    sec_path.write_text(section)
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "e29.py"],
                   PROJ / "lab_e29.py", base="submission_v10.py")
    print("dgrs.py sha256", sha[:16])


if __name__ == "__main__":
    build()

"""Build proj/submission_v12.py = scored/submission_v11b.py with one change in F12M's CUDA source (PE45, PF62): the KSW-sum
array part[KSW][RB][MT] gets a row stride of MT + 3, so its fragment writes and sum reads are <= 2-way bank-conflicted
instead of 4-16-way. Same math, same summation order, bit-identical outputs; every F12M shape (qkv m8 / m16, pooler m8 / m16,
unpooler m16, o m8) runs the padded kernel.

Run: .venv/bin/python proj/local/make_v12.py
"""

import hashlib
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
OLD = "  __shared__ float part[KSW][RB][MT];\n"
NEW = "  __shared__ float part[KSW][RB][MT + 3];   // row stride MT + 3: <= 2-way banks (PE45, PF62)\n"


def build(out=PROJ / "submission_v12.py"):
    src = (PROJ / "scored" / "submission_v11b.py").read_text()
    assert hashlib.sha256(src.encode()).hexdigest().startswith("cfe20913"), "base is not v11b"
    i, j = src.index('F12M_SRC = r"""'), src.index('"""', src.index('F12M_SRC = r"""') + 20)
    assert src.count(OLD) == 1 and i < src.index(OLD) < j, "the declaration must be F12M's, exactly once"
    out_src = src.replace(OLD, NEW)
    Path(out).write_text(out_src)
    print(f"wrote {out} sha256 {hashlib.sha256(out_src.encode()).hexdigest()[:16]} (v11b + F12M part[] padded)")


if __name__ == "__main__":
    build()

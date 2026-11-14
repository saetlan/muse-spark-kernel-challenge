"""Build proj/submission_v9.py: v7a's choices (chosen.V6 + chosen.PE10) plus PE19's F12M x-ring winners, rebuilt with
the CS/XD-capable F12M section (make_v7), then the HL section with PE18's router winners updated by PE19's
(make_hl). This is exactly how PE19's lab base (proj/submission_v8b.py) was built, plus the winners.

Run: .venv/bin/python proj/local/make_v9x.py   (fill XD_WON / HL_WON from PE19's "V15 CHOICE" lines)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chosen  # noqa: E402
import make_hl  # noqa: E402
import make_v7  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]

# PE19's "V15 CHOICE" lines (job 3157060493544dcf) that differ from v8.
XD_WON = {("unpooler", 16): {'kind': 'f12m', 'NT': 2, 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 8, 'F12': True, 'XD': 1}}
HL_WON = {("router", 256): {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 64, 'split': 16, 'warps': 8, 'stages': 3},
          # PE20 (job 796600c789d84aa9): router m16 hl_gemm beat P9 by 3.3% in the lab (v10 = v9 + this line)
          ("router", 16): {'kind': 'hl_gemm', 'BM': 16, 'BN': 32, 'BK': 128, 'split': 32, 'warps': 2, 'stages': 3}}


def build(xd_won, hl_won, out="submission_v9.py"):
    assert all(c["kind"] == "f12m" and "XD" in c for c in xd_won.values()), xd_won
    make_v7.build({**chosen.V6, **chosen.PE10, **xd_won}, "submission_v9_pre.py")
    make_hl.build(str(PROJ / out), "submission_v9_pre.py", {**make_hl.HL_CHOSEN, **hl_won})


if __name__ == "__main__":
    build(XD_WON, HL_WON)

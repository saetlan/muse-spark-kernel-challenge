"""Build proj/submission_v4.py: v3's choices plus PE6's P9 router configs, with the P9 section compiled
for exactly those instances (fast nvcc). The configs come from PE6's "V4 CHOICE" log lines, pasted
into ROUTER_P9 below.

Run: .venv/bin/python proj/local/make_v4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p9"))
import make_p9  # noqa: E402
import make_v1  # noqa: E402
import make_v3  # noqa: E402
import p9_configs  # noqa: E402

# From PE6 (job 231afb123c864c70), its "V4 CHOICE" lines.
ROUTER_P9 = {
    ("router", 8): {"kind": "p9", "NT": 1, "RTW": 1, "RG": 2, "KSW": 8, "D": 2, "CS": 8},
    ("router", 16): {"kind": "p9", "NT": 2, "RTW": 1, "RG": 1, "KSW": 8, "D": 2, "CS": 8},
}

CHOICE = dict(make_v3.CHOICE)
CHOICE.update(ROUTER_P9)


def main():
    assert ROUTER_P9, "paste PE6's V4 CHOICE configs into ROUTER_P9 first"
    make_v1.build(CHOICE, "submission_v4_base.py", "the PE1-PE3 and PE6 labs")
    p9_configs.V4 = tuple(p9_configs.instance(c, layer) for (layer, m), c in ROUTER_P9.items())
    make_p9.build(str(make_v1.PROJ / "submission_v4.py"), base="submission_v4_base.py", instances="V4")


if __name__ == "__main__":
    main()

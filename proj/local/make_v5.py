"""Build proj/submission_v5.py: v4's choices plus PE8's F12V winners at m = 1 (each keeps v4's kernel for
that shape as its fallback), with the P9 section for v4's router instances and the F12V section for
exactly the chosen F12V instances (short nvcc). Paste PE8's "V5 CHOICE" lines into F12V_M1.

Run: .venv/bin/python proj/local/make_v5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p9"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f12v"))
import f12v_configs  # noqa: E402
import make_f12v  # noqa: E402
import make_p9  # noqa: E402
import make_v1  # noqa: E402
import make_v4  # noqa: E402
import p9_configs  # noqa: E402

# From PE8 (job 26edaae38e814a97), its "V5 CHOICE" lines.
F12V_M1 = {
    ("pooler", 1): {"kind": "f12v", "R": 4, "RG": 4, "KSW": 2, "D": 1},
    ("qkv_gate", 1): {"kind": "f12v", "R": 3, "RG": 3, "KSW": 2, "D": 2},
}


def main():
    assert F12V_M1, "paste PE8's V5 CHOICE configs into F12V_M1 first"
    choice = dict(make_v4.CHOICE)
    for key, c in F12V_M1.items():
        assert c["kind"] == "f12v" and key[1] == 1, (key, c)
        choice[key] = dict(c, fallback=make_v4.CHOICE.get(key))
    make_v1.build(choice, "submission_v5_base.py", "the PE1-PE3, PE6 and PE8 labs")
    p9_configs.V4 = tuple(p9_configs.instance(c, layer) for (layer, m), c in make_v4.ROUTER_P9.items())
    make_p9.build(str(make_v1.PROJ / "submission_v5_p9.py"), base="submission_v5_base.py", instances="V4")
    f12v_configs.V5 = tuple(dict.fromkeys(f12v_configs.instance(c, layer) for (layer, m), c in F12V_M1.items()))
    make_f12v.build(str(make_v1.PROJ / "submission_v5.py"), base="submission_v5_p9.py", instances="V5")


if __name__ == "__main__":
    main()

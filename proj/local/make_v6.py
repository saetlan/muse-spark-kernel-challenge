"""Build proj/submission_v6.py: v4's choices plus the F12V (m = 1) and F12M (m = 8/16/32) winners pasted
from a lab's "V5 CHOICE" / "V6 CHOICE" lines into CHOSEN (each keeps v4's kernel for its shape as the
fallback), with the P9 section for v4's router instances, the F12V section for exactly the chosen F12V
instances and the F12M section for exactly the chosen F12M instances (short nvcc).

Run: .venv/bin/python proj/local/make_v6.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for sub in ("p9", "f12v", "f12m"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / sub))
import f12m_configs  # noqa: E402
import f12v_configs  # noqa: E402
import make_f12m  # noqa: E402
import make_f12v  # noqa: E402
import make_p9  # noqa: E402
import make_v1  # noqa: E402
import make_v4  # noqa: E402
import p9_configs  # noqa: E402

# {(layer, m): {"kind": "f12v" | "f12m", ...}} from the lab's V5/V6 CHOICE lines.
CHOSEN = {}


def build(chosen, out_name="submission_v6.py"):
    assert chosen, "paste the lab's V6 CHOICE configs into CHOSEN first"
    choice = dict(make_v4.CHOICE)
    for key, c in chosen.items():
        assert c["kind"] in ("f12v", "f12m"), (key, c)
        assert (c["kind"] == "f12v") == (key[1] == 1), (key, c)
        choice[key] = dict(c, fallback=make_v4.CHOICE.get(key))
    stem = out_name.replace(".py", "")
    make_v1.build(choice, f"{stem}_base.py", "the PE1-PE3, PE6 and PE8-PE9 labs")
    p9_configs.V4 = tuple(p9_configs.instance(c, layer) for (layer, m), c in make_v4.ROUTER_P9.items())
    make_p9.build(str(make_v1.PROJ / f"{stem}_p9.py"), base=f"{stem}_base.py", instances="V4")
    v = tuple(dict.fromkeys(f12v_configs.instance(c, layer) for (layer, m), c in chosen.items() if c["kind"] == "f12v"))
    f12v_configs.CHOSEN = v or (f12v_configs.LAB[0],)      # the section needs at least one instance
    make_f12v.build(str(make_v1.PROJ / f"{stem}_f12v.py"), base=f"{stem}_p9.py", instances="CHOSEN")
    mm = tuple(dict.fromkeys(f12m_configs.instance(c, layer, m) for (layer, m), c in chosen.items() if c["kind"] == "f12m"))
    if mm:
        f12m_configs.CHOSEN = mm
        make_f12m.build(str(make_v1.PROJ / out_name), base=f"{stem}_f12v.py", instances="CHOSEN")
    else:
        (make_v1.PROJ / out_name).write_text((make_v1.PROJ / f"{stem}_f12v.py").read_text())
        print(f"wrote {out_name} (no F12M shapes)")


if __name__ == "__main__":
    build(CHOSEN)

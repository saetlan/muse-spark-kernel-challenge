"""Build a scored file from v4's choices plus any lab winners pasted into CHOSEN ({(layer, m): config}
from the labs' V5/V6/V7 CHOICE lines). Kinds: f12v (m = 1), f12m (m = 8..32), wg (m = 128/256), and
plain Triton configs (gemm / swap / fma, no section needed). New kernels keep v4's kernel for their
shape as the fallback. Sections: P9 (v4's router instances) always; F12V when an f12v or f12m config is
chosen (F12M reuses its format code); F12M and WG with exactly the chosen instances.

Run: .venv/bin/python proj/local/make_v7.py [out.py]   (default submission_v7.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for sub in ("p9", "f12v", "f12m", "wg", "sx"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / sub))
import f12m_configs  # noqa: E402
import f12v_configs  # noqa: E402
import make_f12m  # noqa: E402
import make_f12v  # noqa: E402
import make_p9  # noqa: E402
import make_v1  # noqa: E402
import make_v4  # noqa: E402
import make_sx  # noqa: E402
import make_wg  # noqa: E402
import p9_configs  # noqa: E402
import sx_configs  # noqa: E402
import wg_configs  # noqa: E402

CHOSEN = {}
NEW_KINDS = ("f12v", "f12m", "wg", "wgf", "sx")


def build(chosen, out_name="submission_v7.py", sxw_lab=None):
    assert chosen, "paste the labs' CHOICE configs into CHOSEN first"
    choice = dict(make_v4.CHOICE)
    for key, c in chosen.items():
        c = {k: v for k, v in c.items() if k != "fallback"}
        if c["kind"] in NEW_KINDS:
            assert (c["kind"] == "f12v") == (key[1] == 1) and (c["kind"] != "wg" or key[1] >= 128), (key, c)
            assert c["kind"] != "wgf" or key[1] >= 16, (key, c)
            assert c["kind"] != "sx" or key[1] >= 32, (key, c)
            c = dict(c, fallback=make_v4.CHOICE.get(key))
        choice[key] = c
    stem = out_name.replace(".py", "")
    make_v1.build(choice, f"{stem}_base.py", "the PE1-PE3, PE6 and PE8-PE11 labs")
    p9_configs.V4 = tuple(p9_configs.instance(c, layer) for (layer, m), c in make_v4.ROUTER_P9.items())
    make_p9.build(str(make_v1.PROJ / f"{stem}_p9.py"), base=f"{stem}_base.py", instances="V4")
    cur = f"{stem}_p9.py"
    kinds = {c["kind"] for c in choice.values()}
    if kinds & {"f12v", "f12m", "wgf", "sx"} or sxw_lab:
        v = tuple(dict.fromkeys(f12v_configs.instance(c, layer) for (layer, m), c in choice.items() if c["kind"] == "f12v"))
        f12v_configs.CHOSEN = v or (f12v_configs.LAB[0],)
        make_f12v.build(str(make_v1.PROJ / f"{stem}_f12v.py"), base=cur, instances="CHOSEN")
        cur = f"{stem}_f12v.py"
    if kinds & {"f12m", "wgf", "sx"} or sxw_lab:   # WGF and SX pack their planes with the F12M section's code
        f12m_configs.CHOSEN = tuple(dict.fromkeys(f12m_configs.instance(c, layer, m) for (layer, m), c in choice.items()
                                                  if c["kind"] == "f12m")) or (f12m_configs.LAB[0],)
        make_f12m.build(str(make_v1.PROJ / f"{stem}_f12m.py"), base=cur, instances="CHOSEN")
        cur = f"{stem}_f12m.py"
    if kinds & {"wg", "wgf"}:
        wg_configs.CHOSEN = tuple(dict.fromkeys(wg_configs.instance(c, m) for (layer, m), c in choice.items() if c["kind"] == "wg"))
        wg_configs.CHOSEN_WGF = tuple(dict.fromkeys(wg_configs.wgf_instance(c, m) for (layer, m), c in choice.items()
                                                    if c["kind"] == "wgf"))
        make_wg.build(str(make_v1.PROJ / f"{stem}_wg.py"), base=cur, instances="CHOSEN",
                      wgf_instances="CHOSEN_WGF" if wg_configs.CHOSEN_WGF else None)
        cur = f"{stem}_wg.py"
    if "sx" in kinds or sxw_lab:
        sx_configs.CHOSEN = tuple(dict.fromkeys(sx_configs.instance(c) for (layer, m), c in choice.items()
                                                if c["kind"] == "sx" and not c.get("W")))
        w_chosen = tuple(sx_configs.winstance(c) for (layer, m), c in choice.items() if c["kind"] == "sx" and c.get("W"))
        sx_configs.W_CHOSEN = tuple(dict.fromkeys(w_chosen + (getattr(sx_configs, sxw_lab) if sxw_lab else ())))
        make_sx.build(str(make_v1.PROJ / f"{stem}_sx.py"), base=cur, instances="CHOSEN", lab=bool(sxw_lab),
                      w_instances="W_CHOSEN")
        cur = f"{stem}_sx.py"
    (make_v1.PROJ / out_name).write_text((make_v1.PROJ / cur).read_text())
    print(f"wrote proj/{out_name} (from {cur}; kinds {sorted(kinds)})")


if __name__ == "__main__":
    build(CHOSEN, sys.argv[1] if len(sys.argv) > 1 else "submission_v7.py")

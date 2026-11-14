"""Build an F12M submission file: a base file (one with the P9 and F12V sections, e.g. submission_v5.py or
submission_f12vlab.py) byte for byte, plus the F12M section (proj/f12m/f12m_layout.py and
proj/f12m/f12m.py without their docstrings and imports, the setup and run functions below) and
F12M_INSTANCES set to the given instance list.

Run: .venv/bin/python proj/local/make_f12m.py --base submission_f12vlab.py [--instances LAB] [--lab] out.py
(the instance list names a tuple in proj/f12m/f12m_configs.py; --lab adds F12M_LAB_CANDIDATES)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " F12M (proj/f12m): 12-bit lossless weights, mma.sync at m = 2..32\n\n"

SETUP = '''
F12M_ENABLED = True
F12M_MOD = None


def f12m_setup(state):
    """Pack the BF16 weights into F12M planes (the F12V format in mma A-fragment order; a layer with a
    row of more than 7 escapes is left out), then build every F12M instance. A failed build leaves no
    module, so every F12M shape runs its fallback (slower, still correct)."""
    global F12M_MOD
    state["f12m"] = {}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        planes = f12m_device_planes(state[name])
        if planes is not None:
            state["f12m"][name] = planes
    try:
        F12M_MOD = f12m_build(F12M_INSTANCES)
    except Exception as exc:
        print(f"f12m build FAILED, F12M shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        F12M_MOD = None
        state["f12m"] = {}


def run_f12m(c, x, layer, n, state):
    """c: dict(kind="f12m", NT, RTW, RG, KSW, D, F12[, CS][, XD][, TG], fallback). F12 = False runs the same kernel on
    P9's BF16 fragments; CS > 1 splits K across a cluster of CS CTAs; XD < D keeps a shallower x ring; TG > 1
    splits the tokens across TG warps sharing each weight chunk (m <= 8 NT TG). Without a module, planes or fragments
    for this layer, or with more tokens than 8 NT, it runs c["fallback"] (the shape's previous kernel) or
    SGLang's path."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    wp = pl["p"] if c["F12"] and pl is not None else (state.get("p9", {}).get(layer) if not c["F12"] else None)
    if F12M_MOD is None or wp is None or m > 8 * c["NT"] * c.get("TG", 1):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    rec = pl["rec"] if c["F12"] else state["p9_dummy"]
    cs, xd, tg = c.get("CS", 1), c.get("XD", c["D"]), c.get("TG", 1)
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd, tg) if tg > 1 else ((cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ()))
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12M_MOD, f12m_name(inst))(wp, rec, x, out, m, n)
    return out
'''


def build(out, base, instances="LAB", lab=False):
    sys.path.insert(0, str(PROJ / "f12m"))
    import f12m_configs
    insts = getattr(f12m_configs, instances)
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\nF12M_ENABLED = False\n" in src, f"{base} has no F12M hook (rebuild it from submission.py)"
    assert "\ndef p9_pack(" in src and "\ndef f12v_words(" in src, f"{base} lacks the P9 or F12V section"
    section = (body(PROJ / "f12m" / "f12m_layout.py") + "\n\n" + body(PROJ / "f12m" / "f12m.py")).replace(
        "F12M_INSTANCES = ()", f"F12M_INSTANCES = {tuple(insts)!r}")
    section += SETUP
    if lab:  # candidate configs per (layer, m) whose instances are compiled in, for the lab sweep
        cands = {}
        for layer in f12m_configs.SHAPES:
            for m in f12m_configs.MS:
                keep = [dict(c, NT=m // 8) for c in f12m_configs.candidates(layer, m)
                        if f12m_configs.instance(c, layer, m) in insts]
                if keep:
                    cands[(layer, m)] = keep
        section += f"\n\nF12M_LAB_CANDIDATES = {cands!r}\n"
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} F12M instances)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base, inst, lab = None, "LAB", False
    while args and args[0].startswith("--"):
        if args[0] == "--base":
            base, args = args[1], args[2:]
        elif args[0] == "--instances":
            inst, args = args[1], args[2:]
        elif args[0] == "--lab":
            lab, args = True, args[1:]
    assert base, "--base is required"
    build(args[0], base, inst, lab)

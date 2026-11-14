"""Build an F12V submission file: a base file (default submission_v4.py) byte for byte, plus the F12V
section (proj/f12v/f12v_layout.py and proj/f12v/f12v.py without their docstrings and imports, the
setup and run functions below) and F12V_INSTANCES set to the given instance list.

Run: .venv/bin/python proj/local/make_f12v.py [--base submission_v4.py] [--instances LAB] [--lab] out.py
(the instance list names a tuple in proj/f12v/f12v_configs.py; --lab adds F12V_LAB_CANDIDATES)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " F12V (proj/f12v): 12-bit lossless weights, m = 1 GEMV\n\n"

SETUP = '''
F12V_ENABLED = True
F12V_MOD = None
F12V_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")


def f12v_setup(state):
    """Pack the BF16 weights into the 12-bit format (lossless: per-row exponent window plus exact
    escapes; a layer with a row of more than 7 escapes is left out and its shapes keep their other
    kernels), then build every F12V instance."""
    global F12V_MOD
    state["f12v"] = {}
    for name in F12V_LAYERS:
        planes = f12v_device_planes(state[name])
        if planes is not None:
            state["f12v"][name] = planes
    try:
        F12V_MOD = f12v_build(F12V_INSTANCES)
    except Exception as exc:  # no module: every F12V shape runs its fallback (slower, still correct)
        print(f"f12v build FAILED, F12V shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        state["f12v"] = {}


def run_f12v(c, x, layer, n, state):
    """c: dict(kind="f12v", R, RG, KSW, D, fallback). m = 1 only; without planes for this layer (more
    than 7 escapes in a row) it runs c["fallback"], the kernel the shape had before, or SGLang's path."""
    pl = state["f12v"].get(layer)
    if pl is None or x.shape[0] != 1:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    k = x.shape[1]
    inst = (c["R"], c["RG"], c["KSW"], k // 1024 // c["KSW"], c["D"])
    x = x.contiguous()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12V_MOD, f12v_name(inst))(pl["p"], pl["rec"], x, out, n)
    return out
'''


def build(out, base="submission_v4.py", instances="LAB", lab=False):
    sys.path.insert(0, str(PROJ / "f12v"))
    import f12v_configs
    insts = getattr(f12v_configs, instances)
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\nF12V_ENABLED = False\n" in src, f"{base} has no F12V hook (rebuild it from submission.py)"
    section = (body(PROJ / "f12v" / "f12v_layout.py") + "\n\n" + body(PROJ / "f12v" / "f12v.py")).replace(
        "F12V_INSTANCES = ()", f"F12V_INSTANCES = {tuple(insts)!r}")
    if "import tempfile" not in src:
        section = "import tempfile\n\n" + section
    section += SETUP
    if lab:  # candidate configs per layer whose instances are compiled in, for the lab sweep
        cands = {}
        for layer in f12v_configs.ROWS_M1:
            keep = [c for c in f12v_configs.candidates(layer) if f12v_configs.instance(c, layer) in insts]
            if keep:
                cands[layer] = keep
        section += f"\n\nF12V_LAB_CANDIDATES = {cands!r}\n"
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} F12V instances)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base, inst, lab = "submission_v4.py", "LAB", False
    while args and args[0].startswith("--"):
        if args[0] == "--base":
            base, args = args[1], args[2:]
        elif args[0] == "--instances":
            inst, args = args[1], args[2:]
        elif args[0] == "--lab":
            lab, args = True, args[1:]
    build(args[0], base, inst, lab)

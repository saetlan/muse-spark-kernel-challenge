"""Build an SX submission file: a base file (one with the F12M section, e.g. submission_v7a.py) byte for byte,
plus the SX section (proj/sx/sx_layout.py and proj/sx/sx.py without docstrings and imports, the setup and run
functions below), SX_INSTANCES set to the given instance list.

Run: .venv/bin/python proj/local/make_sx.py --base submission_v7a.py [--instances LAB] [--lab] out.py
(--instances names a tuple in proj/sx/sx_configs.py; --lab adds SX_LAB_CANDIDATES)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " SX (proj/sx): 12-bit mma.sync GEMM, m = 32/128/256\n\n"

SETUP = '''
SX_ENABLED = True
SX_MOD = None


def sx_setup(state):
    """Pack the four BF16 layers as 12-bit planes in natural k order (the F12M format; a layer with a row of
    more than 7 escapes is left out) plus per-(tile, chunk) escape flags, then build every SX instance. A
    failed build leaves no module, so every SX shape runs its fallback."""
    global SX_MOD
    state["sx"] = {}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = state[name]
        planes = f12m_device_planes(w, natural=True)
        if planes is not None:
            state["sx"][name] = {"p": planes["p"], "rec": planes["rec"],
                                 "flags": sx_flags(planes["rec"], w.shape[0], w.shape[1])}
    try:
        SX_MOD = sx_build(SX_INSTANCES, SXW_INSTANCES)
        for inst in SX_INSTANCES:
            getattr(SX_MOD, sx_name(inst) + "_attr")()
        for inst in SXW_INSTANCES:
            getattr(SX_MOD, sxw_name(inst) + "_attr")()
    except Exception as exc:
        print(f"sx build FAILED, SX shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        SX_MOD = None
        state["sx"] = {}


def run_sx(c, x, layer, n, state):
    """c: dict(kind="sx", NT8, TG, RTW, RGN, KSW, NS, D, XS, F12, CS, fallback) for SX (mma.sync), or
    dict(kind="sx", W=True, NB, WGS, KSW, NS, P, CS, F12, fallback) for SXW (wgmma). Without a module or planes
    for this layer, or a shape the instance doesn't cover, it runs c["fallback"] or SGLang's path."""
    m, k = x.shape
    pl = state.get("sx", {}).get(layer)
    if c.get("W"):
        inst = (c["NB"], c["WGS"], c["KSW"], c["NS"], c["P"], c["CS"])
        ok = m <= 128 * c["NB"] and n % (64 * c["WGS"]) == 0
    else:
        inst = (c["NT8"], c["TG"], c["RTW"], c["RGN"], c["KSW"], c["NS"], c["D"], c["XS"], c["F12"], c["CS"])
        ok = m <= 8 * c["NT8"] * c["TG"] and n % (16 * c["RTW"] * c["RGN"]) == 0
    if SX_MOD is None or pl is None or not c["F12"] or not ok or k != 64 * c["NS"] * c["KSW"] * c["CS"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if c.get("W"):
        getattr(SX_MOD, sxw_name(inst))(pl["p"], pl["rec"], x.contiguous(), out, n)
    else:
        getattr(SX_MOD, sx_name(inst))(pl["p"], pl["rec"], pl["flags"], x.contiguous(), out, n)
    return out
'''


def build(out, base, instances="LAB", lab=False, w_instances=None):
    sys.path.insert(0, str(PROJ / "sx"))
    import sx_configs
    insts = getattr(sx_configs, instances)
    w_insts = getattr(sx_configs, w_instances) if w_instances else ()
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\nSX_ENABLED = False\n" in src and 'return run_sx(c, x, layer, n, state)' in src, \
        f"{base} has no SX hook (rebuild it from submission.py)"
    assert "\ndef f12m_device_planes(" in src, f"{base} lacks the F12M section (SX packs its planes with it)"
    assert "\nimport tempfile\n" in src
    section = (body(PROJ / "sx" / "sx_layout.py") + "\n\n" + body(PROJ / "sx" / "sx.py")).replace(
        "SX_INSTANCES = ()", f"SX_INSTANCES = {tuple(insts)!r}").replace(
        "SXW_INSTANCES = ()", f"SXW_INSTANCES = {tuple(w_insts)!r}")
    section += SETUP
    if lab:
        cands = {key: [sx_configs.config(i) for i in v if i in insts] for key, v in sx_configs.CANDIDATES.items()}
        wc = {key: [sx_configs.wconfig(i) for i in v if i in w_insts] for key, v in sx_configs.W_CANDIDATES.items()}
        section += f"\n\nSX_LAB_CANDIDATES = {cands!r}\nSXW_LAB_CANDIDATES = {wc!r}\n"
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} SX, {len(w_insts)} SXW instances)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base, inst, lab, winst = None, "LAB", False, None
    while args and args[0].startswith("--"):
        if args[0] == "--base":
            base, args = args[1], args[2:]
        elif args[0] == "--instances":
            inst, args = args[1], args[2:]
        elif args[0] == "--w-instances":
            winst, args = args[1], args[2:]
        elif args[0] == "--lab":
            lab, args = True, args[1:]
    assert base, "--base is required"
    build(args[0], base, inst, lab, winst)

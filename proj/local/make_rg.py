"""Build an RG submission file: a base file byte for byte plus the RG section (proj/rg/rg_layout.py and
proj/rg/rg.py without docstrings and imports, the setup and run functions below), RG_INSTANCES set to the
given instance list.

Run: .venv/bin/python proj/local/make_rg.py --base submission_v7a.py [--instances LAB] [--lab] out.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " RG (proj/rg): router hi + lo mma.sync kernel, m = 32/128/256\n\n"

SETUP = '''
RG_ENABLED = True
RG_MOD = None


def rg_setup(state):
    """Pack the FP32 router as SGLang's BF16 hi + lo pair in A-fragment order (natural k), build every RG
    instance; a failed build leaves no module, so every RG shape runs its fallback."""
    global RG_MOD
    state["rg"] = rg_pack(state["router"])
    try:
        RG_MOD = rg_build(RG_INSTANCES)
        for inst in RG_INSTANCES:
            getattr(RG_MOD, rg_name(inst) + "_attr")()
    except Exception as exc:
        print(f"rg build FAILED, RG shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        RG_MOD = None


def run_rg(c, x, layer, n, state):
    """c: dict(kind="rg", RTC, NTW, CS, XS, D, fallback). Router only; m <= the instance's token count."""
    inst = (c["RTC"], c["NTW"], c["CS"], c["XS"], c["D"])
    m = x.shape[0]
    if RG_MOD is None or layer != "router" or m > rg_mt(inst):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    getattr(RG_MOD, rg_name(inst))(state["rg"], x.contiguous(), out)
    return out
'''


def build(out, base, instances="LAB", lab=False):
    sys.path.insert(0, str(PROJ / "rg"))
    import rg_configs
    insts = getattr(rg_configs, instances)
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\nRG_ENABLED = False\n" in src, f"{base} has no RG hook (rebuild it from submission.py)"
    section = (body(PROJ / "rg" / "rg_layout.py") + "\n\n" + body(PROJ / "rg" / "rg.py")).replace(
        "RG_INSTANCES = ()", f"RG_INSTANCES = {tuple(insts)!r}")
    if "import tempfile" not in src:
        section = "import tempfile\n\n" + section
    section += SETUP
    if lab:
        cands = {("router", m): [c for c in rg_configs.candidates(m) if rg_configs.instance(c) in insts] for m in rg_configs.MS}
        section += f"\n\nRG_LAB_CANDIDATES = {cands!r}\n"
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} RG instances)")


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

"""Build a P9 submission file: a base file (default submission_v2.py) byte for byte, plus the P9
section (proj/p9/p9_layout.py and proj/p9/p9.py without their docstrings and imports, the setup and
run functions below) and P9_INSTANCES set to the given instance list.

Run: .venv/bin/python proj/local/make_p9.py [--base submission_v2.py] [--instances name] out.py
(the instance list names a tuple in proj/p9/p9_configs.py; default LAB)
"""

import ast
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " P9 (proj/p9): swap-AB mma.sync kernel for m <= 32\n\n"

SETUP = '''
P9_ENABLED = True
P9_MOD = None
P9_BF16_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
P9_MAX_SMEM = 227 * 1024


def p9_setup(state):
    """Pack the weights into fragment order (lossless, same bytes; router as SGLang's hi + lo), build
    every P9 instance, allow the full shared memory, and launch each instance once on zeros so no
    compile or module load lands inside a capture."""
    global P9_MOD
    state["p9"] = {name: p9_pack(state[name]) for name in P9_BF16_LAYERS}
    state["p9"]["router"] = p9_pack_hilo(state["router"])
    dev = state["router"].device
    state["p9_dummy"] = torch.zeros(16, dtype=torch.int32, device=dev)
    state["p22"] = {}
    if any(len(inst) > 8 and inst[8] for inst in P9_INSTANCES):
        for name in P9_BF16_LAYERS:
            planes = p22_device_planes(state[name], state[name].shape[0])
            if planes is not None:
                state["p22"][name] = planes
    P9_MOD = p9_build(P9_INSTANCES)
    for inst in P9_INSTANCES:
        getattr(P9_MOD, p9_name(inst) + "_attr")(P9_MAX_SMEM)


def run_p9(c, x, layer, n, state):
    """c: dict(kind="p9", NT, RTW, RG, KSW, D, CS[, F12])."""
    hilo = layer == "router"
    f12 = bool(c.get("F12", False))
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], hilo, hilo) + ((True,) if f12 else ())
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32 if hilo else torch.bfloat16, device=x.device)
    smem = p9_smem_bytes(c["NT"], c["RTW"], c["RG"], c["KSW"], c["CS"], k)
    dmy = state["p9_dummy"]
    if f12:
        pl = state["p22"][layer]
        args = (dmy, pl["q"], pl["r"], pl["tops"], pl["erow"], pl["ecol"], pl["ebits"])
    else:
        args = (state["p9"][layer], dmy, dmy, dmy, dmy, dmy, dmy)
    getattr(P9_MOD, p9_name(inst))(*args, x.contiguous(), out, m, n, k, smem)
    return out
'''


def body(path):
    """A module's source without its docstring and import lines."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            drop.update(range(node.lineno - 1, node.end_lineno))
    return "".join(l for i, l in enumerate(lines) if i not in drop).strip() + "\n"


def build(out, base="submission_v2.py", instances="LAB", lab=False):
    sys.path.insert(0, str(PROJ / "p9"))
    import p9_configs
    insts = getattr(p9_configs, instances)
    src = (PROJ / base).read_text().rstrip() + "\n"
    section = (body(PROJ / "p9" / "p9_layout.py") + "\n\n" + body(PROJ / "p9" / "p22_layout.py") + "\n\n"
               + body(PROJ / "p9" / "p9.py")).replace(
        "P9_INSTANCES = ()", f"P9_INSTANCES = {tuple(insts)!r}")
    assert "import tempfile" not in section
    section = "import tempfile\n\n" + section + SETUP
    if lab:  # candidate configs per (layer, m) whose instances are compiled in, for the lab sweep
        cands = {}
        for layer in p9_configs.SHAPES:
            for m in p9_configs.SMALL_M:
                keep = [c for c in p9_configs.candidates(layer, m) if p9_configs.instance(c, layer) in insts]
                keep += [c for c in p9_configs.candidates_f12(layer, m) if p9_configs.instance(c, layer) in insts]
                if keep:
                    cands[(layer, m)] = keep
        section += f"\n\nP9_LAB_CANDIDATES = {cands!r}\n"
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} P9 instances)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base, inst, lab = "submission_v2.py", "LAB", False
    while args and args[0].startswith("--"):
        if args[0] == "--base":
            base, args = args[1], args[2:]
        elif args[0] == "--instances":
            inst, args = args[1], args[2:]
        elif args[0] == "--lab":
            lab, args = True, args[1:]
    build(args[0], base, inst, lab)

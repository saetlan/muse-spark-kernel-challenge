"""v3 must be exactly baseline.py + v2/gmm.py + v3/r3.py + make_v3.py's two edits (ASTs, docstrings
stripped; R3_CONFIGS may hold a lab's config line), with SGLang's Marlin path and v2's grouped path
untouched (the latter function by function against the scored v2 snapshot).

Run: .venv/bin/python prefill_layer/local/check_v3_vs_baseline.py
"""

import ast
import importlib.util
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_v3", DIR / "local" / "make_v3.py")
make_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_v3)


def strip(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


sub = (DIR / "submission.py").read_text()
if not sub.startswith("# okc: problem=muse_prefill_layer\n"):
    raise SystemExit("FAIL: first line")
line = next(l for l in sub.splitlines() if l.startswith("R3_CONFIGS = "))
configs = line[len("R3_CONFIGS = "):]
ast.literal_eval(configs)
if strip(ast.parse(sub)) != strip(ast.parse(make_v3.build_text(None if configs == "{}" else configs))):
    raise SystemExit("FAIL: submission.py is not baseline + v2/gmm.py + v3/r3.py + the two edits; rerun make_v3.py")
fns = lambda src: {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
base, v2, v3 = fns((DIR / "baseline.py").read_text()), fns((DIR / "scored/v2.8eb7a168b38743a1.py").read_text()), fns(sub)
for name in ("_experts", "_marlin_bank", "_gemm"):
    if base[name] != v3[name]:
        raise SystemExit(f"FAIL: {name} differs from baseline")
for name in ("_experts_gmm", "_dequant", "_gmm_prepare", "_nvfp4_dequant_kernel", "_gather_rows_kernel",
             "_gmm_act_kernel", "_gmm_combine_kernel", "_e2m1_value", "_gmm_selftest_routing"):
    if v2[name] != v3[name]:
        raise SystemExit(f"FAIL: {name} differs from the scored v2")
print(f"ok: submission.py = baseline + v2 + v3 fragments + 2 edits (R3_CONFIGS = {configs}); Marlin and v2 paths untouched")

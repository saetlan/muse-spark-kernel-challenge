"""v6 must be exactly baseline.py + v2/gmm.py + v6/r14c.py + make_v6.py's two edits (ASTs, docstrings
stripped; R14_CONFIGS may hold a lab's config line), with SGLang's Marlin path and v2's grouped path
untouched (the latter function by function against the scored v2 snapshot).

Run: .venv/bin/python prefill_layer/local/check_v6_vs_baseline.py
"""

import ast
import importlib.util
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_v6", DIR / "local" / "make_v6.py")
make_v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_v6)


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
line = next(l for l in sub.splitlines() if l.startswith("R14_CONFIGS = "))
configs = line[len("R14_CONFIGS = "):]
ast.literal_eval(configs)
if strip(ast.parse(sub)) != strip(ast.parse(make_v6.build_text(None if configs == "{}" else configs))):
    raise SystemExit("FAIL: submission.py is not baseline + v2/gmm.py + v6/r14c.py + the two edits; rerun make_v6.py")
fns = lambda src: {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
base, v2, v3 = fns((DIR / "baseline.py").read_text()), fns((DIR / "scored/v2.8eb7a168b38743a1.py").read_text()), fns(sub)
for name in ("_experts", "_marlin_bank", "_gemm"):
    if base[name] != v3[name]:
        raise SystemExit(f"FAIL: {name} differs from baseline")
for name in ("_experts_gmm", "_dequant", "_gmm_prepare", "_nvfp4_dequant_kernel", "_gather_rows_kernel",
             "_gmm_act_kernel", "_gmm_combine_kernel", "_e2m1_value", "_gmm_selftest_routing"):
    if v2[name] != v3[name]:
        raise SystemExit(f"FAIL: {name} differs from the scored v2")
# R11's twin of _experts_gmm: the same statements minus the two _dequant calls.
twin = ast.parse((DIR / "submission.py").read_text())
fn = {n.name: n for n in twin.body if isinstance(n, ast.FunctionDef)}
g, r = fn["_experts_gmm"], fn["_experts_gmm_r11"]
deq = [s for s in g.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
       and getattr(s.value.func, "id", None) == "_dequant"]
assert len(deq) == 2
keep = [ast.dump(s) for s in g.body[1:] if s not in deq]  # body[0] is the docstring
if keep != [ast.dump(s) for s in r.body[1:]]:
    raise SystemExit("FAIL: _experts_gmm_r11 is not _experts_gmm minus its two _dequant calls")
print(f"ok: submission.py = baseline + v2 + v6 fragments + 3 edits (R14_CONFIGS = {configs}); Marlin and v2 paths untouched")

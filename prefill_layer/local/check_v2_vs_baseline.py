"""v2 must be exactly baseline.py + v2/gmm.py + make_v2.py's two edits (ASTs, docstrings stripped), and
the baseline's Marlin path must still be the code that runs below GMM_MIN_TOKENS.

Run: .venv/bin/python prefill_layer/local/check_v2_vs_baseline.py
"""

import ast
import importlib.util
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_v2", DIR / "local" / "make_v2.py")
make_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_v2)


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
if strip(ast.parse(sub)) != strip(ast.parse(make_v2.build_text())):
    raise SystemExit("FAIL: submission.py is not baseline.py + v2/gmm.py + the two intended edits; rerun make_v2.py")
base_fns = {n.name: ast.dump(n) for n in ast.parse((DIR / "baseline.py").read_text()).body if isinstance(n, ast.FunctionDef)}
sub_fns = {n.name: ast.dump(n) for n in ast.parse(sub).body if isinstance(n, ast.FunctionDef)}
for name in ("_experts", "_marlin_bank", "_gemm"):
    if base_fns[name] != sub_fns[name]:
        raise SystemExit(f"FAIL: {name} differs from baseline")
print("ok: submission.py = baseline + v2/gmm.py + 2 edits; Marlin path untouched")

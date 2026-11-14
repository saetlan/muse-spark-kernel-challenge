"""v1 must be baseline.py (SGLang's layer) apart from its one intended change, the Marlin tile height.

Compares the two files' ASTs with docstrings and comments stripped. The expected v1 is built from
baseline.py by applying exactly the intended edit:
  * a new top-level function moe_block_size(num_tokens) (checked separately: SGLang's rule on
    num_tokens * TOP_K / N_EXPERTS rows, giving the heights in EXPECTED_HEIGHT);
  * _experts gets an optional block_size_m=None argument, and SGLang's height loop becomes
    `if block_size_m is None: block_size_m = moe_block_size(num_tokens)`.
Every other statement, call and constant must be identical.

Run: .venv/bin/python prefill_layer/local/check_v1_vs_baseline.py [submission.py]
"""

import ast
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
# tokens -> height: the scored shapes (753, 3012, 16384) and the judge's tests (300, 753, 1506, 600, 4096).
EXPECTED_HEIGHT = {200: 8, 300: 16, 600: 32, 753: 32, 1506: 64, 3012: 64, 4096: 64, 16384: 64}
SGLANG_LOOP = """
for block_size_m in BLOCK_SIZE_M_CANDIDATES:
    if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
        break
"""
V1_BLOCK = """
if block_size_m is None:
    block_size_m = moe_block_size(num_tokens)
"""


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def dump(node):
    return ast.dump(node, include_attributes=False)


def main(sub_path):
    base = strip_docstrings(ast.parse((DIR / "baseline.py").read_text()))
    sub_src = Path(sub_path).read_text()
    if not sub_src.startswith("# okc: problem=muse_prefill_layer\n"):
        raise SystemExit("FAIL: first line must be '# okc: problem=muse_prefill_layer'")
    sub = strip_docstrings(ast.parse(sub_src))

    # Build the expected v1 from baseline.
    loop = dump(ast.parse(SGLANG_LOOP).body[0])
    new_block = ast.parse(V1_BLOCK).body
    expected, sub_fn = [], None
    for node in base.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_experts":
            hits = [i for i, st in enumerate(node.body) if dump(st) == loop]
            assert len(hits) == 1, "baseline _experts: SGLang's height loop not found exactly once"
            node.body[hits[0]:hits[0] + 1] = new_block
            node.args.args.append(ast.arg(arg="block_size_m"))
            node.args.defaults.append(ast.Constant(value=None))
            moe_fn = next(n for n in sub.body if isinstance(n, ast.FunctionDef) and n.name == "moe_block_size")
            expected.append(moe_fn)
        expected.append(node)
    exp_dumps = [dump(n) for n in expected]
    sub_dumps = [dump(n) for n in sub.body]
    if exp_dumps != sub_dumps:
        for i, (e, s) in enumerate(zip(exp_dumps, sub_dumps)):
            if e != s:
                raise SystemExit(f"FAIL: top-level statement {i} differs from baseline + intended edit:\n"
                                 f"  expected {ast.unparse(expected[i])[:300]}\n  got      {ast.unparse(sub.body[i])[:300]}")
        raise SystemExit(f"FAIL: {len(sub_dumps)} top-level statements, expected {len(exp_dumps)}")

    # moe_block_size's values.
    ns = {}
    consts = [n for n in sub.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("TOP_K", "N_EXPERTS", "LOCAL_EXPERTS", "BLOCK_SIZE_M_CANDIDATES")]
    moe_fn = next(n for n in sub.body if isinstance(n, ast.FunctionDef) and n.name == "moe_block_size")
    exec(compile(ast.Module(consts + [moe_fn], []), "moe_block_size", "exec"), ns)
    for tokens, height in EXPECTED_HEIGHT.items():
        got = ns["moe_block_size"](tokens)
        if got != height:
            raise SystemExit(f"FAIL: moe_block_size({tokens}) = {got}, expected {height}")
    print(f"ok: {sub_path} = baseline.py + the tile-height change; heights {EXPECTED_HEIGHT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DIR / "submission.py")

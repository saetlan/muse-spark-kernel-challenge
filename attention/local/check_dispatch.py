"""Checks the hybrid dispatch in submission.py:
1. sglang_path is baseline.py's kernel_fn exactly (AST equal after the helper renames), and so
   are its helpers, so falling back can never change numerics or timing vs SGLang;
2. kernel_fn routes each scored shape class to run() or sglang_path as OURS says, and the
   warm-up visits every scored class through the same route.
Run: .venv/bin/python attention/local/check_dispatch.py [attention/submission.py]"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from emulate import load_submission  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROB, "submission.py")
sub_tree = ast.parse(open(path).read())
base_tree = ast.parse(open(os.path.join(PROB, "baseline.py")).read())
RENAMES = {"_qk_norm": "_sg_qk_norm", "_cos_sin": "_sg_cos_sin", "_NORMS": "_SG_NORMS", "_ROPE": "_SG_ROPE",
           "ROPE_POSITIONS": "SG_ROPE_POSITIONS", "kernel_fn": "sglang_path"}


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAMES.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = RENAMES.get(node.name, node.name)
        node.body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
        self.generic_visit(node)
        return node


def funcs(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


base = {RENAMES.get(k, k): Rename().visit(v) for k, v in funcs(base_tree).items()}
sub = funcs(sub_tree)
for name in ("sglang_path", "_sg_qk_norm", "_sg_cos_sin"):
    b, s = ast.dump(base[name], annotate_fields=False), ast.dump(Rename().visit(sub[name]), annotate_fields=False)
    assert b == s, f"{name} differs from baseline.py"
print("ok: sglang_path and its helpers equal baseline.py's (AST, after renames)")

mod = load_submission(path)
calls = []
mod.run = lambda data, tune=None: calls.append(("ours", mod.shape_class(data["config"])))
mod.sglang_path = lambda data: calls.append(("sglang", mod.shape_class(data["config"])))
mod.torch.cuda.is_current_stream_capturing = lambda: False
mod._WARM.append(True)        # skip the warm-up (it needs a GPU); its routing is checked below
cuda_route = getattr(mod, "CUDA_ROUTE", {})
if cuda_route:                # v3: CUDA routes first, through _cuda_run; the rest as v2.2
    mod._CUDA["mod"] = object()
    mod._cuda_ok = lambda cfg, route: True
    mod._cuda_run = lambda data, route: calls.append(("cuda:" + route[0], mod.shape_class(data["config"])))
pdl_route = getattr(mod, "PDL_ROUTE", set())
if pdl_route:                 # v5: PDL Triton chain (late trigger) for its classes, after the CUDA routes
    mod.run_pdl = lambda data, tune=None, early=False: calls.append(("pdl-early" if early else "pdl",
                                                                       mod.shape_class(data["config"])))
for length in (1, 8):
    for window in (2048, 0):
        for batch in (1, 16, 32):
            cfg = dict(batch=batch, q_len=length, window=window, num_q_heads=16, num_kv_heads=2)
            calls.clear()
            mod.kernel_fn({"config": cfg})
            cls = mod.shape_class(cfg)
            want = (("cuda:" + cuda_route[cls][0]) if cls in cuda_route else "pdl" if cls in pdl_route
                    else "ours" if cls in mod.OURS else "sglang")
            assert calls == [(want, mod.shape_class(cfg))], (cfg, calls)
text = open(path).read()
warms = [ast.get_source_segment(text, n) for n in sub_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_warm_up"]
assert any("if shape_class(data[\"config\"]) in OURS" in w and "sglang_path(data)" in w for w in warms), \
    "the v2.2 warm-up must route every scored class as kernel_fn does"
if cuda_route:   # v3's warm-up: v2.2's first, then the CUDA build, then a launch of every CUDA route
    w3 = warms[-2] if pdl_route else warms[-1]
    assert "_warm_up_v22()" in w3 and "_cuda_build()" in w3 and (
        "for cls, route in CUDA_ROUTE.items()" in w3 or "for cls, route in sorted(CUDA_ROUTE.items())" in w3)
if pdl_route:    # v5's warm-up: v4's first, then every PDL class compiled and checked (late trigger)
    w5 = warms[-1]
    assert "_warm_up_v4()" in w5 and "for full, verify, bclass in sorted(PDL_ROUTE)" in w5 and "run_pdl(data, early=False)" in w5
    print("ok: PDL routes (late trigger) " + ", ".join(f"{'F' if f else 'S'}{'v' if v else 'd'}{b}" for f, v, b in sorted(pdl_route)))
if cuda_route:
    print("ok: CUDA routes " + ", ".join(f"{'F' if f else 'S'}{'v' if v else 'd'}{b}->{r[0]}" for (f, v, b), r in sorted(cuda_route.items())))
print(f"ok: dispatch routes {len(mod.OURS)} scored classes to our kernels, the rest to SGLang: "
      + ", ".join(f"{'F' if f else 'S'}{'v' if v else 'd'}{b}" for f, v, b in sorted(mod.OURS)))

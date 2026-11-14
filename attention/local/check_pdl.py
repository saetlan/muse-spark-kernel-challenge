"""Checks the generated PDL code (make_pdl.py) against the verified code it came from:
1. Triton (pdl_block.py): with the PDL lines removed (gdc_* calls, TRIGGER_EARLY branches and
   parameter, launch_pdl=True) and the names reverted, _prologue_p / _attend_p / _combine_p /
   run_pdl are AST-equal to submission.py's _prologue / _attend / _combine / run; the dependents'
   first statement is gdc_wait();
2. CUDA (cuda_dec3p.py, cuda_dec4p.py): in each kernel, nothing before `griddepcontrol.wait`
   touches what the producer (the Triton prologue) writes: no query load, no K/V cache access,
   no K/V copy (issue()), no global store; every host launch goes through launch_pdl.
Run: .venv/bin/python attention/local/check_pdl.py"""
import ast
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
bad = 0


class Strip(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = node.name[:-2] if node.name.endswith("_p") else node.name
        node.name = "run" if node.name == "run_pdl" else node.name
        node.args.args = [a for a in node.args.args if a.arg != "TRIGGER_EARLY"]
        node.args.args = [a for a in node.args.args if a.arg != "early"]
        node.args.defaults = [d for d in node.args.defaults if not (isinstance(d, ast.Constant) and d.value is False)]
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        t = ast.unparse(node.test)
        if t in ("TRIGGER_EARLY", "not TRIGGER_EARLY"):
            return None
        self.generic_visit(node)
        return node

    def visit_Expr(self, node):
        if "gdc_" in ast.unparse(node):
            return None
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg not in ("launch_pdl", "TRIGGER_EARLY")]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in ("_prologue_p", "_attend_p", "_combine_p"):
            node.id = node.id[:-2]
        return node


def funcs(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


sub = funcs(ast.parse(open(os.path.join(PROB, "submission.py")).read()))
pdl_tree = ast.parse(open(os.path.join(HERE, "pdl_block.py")).read())
pdl = funcs(pdl_tree)
for name in ("_attend_p", "_combine_p"):
    first = ast.unparse(pdl[name].body[0])
    if first != "tlcuda.gdc_wait()":
        print(f"FAIL: {name} starts with {first!r}, not gdc_wait()")
        bad += 1
for new, old in (("_prologue_p", "_prologue"), ("_attend_p", "_attend"), ("_combine_p", "_combine"), ("run_pdl", "run")):
    a = ast.dump(Strip().visit(pdl[new]), annotate_fields=False)
    b = ast.dump(sub[old], annotate_fields=False)
    same = a == b
    print(("ok: " if same else "FAIL: ") + f"{new} minus its PDL lines == submission.py {old}")
    bad += not same


def source(path, var):
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, var)


FORBIDDEN_BEFORE_WAIT = [r"\bqp\[", r"\bq \+", r"\bkc \+", r"\bvc \+", r"\bissue\(\d", r"\bissue\(i\)",
                         r"_part\[", r"\bout\["]
for path, var in (("cuda_dec3p.py", "CUDA_DEC3P_SRC"), ("cuda_dec4p.py", "CUDA_DEC4P_SRC")):
    src = source(path, var)
    kstart = src.index("__global__")
    body = src[kstart:]
    pre = body[: body.index("griddepcontrol.wait")]
    # lambda definitions only define; strip them before looking for uses
    pre_uses = re.sub(r"auto \w+ = \[&\]\([^)]*\)[^{]*\{.*?\n  \};", "", pre, flags=re.S)
    hits = [p for p in FORBIDDEN_BEFORE_WAIT if re.search(p, pre_uses)]
    launches = len(re.findall(r"<<<", src))
    pdl_launches = len(re.findall(r"launch_pdl\(okc_", src))
    ok = not hits and launches == 0 and pdl_launches >= 8
    print(("ok: " if ok else "FAIL: ") + f"{path}: before the wait {hits or 'no producer data'}; "
          f"<<<>>> launches {launches}, launch_pdl {pdl_launches}")
    bad += not ok
print("check_pdl:", "ok" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)

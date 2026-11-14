"""Embed the part of a sibling's scored file that some root names need, every top-level name prefixed.

Like make_attn.py (AST prefixing; CUDA source strings untouched; no exec at run time), but only the transitive
closure of top-level definitions the roots reference is kept, in the source's order (a later definition of the same
name wins, as in the source). Default: MoE's scored E19 (moe/scored/submission_m13t_e19.py, scored 1.4877x on
muse_moe_w4a16, job 9df74163eea14095) for its large-batch path at 128 / 256 tokens: M19 (M14 with the weight-code
ring in shared memory) for GEMM1 / GEMM2 and M13's CUDA post-norm, with M13's bank built from M11's.

Run: .venv/bin/python decode_layer/local/make_embed_subset.py [--check]                     (E19 -> parts/moe19.py)
     .venv/bin/python decode_layer/local/make_embed_subset.py --which e25 [--check]          (E25 -> parts/moe25.py)
     .venv/bin/python decode_layer/local/make_embed_subset.py --which e35 [--check]          (E35 -> parts/moe35.py)
     .venv/bin/python decode_layer/local/make_embed_subset.py --which e36 [--check]          (E36 -> parts/moe36.py)
     .venv/bin/python decode_layer/local/make_embed_subset.py --which dgrs [--check]         (bf16_proj v11b DG12-RS -> parts/dgrs.py)
"""

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "moe" / "scored" / "submission_m13t_e19.py"
OUT = ROOT / "decode_layer" / "parts" / "moe19.py"
PREFIX = "moe_"
ROOTS = ("m13_source", "m13_bank", "m13_scales_ok", "m13_run", "_M11_CPP", "_M13_CPP", "M13_CONFIGS", "M19_INSTANCES")
JUDGED = "scored 1.4877x on\n# muse_moe_w4a16, job 9df74163eea14095"
# MoE's scored E25 (1.6279x, job b5da76a6e7cb4267): E24's M11b (M11's kernel on M13's bank and cheaper decode) at
# 1-32 tokens + M19 at 65+; its m13_source also appends E21/E22's CR/SR kernels and G2P (M11F), whose instance lists
# are module globals read at call time, so a consumer can empty them.
WHICH = {
    "e25": (ROOT / "moe" / "scored" / "submission_m13t_e25.py", ROOT / "decode_layer" / "parts" / "moe25.py", "m25_",
            ("m13_source", "m13_bank", "m13_scales_ok", "m13_run", "_M11_CPP", "_M13_CPP", "_M11F_CPP", "M19_INSTANCES",
             "M11B_INSTANCES", "M13_M11_CR_INSTANCES", "M13_M11_SR_INSTANCES", "M11F_INSTANCES"),
            "scored 1.6279x on\n# muse_moe_w4a16, job b5da76a6e7cb4267"),
    # MoE's scored E35 (1.6285x, job 2916d19a0de141cc; best check E35 1.6448x): E26's hot/cold split (M11b cold +
    # M19 hot at a pair threshold; M11b gains a sel argument) and E35's late wait on M19's hot GEMM1 (pf bit 29).
    "e35": (ROOT / "moe" / "scored" / "submission_m13t_e35.py", ROOT / "decode_layer" / "parts" / "moe35.py", "m35_",
            ("m13_source", "m13_bank", "m13_scales_ok", "m13_run", "_M11_CPP", "_M13_CPP", "_M11F_CPP", "M19_INSTANCES",
             "M11B_INSTANCES", "M13_M11_CR_INSTANCES", "M13_M11_SR_INSTANCES", "M11F_INSTANCES", "M11B_OCC_INSTANCES",
             "M11B_EARLY_INSTANCES", "M11B_PF_INSTANCES", "M11B_NU_INSTANCES", "M11B_RECON_SHAPES"),
            "scored 1.6285x on\n# muse_moe_w4a16, job 2916d19a0de141cc"),
    # MoE's scored E36 (1.6520x, job 73e71876d57147a7): E35 + GEMM2's late wait at bs256 (cold GEMM2 signals its
    # dependents only after its own wait: M19 pf bit 15 / M11b sel bit 18; hot GEMM2 waits at its end: pf bit 29).
    "e36": (ROOT / "moe" / "scored" / "submission_m13t_e36.py", ROOT / "decode_layer" / "parts" / "moe36.py", "m36_",
            ("m13_source", "m13_bank", "m13_scales_ok", "m13_run", "_M11_CPP", "_M13_CPP", "_M11F_CPP", "M19_INSTANCES",
             "M11B_INSTANCES", "M13_M11_CR_INSTANCES", "M13_M11_SR_INSTANCES", "M11F_INSTANCES", "M11B_OCC_INSTANCES",
             "M11B_EARLY_INSTANCES", "M11B_PF_INSTANCES", "M11B_NU_INSTANCES", "M11B_RECON_SHAPES"),
            "scored 1.6520x on\n# muse_moe_w4a16, job 73e71876d57147a7"),
    # MoE's scored E49 (1.6958x, job 6edb977e1d414efc): E36 + the dynamic-deal M11b (dyn_kernel, per-role counters),
    # hot-first GEMM1 / GEMM2 and split threshold 8.
    "e49": (ROOT / "moe" / "scored" / "submission_m13t_e49.py", ROOT / "decode_layer" / "parts" / "moe49.py", "m49_",
            ("m13_source", "m13_bank", "m13_scales_ok", "m13_run", "_M11_CPP", "_M13_CPP", "_M11F_CPP", "M19_INSTANCES",
             "M11B_INSTANCES", "M13_M11_CR_INSTANCES", "M13_M11_SR_INSTANCES", "M11F_INSTANCES", "M11B_OCC_INSTANCES",
             "M11B_EARLY_INSTANCES", "M11B_PF_INSTANCES", "M11B_NU_INSTANCES", "M11B_RECON_SHAPES", "M11B_DYN_INSTANCES"),
            "scored 1.6958x on\n# muse_moe_w4a16, job 6edb977e1d414efc"),
    # bf16_proj's scored v11b (1.3511x, job ee5b3e6584ef4da6): DG12-RS (12-bit lossless weights decoded into register-A
    # wgmma, cluster / global split-K; kernel by muse_prefill_layer, pinned proj/dg/pinned/dgrs_v10.py) for the pooler
    # and unpooler at m = 32, with the F12M natural-order plane encoder it reads.
    "dgrs": (ROOT / "proj" / "scored" / "submission_v11b.py", ROOT / "decode_layer" / "parts" / "dgrs.py", "pb_",
             ("dgrs_build", "dgrs_pack", "dgrs_buffers", "dgrs_run", "dgrs_inst", "dgrs_key", "f12m_device_planes",
              "DGRS_CHOICE"),
             "scored 1.3511x on\n# muse_bf16_proj, job ee5b3e6584ef4da6"),
}


def defined(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return {node.name}
    names = set()
    if isinstance(node, ast.Assign):
        for t in node.targets:
            names |= {e.id for e in ast.walk(t) if isinstance(e, ast.Name)}
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.add(node.target.id)
    return names


def used(node):
    return {e.id for e in ast.walk(node) if isinstance(e, ast.Name)}


class Prefix(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = PREFIX + node.id
        return node

    def _def(self, node):
        if node.name in self.names:
            node.name = PREFIX + node.name
        outer = self.names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # a parameter is local inside its function
            a = node.args
            self.names = outer - {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
            for d in a.defaults + a.kw_defaults:  # defaults are evaluated in the enclosing scope
                if d is not None:
                    Prefix(outer).visit(d)
            for stmt in node.body:
                self.visit(stmt)
            for dec in node.decorator_list:
                Prefix(outer).visit(dec)
            self.names = outer
            return node
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def

    def visit_Global(self, node):
        node.names = [PREFIX + n if n in self.names else n for n in node.names]
        return node


def main(check=False, which=None):
    global SRC, OUT, PREFIX, ROOTS, JUDGED
    if which:
        SRC, OUT, PREFIX, ROOTS, JUDGED = WHICH[which]
    src = SRC.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    tree = ast.parse(src)
    body = tree.body
    top = {}
    for i, node in enumerate(body):
        for name in defined(node):
            top.setdefault(name, []).append(i)
    keep, todo = set(), list(ROOTS)
    while todo:
        name = todo.pop()
        for i in top.get(name, ()):
            if i not in keep:
                keep.add(i)
                todo.extend(n for n in used(body[i]) if n in top)
    imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
    kept = [body[i] for i in sorted(keep)]
    names = set().union(*(defined(n) for n in kept))
    shadowed = sorted({(node.name, n) for node in ast.walk(ast.Module(body=kept, type_ignores=[]))
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       for n in {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs} & names})
    # A parameter that shadows a kept name stays local inside its function (Prefix keeps it); a local *assignment*
    # that shadows one would not, so refuse those.
    for node in ast.walk(ast.Module(body=kept, type_ignores=[])):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)) and sub is not node:
                    tg = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for t in tg:
                        bad = {e.id for e in ast.walk(t) if isinstance(e, ast.Name)} & names
                        assert not bad, f"{node.name}: a local assignment shadows a kept top-level name: {bad}"
    module = ast.Module(body=kept, type_ignores=[])
    module = Prefix(names).visit(module)
    ast.fix_missing_locations(module)
    text = ast.unparse(module)
    imp = "\n".join(ast.unparse(n) for n in imports)
    header = (f"# Embedded subset: a sibling's scored file ({SRC.relative_to(ROOT)}, sha256 {sha}...; {JUDGED}): the\n"
              f"# {len(kept)} top-level definitions that {', '.join(ROOTS)}\n"
              f"# need, every one prefixed with '{PREFIX}' by decode_layer/local/make_embed_subset.py (AST; CUDA strings\n"
              f"# untouched). Its imports (below, commented) are the decode file's own. Do not edit by hand.\n"
              + "".join(f"#   {ln}\n" for ln in imp.splitlines()) + "\n")
    if check:
        print(f"{len(kept)} definitions kept of {len(body)}; {len(text)} chars; parameters kept local: {shadowed}")
        return
    OUT.write_text(header + text + "\n")
    print(f"wrote {OUT} ({len(text)} chars, {len(kept)} definitions, {len(names)} names prefixed, source sha {sha})")


if __name__ == "__main__":
    args = sys.argv[1:]
    main("--check" in args, args[args.index("--which") + 1] if "--which" in args else None)

"""Embed a sibling session's scored file in the decode file, with every top-level name prefixed.

Default: the attention session's scored v3 (attention/scored/v3_submission.py) as decode_layer/parts/attn.py:
every top-level name it defines (functions, classes, globals) gets the prefix "att_", everywhere it is
referenced, so nothing collides with the decode file's own names; imports keep their names (they are
the same modules and objects). Done on the AST, so CUDA source strings are untouched; no exec at run
time. The attention problem computes exactly our layer's attention segment (qkv -> QK-norm, RoPE, FP8
cache store, attention, rmsnorm * sigmoid(gate)), so att_kernel_fn(data) is a drop-in.

Also: --proj embeds the bf16_proj session's scored v2 (proj/scored/submission_v2.py) as parts/proj.py
with prefix "pj_" (its CHOICE table picks, per (layer, m), the kernel measured fastest on the judge).

Run: .venv/bin/python decode_layer/local/make_attn.py [--proj | --which attn5 | --which proj4]
"""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "attention" / "scored" / "v3_submission.py"
OUT = ROOT / "decode_layer" / "parts" / "attn.py"
PREFIX = "att_"


def top_level_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for e in ast.walk(t):
                    if isinstance(e, ast.Name):
                        names.add(e.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
    return names


class Prefix(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names
        self.globals_tests = 0

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = PREFIX + node.id
        return node

    def _def(self, node):
        if node.name in self.names:
            node.name = PREFIX + node.name
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def

    def visit_Global(self, node):
        node.names = [PREFIX + n if n in self.names else n for n in node.names]
        return node

    def visit_Compare(self, node):
        """`"NAME" in globals()` (attention v16+ registers kinds that way): the string names a top-level name, so it
        gets the prefix too, or the test would be False in the embed and the kind never registered."""
        self.generic_visit(node)
        if is_globals_test(node) and node.left.value in self.names:
            node.left = ast.Constant(PREFIX + node.left.value)
            self.globals_tests += 1
        return node


def is_globals_test(node):
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
            and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and isinstance(node.comparators[0], ast.Call) and isinstance(node.comparators[0].func, ast.Name)
            and node.comparators[0].func.id == "globals")


EMBEDS = {
    "attn": (ROOT / "attention" / "scored" / "v3_submission.py", ROOT / "decode_layer" / "parts" / "attn.py", "att_",
             "the attention session's scored v3", "judged 1.281x on muse_attention_decode, AE12",
             "att_kernel_fn(data) computes our layer's attention segment."),
    "proj": (ROOT / "proj" / "scored" / "submission_v2.py", ROOT / "decode_layer" / "parts" / "proj.py", "pj_",
             "the bf16_proj session's scored v2", "scored 1.192x on muse_bf16_proj, job aae716920c1d403f",
             "pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m)."),
    "attn5": (ROOT / "attention" / "scored" / "v5_submission.py", ROOT / "decode_layer" / "parts" / "attn5.py", "att_",
              "the attention session's scored v5", "scored 1.406x on muse_attention_decode, job 47db0b38ebaf49ec",
              "att_kernel_fn(data) computes our layer's attention segment (D3 on full decode 16/32, PDL chain)."),
    "proj4": (ROOT / "proj" / "scored" / "submission_v4.py", ROOT / "decode_layer" / "parts" / "proj4.py", "pj_",
              "the bf16_proj session's scored v4", "scored 1.247x on muse_bf16_proj, job 0d1dc429eebf455f",
              "pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m); "
              "P9 needs pj_p9_setup(state)."),
    "attn6": (ROOT / "attention" / "scored" / "v6_submission.py", ROOT / "decode_layer" / "parts" / "attn6.py", "att_",
              "the attention session's scored v6", "scored 1.459x on muse_attention_decode, job f9b4a77bb7b846f1",
              "att_kernel_fn(data) computes our layer's attention segment (v5 + UD3 sliding decode 1, D3 whole-wave "
              "full decode 1/16, V4 FP8-query verify full 1)."),
    "proj6": (ROOT / "proj" / "scored" / "submission_v6.py", ROOT / "decode_layer" / "parts" / "proj6.py", "pj_",
              "the bf16_proj session's scored v6", "scored 1.318x on muse_bf16_proj, job 33748bdba9694bcb",
              "pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m); "
              "P9 needs pj_p9_setup(state), the 12-bit lossless kernels pj_f12v_setup / pj_f12m_setup(state)."),
    "attn8": (ROOT / "attention" / "scored" / "v8_submission.py", ROOT / "decode_layer" / "parts" / "attn8.py", "att_",
              "the attention session's scored v8", "submitted Nov 4 15:41 UTC on muse_attention_decode; checks 1.471x / "
              "1.472x / 1.481x (L19 73941e40, L20 23a9335a)",
              "att_kernel_fn(data) computes our layer's attention segment (v6 + UD3 (8,4,3) sliding decode 16, V5 (1,4,1) "
              "full verify 1, FA3 with num_splits 4 + pack_gqa behind a Triton FP8 prologue on full verify 16)."),
    "attn9": (ROOT / "attention" / "scored" / "v9_submission.py", ROOT / "decode_layer" / "parts" / "attn9.py", "att_",
              "the attention session's scored v9", "scored 1.5655x on muse_attention_decode, job 2e994b0659da4d84",
              "att_kernel_fn(data) computes our layer's attention segment (v8 + W1-N, wgmma FP8 verify, on sliding "
              "verify 16 / 32 and full verify 1 / 16 / 32; fa3t and ver5 routes gone; decode unchanged)."),
    "attn11": (ROOT / "attention" / "scored" / "v11_submission.py", ROOT / "decode_layer" / "parts" / "attn11.py", "att_",
               "the attention session's scored v11", "scored 1.6102x on muse_attention_decode, job c98e26a4329c4fab",
               "att_kernel_fn(data) computes our layer's attention segment (v9's W1-N on full verify 16 / 32 and sliding "
               "verify 32; W2 on full verify 1; UW2, one launch per sliding verify layer, on sliding verify 1 / 16)."),
    "attn14": (ROOT / "attention" / "scored" / "v14_submission.py", ROOT / "decode_layer" / "parts" / "attn14.py", "att_",
               "the attention session's scored v14", "scored 1.722x on muse_attention_decode, job 01bc09ba1a3e437b",
               "att_kernel_fn(data) computes our layer's attention segment (v11 + W3b, 4 consumer warpgroups and the "
               "whole-row copier, on full verify 16 / 32; UD4, row-owner finalize, on sliding decode 1 / 16 / 32). The CUDA "
               "sources of kinds v14 no longer routes are emptied (TRIM below; the judge's 1,000,000-byte limit, F61)."),
    "attn17": (ROOT / "attention" / "scored" / "v17_submission.py", ROOT / "decode_layer" / "parts" / "attn17.py", "att_",
               "the attention session's scored v17", "scored 1.7973x on muse_attention_decode, job f033cf3eca5144e1",
               "att_kernel_fn(data) computes our layer's attention segment (v15's routes with UD8 on sliding decode 1 / "
               "16 / 32, UW2h (FP16 pushes) on sliding verify 1 / 16, the multi-row combine at <= 4 splits). Unrouted "
               "kinds' CUDA sources emptied (TRIM); its `\"CUDA_X_SRC\" in globals()` registrations prefixed."),
    "attn18": (ROOT / "attention" / "scored" / "v18_submission.py", ROOT / "decode_layer" / "parts" / "attn18.py", "att_",
               "the attention session's scored v18", "scored 1.8031x on muse_attention_decode, job 0b3cd2ffe25f497e",
               "att_kernel_fn(data) computes our layer's attention segment (v17's routes with UD9 on sliding decode 32 and "
               "D3p, D3 as a PDL dependent of the Triton prologue with a PDL combine, on full decode 16 / 32). Unrouted "
               "kinds' CUDA sources emptied (TRIM); its `\"CUDA_X_SRC\" in globals()` registrations prefixed."),
    "attn15": (ROOT / "attention" / "submission_v15.py", ROOT / "decode_layer" / "parts" / "attn15.py", "att_",
               "the attention session's v15 (its scored run pending at embed time; check L49 1.7688x passed)",
               "md5 808d3fe6; v14 + UD5 on sliding decode, D3 at 1 CTA per SM on full decode 1",
               "att_kernel_fn(data) computes our layer's attention segment (v14's routes, with UD5 = UD4 + padded warp "
               "partials, cache_loc at entry, 16-byte pushes on sliding decode 1 / 16 / 32, cs 4 at batch 32). Unrouted "
               "kinds' CUDA sources emptied (TRIM)."),
}

# Kinds whose CUDA source string is emptied in the embed: v14's _cuda_build compiles only the kinds its final
# CUDA_ROUTE names (dec4, ud4, uw2, w1n, w2, w3b), and a route dropped at warm-up falls back to the Triton / PDL
# chain, never to another CUDA kind. Each name must be a string literal referenced exactly once (its _KINDS entry);
# parts/v20.py asserts at import that every routed kind still has its source.
TRIM = {
    "attn14": ("CUDA_DEC2_SRC", "CUDA_DEC3_SRC", "CUDA_UD_SRC", "CUDA_VER2_SRC", "CUDA_VER4_SRC", "CUDA_UD3_SRC",
               "CUDA_UV4_SRC", "CUDA_VER5_SRC", "CUDA_VER5T_SRC", "CUDA_W3_SRC"),
    "attn15": ("CUDA_DEC2_SRC", "CUDA_DEC3_SRC", "CUDA_UD_SRC", "CUDA_VER2_SRC", "CUDA_VER4_SRC", "CUDA_UD3_SRC",
               "CUDA_UV4_SRC", "CUDA_VER5_SRC", "CUDA_VER5T_SRC", "CUDA_W3_SRC", "CUDA_UD4_SRC", "CUDA_UD4C_SRC",
               "CUDA_UW2V_SRC", "CUDA_UD6_SRC", "CUDA_UW2G_SRC"),
    "attn18": ("CUDA_DEC2_SRC", "CUDA_DEC3_SRC", "CUDA_UD_SRC", "CUDA_VER2_SRC", "CUDA_VER4_SRC", "CUDA_UD3_SRC",
               "CUDA_UV4_SRC", "CUDA_VER5_SRC", "CUDA_VER5T_SRC", "CUDA_W3_SRC", "CUDA_UD4_SRC", "CUDA_UD5_SRC",
               "CUDA_UD4C_SRC", "CUDA_UW2V_SRC", "CUDA_UD6_SRC", "CUDA_UW2G_SRC", "CUDA_UW2_SRC"),
    "attn17": ("CUDA_DEC2_SRC", "CUDA_DEC3_SRC", "CUDA_UD_SRC", "CUDA_VER2_SRC", "CUDA_VER4_SRC", "CUDA_UD3_SRC",
               "CUDA_UV4_SRC", "CUDA_VER5_SRC", "CUDA_VER5T_SRC", "CUDA_W3_SRC", "CUDA_UD4_SRC", "CUDA_UD5_SRC",
               "CUDA_UD4C_SRC", "CUDA_UW2V_SRC", "CUDA_UD6_SRC", "CUDA_UW2G_SRC", "CUDA_UW2_SRC"),
}


def trim(tree, names):
    """Replace each named top-level string literal with "" (checked: defined once, read once)."""
    loads = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
            loads[node.id] = loads.get(node.id, 0) + 1
    done = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names):
            name = node.targets[0].id
            assert name not in done, f"{name} assigned twice"
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), name
            assert loads.get(name) == 1, f"{name} read {loads.get(name)} times (expected once, in _KINDS)"
            node.value = ast.Constant("")
            done.add(name)
    assert done == set(names), f"not found: {set(names) - done}"
    return tree


def main(which="attn"):
    global SRC, OUT, PREFIX
    SRC, OUT, PREFIX, what, judged, use = EMBEDS[which]
    src = SRC.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    tree = ast.parse(src)
    names = top_level_names(tree)
    # parameters / locals that shadow a top-level name would be renamed too: refuse if any exist
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
            clash = params & names
            assert not clash, f"{node.name}: parameter shadows a top-level name: {clash}"
    if which in TRIM:
        tree = trim(tree, TRIM[which])
    pre = Prefix(names)
    tree = pre.visit(tree)
    ast.fix_missing_locations(tree)
    # every remaining `"X" in globals()` must name something not defined at top level (its test stays as in the source)
    left = [n.left.value for n in ast.walk(tree) if is_globals_test(n) and not n.left.value.startswith(PREFIX)]
    assert not set(left) & names, f"unprefixed globals() tests on top-level names: {set(left) & names}"
    if pre.globals_tests or left:
        print(f"globals() tests: {pre.globals_tests} prefixed; left as in the source (names it never defines): {left}")
    body = ast.unparse(tree)
    assert not body.startswith("# okc:")
    header = (f"# Embedded: {what} ({SRC.relative_to(ROOT)}, sha256 {sha}...;\n"
              f"# {judged}), every top-level name prefixed with '{PREFIX}' by\n"
              f"# decode_layer/local/make_attn.py. {use}\n\n")
    OUT.write_text(header + body + "\n")
    print(f"wrote {OUT} ({len(body)} chars, {len(names)} names prefixed, source sha {sha})")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args[args.index("--which") + 1] if "--which" in args else ("proj" if "--proj" in args else "attn"))

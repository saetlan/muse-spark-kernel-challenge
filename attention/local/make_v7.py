"""Builds a v7 file: v6's parts (make_v6.py: submission.py + CUDA sources + v4/pdl/v5 blocks +
cuda_ver4.py + cuda_ud3.py + V6_ROUTES + v6_block.py) + cuda_uv4.py + `V7_ROUTES = {...}` + v7_block.py.
Usage: .venv/bin/python attention/local/make_v7.py <output name> <routes dict literal> [lab]
  routes: {class: route} for every CUDA class beyond v5's defaults (ver4 / dec4 "w" / ud3 routes go to
  V6_ROUTES, uv4 routes to V7_ROUTES). "lab" adds a UV4 guess on each sliding verify class that has
  no uv4 route and a UD3 guess on Sd16 / Sd32 if unrouted (so the kinds are built and self-checked;
  the L17 lab times and re-routes them)."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
UV4_GUESS = {(False, True, 1): ("uv4", 16, 3, 1), (False, True, 16): ("uv4", 8, 3, 1),
             (False, True, 32): ("uv4", 4, 4, 1)}
UD3_GUESS = {(False, False, 16): ("ud3", 8, 4, 3), (False, False, 32): ("ud3", 8, 4, 3)}
UV4_VARIANTS = {(3, 1, 4), (3, 1, 8), (3, 1, 16), (4, 1, 4), (4, 1, 8), (3, 2, 4), (3, 2, 8), (3, 2, 16),
                (6, 1, 4), (6, 1, 8), (8, 1, 4), (8, 1, 8)}
out_name = sys.argv[1]
routes = ast.literal_eval(sys.argv[2])
if "lab" in sys.argv[3:]:
    for c, r in UV4_GUESS.items():
        if routes.get(c, ("",))[0] != "uv4":
            routes[c] = r
    for c, r in UD3_GUESS.items():   # re-checked with disjoint rows (v7's self-check inputs)
        routes.setdefault(c, r)
v6r = {c: r for c, r in routes.items() if r[0] != "uv4"}
v7r = {c: r for c, r in routes.items() if r[0] == "uv4"}
assert all((r[0] == "ver4" and c[1] and len(r) == 5) or (r[0] == "dec4" and r[4:] == ("w",) and not c[1])
           or (r[0] == "ud3" and len(r) == 4 and not c[0] and not c[1]) for c, r in v6r.items()), v6r
# uv4: verify classes, ("uv4", cs, stages, mt[, ncl]) with a compiled variant (stages, mt, cs)
assert all(c[1] and len(r) in (4, 5) and (r[2], r[3], r[1]) in UV4_VARIANTS and (len(r) == 4 or r[4] >= 1)
           for c, r in v7r.items()), v7r
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_dec3.py", "cuda_dec4.py", "cuda_unified.py", "cuda_ver2.py", "v4_block.py",
             "pdl_block.py", "v5_block.py", "cuda_ver4.py", "cuda_ud3.py", "cuda_uv4.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())


def table(name, rs):
    lines = ",\n".join(f"    {c!r}: {r!r}" for c, r in sorted(rs.items()))
    return f"\n\n# ===== {name.lower()} (make_v7.py) =====\n{name} = {{\n" + lines + ("\n" if lines else "") + "}\n"


parts.append(table("V6_ROUTES", v6r))
parts.append("\n\n# ===== appended: attention/local/v6_block.py =====\n" + open(os.path.join(HERE, "v6_block.py")).read())
parts.append(table("V7_ROUTES", v7r))
parts.append("\n\n# ===== appended: attention/local/v7_block.py =====\n" + open(os.path.join(HERE, "v7_block.py")).read())
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes), V6_ROUTES: {v6r}, V7_ROUTES: {v7r}")

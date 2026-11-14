"""Builds a v8 file: v7's parts (make_v7.py) + cuda_ver5.py + `V8_ROUTES = {...}` + v8_block.py +
`FA3T_ROUTE.update({...})` via fa3t_block.py.
Usage: .venv/bin/python attention/local/make_v8.py <output name> <routes dict literal> [lab]
  routes: {class: route} for every class beyond v5's defaults: ver4 / dec4 "w" / ud3 -> V6_ROUTES,
  uv4 -> V7_ROUTES, ver5 / ver5t -> V8_ROUTES, ("fa3t", num_splits, pack_gqa) -> FA3T_ROUTE.
  "lab" adds a V5 guess on each verify class without a route (so the kind is built and
  self-checked; the L18 lab times and re-routes them)."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
VER5_GUESS = {(True, True, 16): ("ver5", 4, 3, 2), (True, True, 32): ("ver5", 4, 3, 2),
              (False, True, 16): ("ver5", 1, 3, 2), (False, True, 32): ("ver5", 1, 3, 2)}
VER5_VARIANTS = {(3, 1), (4, 1), (3, 2), (4, 2)}
UV4_VARIANTS = {(3, 1, 4), (3, 1, 8), (3, 1, 16), (4, 1, 4), (4, 1, 8), (3, 2, 4), (3, 2, 8), (3, 2, 16),
                (6, 1, 4), (6, 1, 8), (8, 1, 4), (8, 1, 8)}
out_name = sys.argv[1]
routes = ast.literal_eval(sys.argv[2])
if "lab" in sys.argv[3:]:
    for c, r in VER5_GUESS.items():
        routes.setdefault(c, r)
v6r = {c: r for c, r in routes.items() if r[0] in ("ver4", "dec4", "ud3")}
v7r = {c: r for c, r in routes.items() if r[0] == "uv4"}
v8r = {c: r for c, r in routes.items() if r[0] in ("ver5", "ver5t")}
fa3 = {c: r[1:] for c, r in routes.items() if r[0] == "fa3t"}
assert len(v6r) + len(v7r) + len(v8r) + len(fa3) == len(routes), routes
assert all((r[0] == "ver4" and c[1] and len(r) == 5) or (r[0] == "dec4" and r[4:] == ("w",) and not c[1])
           or (r[0] == "ud3" and len(r) == 4 and not c[0] and not c[1]) for c, r in v6r.items()), v6r
assert all(c[1] and len(r) in (4, 5) and (r[2], r[3], r[1]) in UV4_VARIANTS for c, r in v7r.items()), v7r
assert all(c[1] and len(r) == 4 and (r[2], r[3]) in VER5_VARIANTS for c, r in v8r.items()), v8r
assert all(c[1] and len(r) == 2 and isinstance(r[0], int) and r[1] in (None, True, False) for c, r in fa3.items()), fa3
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_dec3.py", "cuda_dec4.py", "cuda_unified.py", "cuda_ver2.py", "v4_block.py",
             "pdl_block.py", "v5_block.py", "cuda_ver4.py", "cuda_ud3.py", "cuda_uv4.py", "cuda_ver5.py", "cuda_ver5t.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())


def table(name, rs):
    lines = ",\n".join(f"    {c!r}: {r!r}" for c, r in sorted(rs.items()))
    return f"\n\n# ===== {name.lower()} (make_v8.py) =====\n{name} = {{\n" + lines + ("\n" if lines else "") + "}\n"


def block(name):
    return f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read()


parts += [table("V6_ROUTES", v6r), block("v6_block.py"), table("V7_ROUTES", v7r), block("v7_block.py"),
          table("V8_ROUTES", v8r), block("v8_block.py"), block("fa3t_block.py"),
          "\n\n# ===== fa3t routes (make_v8.py) =====\nFA3T_ROUTE.update(" + repr(dict(sorted(fa3.items()))) + ")\n"]
out = os.path.join(PROB, out_name) if not os.path.isabs(out_name) else out_name
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes), V6: {v6r}, V7: {v7r}, V8: {v8r}, FA3T: {fa3}")

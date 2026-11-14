"""Builds a v6 file: v5's parts (submission.py + CUDA sources + v4_block.py + pdl_block.py +
v5_block.py, as make_v5.py) + cuda_ver4.py + cuda_ud3.py + `V6_ROUTES = {...}` + v6_block.py.
Usage: .venv/bin/python attention/local/make_v6.py <output name> [routes]
  routes: "lab" (every verify class on a V4 guess, so the kind is built and self-checked; the L16
  lab replaces them with measured winners) or a Python dict literal {class: route} (the measured
  routes, for a scored file)."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
LAB_GUESS = {
    (True, True, 1): ("ver4", 2, 3, 2, 1), (True, True, 16): ("ver4", 4, 3, 2, 1),
    (True, True, 32): ("ver4", 4, 3, 2, 1), (False, True, 1): ("ver4", 1, 3, 2, 1),
    (False, True, 16): ("ver4", 2, 3, 2, 1), (False, True, 32): ("ver4", 2, 3, 2, 1),
    (False, False, 1): ("ud3", 8, 4, 3), (False, False, 16): ("ud3", 8, 4, 3), (False, False, 32): ("ud3", 8, 4, 3),
}
out_name = sys.argv[1]
spec = sys.argv[2] if len(sys.argv) > 2 else "{}"
routes = LAB_GUESS if spec == "lab" else ast.literal_eval(spec)
assert all((r[0] == "ver4" and c[1] and len(r) == 5) or (r[0] == "dec4" and r[4:] == ("w",) and not c[1])
           or (r[0] == "ud3" and len(r) == 4 and not c[0] and not c[1]) for c, r in routes.items()), routes
parts = [open(os.path.join(PROB, "submission.py")).read()]
for name in ("cuda_dec2.py", "cuda_dec3.py", "cuda_dec4.py", "cuda_unified.py", "cuda_ver2.py", "v4_block.py",
             "pdl_block.py", "v5_block.py", "cuda_ver4.py", "cuda_ud3.py"):
    parts.append(f"\n\n# ===== appended: attention/local/{name} =====\n" + open(os.path.join(HERE, name)).read())
lines = ",\n".join(f"    {c!r}: {r!r}" for c, r in sorted(routes.items()))
parts.append("\n\n# ===== v6 routes (make_v6.py) =====\nV6_ROUTES = {\n" + lines + ("\n" if lines else "") + "}\n")
parts.append("\n\n# ===== appended: attention/local/v6_block.py =====\n" + open(os.path.join(HERE, "v6_block.py")).read())
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out} ({sum(map(len, parts))} bytes), routes: {routes}")

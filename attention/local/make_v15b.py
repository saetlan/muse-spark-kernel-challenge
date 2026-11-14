"""Builds a v15 file: submission_v14.py (byte for byte) + cuda_ud5.py / cuda_ud4c.py (only the routed variants) +
cuda_uw2v.py (when routed) + `V15_ROUTES` + v15b_block.py.
Usage: .venv/bin/python attention/local/make_v15b.py <output> <routes: {class: route}>
Route kinds: ('dec4', ...) as v14, ('ud5' | 'ud4c', cs, warps, stages), ('uw2v', stages, pp, cs)."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_ud4c  # noqa: E402
import make_ud5  # noqa: E402
import make_ud6  # noqa: E402
import make_ud7  # noqa: E402
import make_ud8  # noqa: E402
import make_ud9  # noqa: E402

out_name, routes = sys.argv[1], ast.literal_eval(sys.argv[2])
UW2V = {(4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)}
ud5 = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud5"})
ud4c = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud4c"})
ud6 = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud6"})
ud7 = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud7"})
ud8 = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud8"})
ud9 = sorted({(r[2], r[3], r[1]) for r in routes.values() if r[0] == "ud9"})
UW2G = {(4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16), (6, 0, 8), (6, 1, 8), (6, 1, 16)}
for c, r in routes.items():
    assert r[0] in ("dec4", "ud5", "ud4c", "uw2v", "ud6", "uw2g", "ud7", "uw2h", "ud8", "ud9"), r
    if r[0] == "ud9":
        assert not c[0] and not c[1] and (r[2], r[3], r[1]) in make_ud9.VARIANTS, (c, r)
    if r[0] == "ud8":
        assert not c[0] and not c[1] and (r[2], r[3], r[1]) in make_ud8.VARIANTS, (c, r)
    if r[0] == "uw2h":
        assert not c[0] and c[1] and tuple(r[1:]) in UW2V, (c, r)
    if r[0] == "ud7":
        assert not c[1] and (r[2], r[3], r[1]) in make_ud7.VARIANTS and r[4] >= 1, (c, r)
    if r[0] == "ud6":
        assert not c[1] and (r[2], r[3], r[1]) in make_ud6.VARIANTS and r[4] >= 1, (c, r)
    if r[0] == "uw2g":
        assert c[1] and tuple(r[1:4]) in UW2G and r[4] >= 1, (c, r)
    if r[0] in ("ud5", "ud4c"):
        assert not c[0] and not c[1] and (r[2], r[3], r[1]) in make_ud4c.VARIANTS, (c, r)
    if r[0] == "uw2v":
        assert not c[0] and c[1] and tuple(r[1:]) in UW2V, (c, r)
    if r[0] == "dec4":
        assert c[0] and not c[1], (c, r)
# a kind with no route still needs its names defined (v15b_block registers all three): one small variant
src = open(os.path.join(PROB, "submission_v14.py")).read()
parts = [src,
         "\n\n# ===== appended: cuda_ud5.py (make_ud5.py, variants " + repr(ud5 or [(4, 3, 16)]) + ") =====\n" + make_ud5.render(ud5 or [(4, 3, 16)]),
         "\n\n# ===== appended: cuda_ud4c.py (make_ud4c.py, variants " + repr(ud4c or [(4, 3, 8)]) + ") =====\n" + make_ud4c.render(ud4c or [(4, 3, 8)]),
         "\n\n# ===== appended: attention/local/cuda_uw2v.py =====\n" + open(os.path.join(HERE, "cuda_uw2v.py")).read(),
         "\n\n# ===== appended: cuda_ud6.py (make_ud6.py, variants " + repr(ud6 or [(4, 3, 16)]) + ") =====\n" + make_ud6.render(ud6 or [(4, 3, 16)]),
         "\n\n# ===== appended: attention/local/cuda_uw2g.py =====\n" + open(os.path.join(HERE, "cuda_uw2g.py")).read(),
         *(["\n\n# ===== appended: attention/local/cuda_uw2h.py =====\n" + open(os.path.join(HERE, "cuda_uw2h.py")).read()]
           if any(r[0] == "uw2h" for r in routes.values()) else []),
         *(["\n\n# ===== appended: cuda_ud9.py (make_ud9.py, variants " + repr(ud9) + ") =====\n" + make_ud9.render(ud9)] if ud9 else []),
         *(["\n\n# ===== appended: cuda_ud8.py (make_ud8.py, variants " + repr(ud8) + ") =====\n" + make_ud8.render(ud8)] if ud8 else []),
         *(["\n\n# ===== appended: cuda_ud7.py (make_ud7.py, variants " + repr(ud7) + ") =====\n" + make_ud7.render(ud7)] if ud7 else []),
         "\n\n# ===== v15 routes (make_v15b.py) =====\nV15_ROUTES = " + repr(dict(sorted(routes.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v15b_block.py =====\n" + open(os.path.join(HERE, "v15b_block.py")).read()]
if "--d3p" in sys.argv:   # v18: D3p (PDL chain) routes, given as the argument after --d3p
    d3p = ast.literal_eval(sys.argv[sys.argv.index("--d3p") + 1])
    assert all(c[0] and not c[1] and r[0] == "dec4p" for c, r in d3p.items()), d3p
    parts.append("\n\n# ===== appended: attention/local/cuda_dec4p.py =====\n" + open(os.path.join(HERE, "cuda_dec4p.py")).read())
    parts.append("\n\n# ===== v18 D3p routes (make_v15b.py) =====\nV18_D3P_ROUTES = " + repr(dict(sorted(d3p.items()))) + "\n")
    parts.append("\n\n# ===== appended: attention/local/v18_d3p_block.py =====\n" + open(os.path.join(HERE, "v18_d3p_block.py")).read())
if "--combine-r" in sys.argv:
    parts.append("\n\n# ===== appended: attention/local/v16_combine_block.py =====\n" + open(os.path.join(HERE, "v16_combine_block.py")).read())
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V15 {routes}")

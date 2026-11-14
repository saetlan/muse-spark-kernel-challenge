"""Mutation test of check_v3_cpu.py: plants bugs in v3/r3.py one at a time, rebuilds submission.py, expects
each to fail the check, then restores the clean files.
Run from the project root: .venv/bin/python prefill_layer/local/mutate_v3_checks.py"""
import subprocess

G = "prefill_layer/v3/r3.py"
clean = open(G).read()
MUTS = {
    "tile row step": ("(p - t_start[e_c]) * bt", "(p - t_start[e_c]) * (bt - 1)"),
    "searchsorted left": ("torch.searchsorted(t_end, p, right=True)", "torch.searchsorted(t_end, p, right=False)"),
    "c1 uses w2 scale": ('"c1": (gmm["w13_gs"] * R3_OUT_SHIFT)', '"c1": (gmm["w2_gs"] * R3_OUT_SHIFT)'),
    "GEMM2 wrong BK bank": ('r3["w2"][cfg2[3]]', 'r3["w2"][cfg1[3]]'),
    "a_hi offset": ("a_hi = tl.load(a_tile + (k0 + HB), mask=a_mask, other=0.0)",
                    "a_hi = tl.load(a_tile + (k0 + HB + 1), mask=a_mask, other=0.0)"),
    "repack nibbles swapped": ("(t[..., 0, :] | (t[..., 1, :] << 4))", "(t[..., 1, :] | (t[..., 0, :] << 4))"),
    "no spare tiles": ("return -(-slots // bt) + LOCAL_EXPERTS", "return -(-slots // bt)"),
    "invalid tiles keep expert": ("expert = torch.where(e < LOCAL_EXPERTS, e, torch.full_like(e, -1))", "expert = e_c"),
    "epilogue no up scale": ("res = g * tl.sigmoid(g) * (acc_u * c_e)", "res = g * tl.sigmoid(g) * acc_u"),
}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) == 1, name
        open(G, "w").write(clean.replace(a, b))
        subprocess.run([".venv/bin/python", "prefill_layer/local/make_v3.py"], check=True, capture_output=True)
        r = subprocess.run([".venv/bin/python", "prefill_layer/local/check_v3_cpu.py"], capture_output=True, text=True)
        results[name] = "CAUGHT" if r.returncode else "MISSED"
finally:
    open(G, "w").write(clean)
    subprocess.run([".venv/bin/python", "prefill_layer/local/make_v3.py"], check=True, capture_output=True)
for k, v in results.items():
    print(f"{v:7s} {k}")

"""Mutation test of the G2P checks: plants one bug at a time in make_g2p.py, regenerates g2p.py (and g2p_moe.py),
runs emu_g2p.py and check_g2p_syntax.py, expects a failure, then restores the clean generator and outputs.

Run: .venv/bin/python shared/glue/mutate_g2p.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
GEN = HERE / "make_g2p.py"
clean = GEN.read_text()
MUTS = {
    "expert finish off by one": ("fin = atomicAdd(cnt + e, 1) == per_expert - 1;", "fin = atomicAdd(cnt + e, 1) == per_expert;"),
    "no expert reset": ("            atomicExch(cnt + e, 0);  // self-reset", "            (void)0;  // self-reset"),
    "token local count": ("            nl += (e2 >= 0 && e2 < LOCAL) ? 1 : 0;", "            nl += 1;"),
    "token finish off by one": ("if (atomicAdd(cnt + LOCAL + tk, 1) == nl - 1) {", "if (atomicAdd(cnt + LOCAL + tk, 1) == nl) {"),
    "no token reset": ("            atomicExch(cnt + LOCAL + tk, 0);\n", "\n"),
    "zero rows for local tokens": ("    if (!any)\n", "    if (any)\n"),
    "zero rows skipped": ("        reinterpret_cast<uint4*>(pout + static_cast<size_t>(zt) * K)[i] = make_uint4(0u, 0u, 0u, 0u);",
                          "        (void)i;"),
    "weight without rsqrt": ("f[k] = loc[k] ? tw[tk * TOP_K + k] * s_rs[grp][k] : 0.0f;", "f[k] = loc[k] ? tw[tk * TOP_K + k] : 0.0f;"),
    "mean over 1": ("rsqrtf(s / static_cast<float>(K) + eps)", "rsqrtf(s + eps)"),
    "column map": ("    const int col = tig + c * NT_;", "    const int col = tig * PER + c;"),
    "y row of slot 0": ("y + static_cast<size_t>(tk * TOP_K + k) * K) + col);", "y + static_cast<size_t>(tk * TOP_K) * K) + col);"),
    "no release fence": ("        __threadfence();  // release: this lane's y stores before the counter\n", "\n"),
    "no acquire fence (expert)": ("            __threadfence();  // acquire: every other task's y stores of expert e\n", "\n"),
    "no acquire fence (token)": ("            __threadfence();  // acquire: the other experts' y rows of token tk\n", "\n"),
    "no release fence (token)": ("          __threadfence();  // release by this thread: expert e's y rows", "          (void)0;  // release by this thread: expert e's y rows"),
    "group size": ("        for (int i = tig; i < total; i += KS * 32) {", "        for (int i = tig; i < total; i += 32) {"),
    "nw ignored in sum of squares": ("const float v0 = __uint_as_float(u << 16) * __uint_as_float(wu << 16);",
                                     "const float v0 = __uint_as_float(u << 16);"),
    "C++ typo": ("  static_assert(!GATED, \"G2P: GEMM2 only\");", "  static_assert(!GATED, \"G2P: GEMM2 only\")"),
}
outs = [HERE / "g2p.py", HERE / "g2p_moe.py"]
saved = {p: p.read_text() for p in outs if p.exists()}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) >= 1, name
        GEN.write_text(clean.replace(a, b))
        r = subprocess.run([PY, str(GEN)], capture_output=True, text=True)
        caught = r.returncode != 0
        for chk in ("emu_g2p.py", "check_g2p_syntax.py"):
            if caught:
                break
            r = subprocess.run([PY, str(HERE / chk)], capture_output=True, text=True)
            caught = r.returncode != 0
        results[name] = "CAUGHT" if caught else "MISSED"
finally:
    GEN.write_text(clean)
    for p, t in saved.items():
        p.write_text(t)
for k, v in results.items():
    print(f"{v:7s} {k}")
sys.exit(1 if "MISSED" in results.values() else 0)

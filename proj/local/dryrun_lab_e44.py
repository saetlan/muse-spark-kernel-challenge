"""CPU dry run of proj/lab_e44.py: v11b's modules mocked as in check_v11b.py; the six okc_diag_* modules mocked by checking
their source (namespace renamed, e1 / e2 edits applied exactly where intended, e0 untouched) and delegating each renamed
function to v11b's own mocked F12M / F12V function. Every F12M / F12V shape in v11b's CHOICE must print a diag line with
e0 output == base; the timed path must stay v11b's (judge's check on every shape).

Run: .venv/bin/python proj/local/dryrun_lab_e44.py
"""

import io
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_make_dgrs as cmd  # noqa: E402
import check_v11b  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import make_e44  # noqa: E402

HOLD = {}


def mock_diag(seen):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        mo = re.fullmatch(r"okc_diag_(f12m|f12v)(e[012])", name)
        if not mo:
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        fam, tag = mo.groups()
        ns = fam + tag
        assert f"namespace {ns} {{" in cuda_sources and f"namespace {fam} {{" not in cuda_sources and f" {fam}::" not in cuda_sources
        esc_off = "if (j < count && false) {" in cuda_sources
        assert esc_off == (tag != "e0") and ("if (j < count) {" in cuda_sources) == (tag == "e0"), (name, esc_off)
        one = "sum = part[0][row][tok];" if fam == "f12m" else "sum = red[0][tid];"
        assert (one in cuda_sources) == (tag == "e2"), name
        assert all(f.startswith(f"{ns}_run_") for f in functions), functions
        seen.add(name)
        mod = types.SimpleNamespace()
        for f in functions:
            base_name = f.replace(f"{ns}_run_", f"{fam}_run_")

            def fn(*a, base_name=base_name):
                return getattr(HOLD["lab"].F12M_MOD if fam == "f12m" else HOLD["lab"].F12V_MOD, base_name)(*a)
            setattr(mod, f, fn)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    make_e44.build()
    cmd.setup()
    check_v11b.mock_v10({})
    seen = set()
    mock_diag(seen)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e44.py", "lab_e44_cpu")
    HOLD["lab"] = lab
    lab.Lab.time_graph = lambda self, fn, reps=20, cold=True: (fn(), 5.0)[1]
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    assert len(seen) == 6, seen
    shapes = [k for k, c in lab.CHOICE.items() if c.get("kind") in ("f12m", "f12v")]
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] diag ") and "base" in ln and "e0 output" in ln]
    assert len(lines) == len(shapes) and all("e0 output == base: True" in ln for ln in lines), (shapes, lines, out[-2000:])
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        data = reference.generate_input(m, layer, 6600 + m, device="cpu")
        ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
        assert ok, (layer, m, msg)
    print(f"ok: lab_e44.py dry run; 6 diag modules with the intended edits; {len(lines)} shapes "
          f"({sorted(shapes)}) timed with e0 == base; timed path = v11b (judge's check on {len(lab.CHOICE)} shapes)")


if __name__ == "__main__":
    main()

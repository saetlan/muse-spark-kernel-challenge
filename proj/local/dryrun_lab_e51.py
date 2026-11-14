"""CPU dry run of proj/lab_e51.py: v11b's modules mocked as in check_v11b.py; okc_proj_f12mp mocked by checking its source
(only the part declaration padded, namespace and function names renamed, the same instances as v11b's F12M) and delegating
each function to v11b's own mocked F12M function. Every F12M shape of v11b gets an "mp" line with bit-identical outputs and
moves to f12mp (fake timer: padded faster), with v11b's entry as fallback; the judge's check passes on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e51.py
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_make_dgrs as cmd  # noqa: E402
import check_v11b  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import make_e51  # noqa: E402

HOLD = {}


def mock_mp(seen):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_f12vx":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "g * 32 + lane] = xv[slot][g];" in cuda_sources and "f12v_xpos(static_cast<int>(esc[j] & 0xFFFFu))" in cuda_sources
        assert "namespace f12vx {" in cuda_sources and "namespace f12v {" not in cuda_sources and " f12v::" not in cuda_sources
        assert all(f.startswith("f12vx_run_") for f in functions)
        seen["functions"] = list(functions)
        mod = types.SimpleNamespace()
        for f in functions:
            base_name = f.replace("f12vx_run_", "f12v_run_")
            setattr(mod, f, lambda *a, base_name=base_name: getattr(HOLD["lab"].F12V_MOD, base_name)(*a))
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    make_e51.build()
    cmd.setup()
    check_v11b.mock_v10({})
    import check_v13
    check_v13.mock_mh({})
    seen = {}
    mock_mp(seen)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e51.py", "lab_e51_cpu")
    HOLD["lab"] = lab
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 9.0 if c is not None and c["kind"] == "f12vx" else 10.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    names = sorted(lab.f12v_name(i).replace("f12v_run_", "f12vx_run_") for i in lab.F12V_INSTANCES)
    assert sorted(seen["functions"]) == sorted(names + [n + "_info" for n in names]), seen
    f12m_shapes = sorted(k for k, c in base_choice.items() if c.get("kind") == "f12v")
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] vx ") and "bit-identical: True" in ln]
    assert len(lines) == len(f12m_shapes), (lines, out[-2000:])
    moved = sorted(k for k, c in lab.CHOICE.items() if c.get("kind") == "f12vx")
    assert moved == f12m_shapes, (moved, f12m_shapes)
    for k in moved:
        c = lab.CHOICE[k]
        assert c["fallback"] == base_choice[k] and {a: b for a, b in c.items() if a not in ("kind", "fallback")} == \
            {a: b for a, b in base_choice[k].items() if a not in ("kind", "fallback")}, (k, c)
    assert out.count("VX CHOICE") == len(moved)
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 7800 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e51.py dry run; f12vx source = F12V with conflict-free x staging; {len(moved)} F12M shapes {moved} race with "
          f"bit-identical outputs and move with fallbacks; judge's check on {len(lab.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main()

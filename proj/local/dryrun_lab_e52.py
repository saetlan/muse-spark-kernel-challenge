"""CPU dry run of proj/lab_e52.py (base v14): v14's modules mocked (check_v11b + check_v13's helper mock); okc_proj_p9l mocked by
checking its source (both PE52 edits, namespace p9l, P9_INSTANCES) and delegating each function to v14's own mocked P9 function
(same math). Fake timer: p9l faster, so router m8 and m16 move with their v14 entries as fallbacks; the judge's check passes on
every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e52.py
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_make_dgrs as cmd  # noqa: E402
import check_v11b  # noqa: E402
import check_v13  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import make_e52  # noqa: E402

HOLD = {}


def mock_pl(seen):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_p9l":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "PE52: each CTA sums its own K parts in place first" in cuda_sources
        assert "for (int kp = 0; kp < (CS > 1 ? 1 : KSW); ++kp) {" in cuda_sources
        assert "namespace p9l {" in cuda_sources and "namespace p9 {" not in cuda_sources and " p9::" not in cuda_sources
        seen["functions"] = list(functions)
        mod = types.SimpleNamespace()
        for f in functions:
            base_name = f.replace("p9l_run_", "p9_run_")
            setattr(mod, f, lambda *a, base_name=base_name: getattr(HOLD["lab"].P9_MOD, base_name)(*a))
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    make_e52.build()
    cmd.setup()
    check_v11b.mock_v10({})
    check_v13.mock_mh({})
    seen = {}
    mock_pl(seen)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e52.py", "lab_e52_cpu")
    HOLD["lab"] = lab
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 9.0 if c is not None and c["kind"] == "p9l" else 10.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    names = [lab.p9_name(i).replace("p9_run_", "p9l_run_") for i in lab.P9_INSTANCES]
    assert sorted(seen["functions"]) == sorted(names + [n + "_attr" for n in names] + [n + "_info" for n in names]), seen
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] pl router") and "two-phase" in ln]
    assert len(lines) == 2 and all(" 1 ok, 0 wrong" in ln for ln in lines), (lines, out[-2000:])
    for key in (("router", 8), ("router", 16)):
        c = lab.CHOICE[key]
        assert c["kind"] == "p9l" and c["fallback"] == base_choice[key], (key, c)
    assert out.count("PL CHOICE") == 2
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 7300 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e52.py dry run; p9l source has both edits; router m8 / m16 race and move with fallbacks; judge's check on "
          f"{len(lab.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main()

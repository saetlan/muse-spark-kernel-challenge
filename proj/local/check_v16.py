"""CPU check of proj/submission_v16.py (v15 + two-phase P9 on router m8 / m16): v15's modules mocked (check_v11b + check_v13's
helper mock); okc_proj_p9l mocked by checking its source (both PE52 edits, namespace p9l, exactly P9_INSTANCES) and delegating to
v15's mocked P9 functions. prepare(); CHOICE = v15's except router m8 / m16 (kind p9l, v15's entries as fallbacks); the judge's
check on every shape x 2 inputs; router m8 / m16 really call the p9l module.

Run: .venv/bin/python proj/local/check_v16.py
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

HOLD = {}


def mock_pl(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_p9l":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "PE52: each CTA sums its own K parts in place first" in cuda_sources
        assert "for (int kp = 0; kp < (CS > 1 ? 1 : KSW); ++kp) {" in cuda_sources and "namespace p9l {" in cuda_sources
        calls["functions"] = list(functions)
        mod = types.SimpleNamespace()
        for f in functions:
            base_name = f.replace("p9l_run_", "p9_run_")

            def fn(*a, f=f, base_name=base_name):
                calls[f] = calls.get(f, 0) + 1
                return getattr(HOLD["mod"].P9_MOD, base_name)(*a)
            setattr(mod, f, fn)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main(path):
    cmd.setup()
    check_v11b.mock_v10({})
    check_v13.mock_mh({})
    calls = {}
    mock_pl(calls)
    w = d.weights()
    base = d.load(d.PROJ / "submission_v15.py", "v15_ref_cpu")
    mod = d.load(Path(path), "v16_cpu")
    HOLD["mod"] = mod
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    assert mod.P9L_MOD is not None
    names = [mod.p9_name(i).replace("p9_run_", "p9l_run_") for i in mod.P9_INSTANCES]
    assert sorted(calls["functions"]) == sorted(names + [n + "_attr" for n in names] + [n + "_info" for n in names])
    for key, c in mod.CHOICE.items():
        if key in (("router", 8), ("router", 16)):
            assert c["kind"] == "p9l" and c["fallback"] == base.CHOICE.get(key), (key, c)
        else:
            assert c == base.CHOICE[key], key
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(mod.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 7600 + m, dist=dist, device="cpu")
            before = sum(v for k, v in calls.items() if k != "functions" and not k.endswith("_attr"))
            ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
            ran = sum(v for k, v in calls.items() if k != "functions" and not k.endswith("_attr")) > before
            assert ran == ((layer, m) in (("router", 8), ("router", 16))), (layer, m, ran)
    print(f"ok: {Path(path).name} CPU check; two-phase P9 on router m8 / m16 (fallbacks v15's), the rest = v15; judge's check "
          f"on all {len(mod.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(d.PROJ / "submission_v16.py"))

"""CPU dry run of proj/lab_e58.py (base v16): v16's modules mocked (check_v11b + check_v13's helper + check_v16's p9l mock);
okc_proj_p9l2 mocked from its source (two-phase edits, namespace p9l2, exactly PL2_INSTANCES), each function computing x @ (hi +
lo)^T in FP64 from the router's P9 hi / lo fragments (FP32 out). Fake timer: p9l2 faster, so router m32 moves with v16's entry as
fallback; the judge's check passes on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e58.py
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
import check_v13  # noqa: E402
import check_v16  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e42 as dr  # noqa: E402
import make_e58  # noqa: E402


def mock_pl2(seen):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_p9l2":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "PE52: each CTA sums its own K parts in place first" in cuda_sources and "namespace p9l2 {" in cuda_sources
        seen["functions"] = list(functions)
        mod = types.SimpleNamespace()
        for f in functions:
            if f.endswith("_attr"):
                setattr(mod, f, lambda smem: None)
                continue
            if f.endswith("_info"):
                setattr(mod, f, lambda: 120 * 1000000)
                continue
            nt = int(f[len("p9l2_run_"):].split("_")[0])

            def run(wp, qp, rp, tops, erow, ecol, ebits, x, out, m, n, k, smem, nt=nt):
                assert 1 <= m <= 8 * nt and out.dtype == torch.float32 and out.shape == (m, n)
                if id(wp) not in cache:
                    cache[id(wp)] = dr.unpack_hilo(wp, k)
                out.copy_((x.double() @ cache[id(wp)][:n].T).float())
            setattr(mod, f, run)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    make_e58.build()
    cmd.setup()
    check_v11b.mock_v10({})
    check_v13.mock_mh({})
    check_v16.mock_pl({})
    seen = {}
    mock_pl2(seen)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e58.py", "lab_e58_cpu")
    check_v16.HOLD["mod"] = lab
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 9.0 if c is not None and c["kind"] == "p9l2" else 10.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    names = [lab.p9_name(i).replace("p9_run_", "p9l2_run_") for i in lab.PL2_INSTANCES]
    assert sorted(seen["functions"]) == sorted(names + [n + "_attr" for n in names] + [n + "_info" for n in names]), seen
    c = lab.CHOICE[("router", 32)]
    assert c["kind"] == "p9l2" and c["fallback"] == base_choice[("router", 32)], c
    assert "PL2 CHOICE router m32" in out and " 0 wrong)" in out, out[-1500:]
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 7450 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e58.py dry run; {len(lab.PL2_INSTANCES)} two-phase P9 instances at NT 4; router m32 moves with v16's entry as "
          f"fallback; judge's check on {len(lab.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main()

"""CPU dry run of proj/lab_e6.py (v3 + P9 router-only section + lab/sweep.py + lab/mb.py + lab/e6.py):
FlashInfer/DeepGEMM mocked exactly (v3's prepare calls them), the P9 module mocked by unpacking its
packed weights (dryrun_lab_e4.mock_p9), the microbenchmark modules mocked (cluster reduction exact).
Checks: the router sweep moves every router shape to P9 under a timer that favors it, the V4 CHOICE
lines are printed, the microbenchmark sections run, every final choice is correct.

Run: .venv/bin/python proj/local/dryrun_lab_e6.py   (after make_lab.py has built proj/lab_e6.py)
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e3  # noqa: E402
import dryrun_lab_e4  # noqa: E402


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: torch.nn.functional.linear(x, w)
    dryrun_lab_e3.mock_modules({}, planted=False)
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    calls = {}
    dryrun_lab_e4.mock_p9(calls, k_perm, planted=False)
    p9_load = sys.modules["torch.utils.cpp_extension"].load_inline

    def mb_load(name, functions, **kw):
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn == "mb_run_cluster":
                def run_cluster(src, out, ctas, c, n_per):
                    out.copy_(src.view(ctas // c, c, n_per).sum(1).flatten())
                mod.mb_run_cluster = run_cluster
            else:
                setattr(mod, fn, lambda *a, fn=fn: calls.__setitem__(fn, calls.get(fn, 0) + 1))
        return mod

    def load_inline(name, **kw):
        return p9_load(name=name, **kw) if name == "okc_proj_p9" else mb_load(name, kw["functions"])
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=132, L2_cache_size=60 << 20)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e6.py", "lab_e6_cpu")
    lab.MB_SIZES = (("8.4MB", 1024 * 4096 * 2),)
    v3_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 5.0 if c is not None and c["kind"] == "p9" else 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    text = buf.getvalue()
    new = {k: c for k, c in lab.CHOICE.items() if v3_choice.get(k) != c}
    assert set(new) == {("router", m) for m in (1, 8, 16, 32)}, sorted(new)
    assert text.count("V4 CHOICE router") == 4, text[-2000:]
    assert calls.get("mb_run_ldg", 0) > 5 and calls.get("mb_run_bulk", 0) > 0 and "mb cluster 8" in text, calls
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: lab_e6 dry run; router moved to P9 at m 1/8/16/32 with V4 CHOICE lines printed; microbenchmarks ran "
          f"({calls.get('mb_run_ldg')} ldg, {calls.get('mb_run_bulk')} bulk calls)")


if __name__ == "__main__":
    main()

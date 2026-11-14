"""CPU dry run of proj/lab_e13.py (v7a + WG section with LAB_DEEP + lab/sweep.py + lab/e10.py + lab/e11.py +
lab/e13.py). WG mocked by x @ W.T (kernel math checked by check_wg.py); planted: ST = 5 instances drop
their last 64 k (fastest by the fake timer; the gate must reject them); every WG shape moves to an ST 6
config; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e13.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: torch.nn.functional.linear(x, w)
    import dryrun_lab_e3
    import dryrun_lab_e4
    import dryrun_lab_e8
    import dryrun_lab_e9
    dryrun_lab_e3.mock_modules({}, planted=False)
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    dryrun_lab_e4.mock_p9({}, k_perm, planted=False)
    dryrun_lab_e8.mock_f12v({}, planted=False)
    dryrun_lab_e9.mock_f12m({}, planted=False)
    calls = {}
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_wg":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda: None)
            elif fn.endswith("_info"):
                setattr(mod, fn, lambda: 90 * 1000000)
            else:
                ks, st, tt = (int(v) for v in fn.split("_")[2:])

                def run(w, x, out, n_rows, st=st, tt=tt, fn=fn):
                    calls[fn] = calls.get(fn, 0) + 1
                    assert x.shape[0] <= 128 * tt and out.shape == (x.shape[0], n_rows)
                    xd = x.double()
                    if st == 5:
                        xd = xd.clone()
                        xd[:, -64:] = 0
                    out.copy_((xd @ w[:n_rows].double().T).to(torch.bfloat16))
                setattr(mod, fn, run)
        return mod
    cpp = types.ModuleType("torch.utils.cpp_extension")
    cpp.load_inline = load_inline
    sys.modules["torch.utils.cpp_extension"] = cpp
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e13.py", "lab_e13_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "wg" or any(c is v for v in base_choice.values()):
            return 20.0
        return 1.0 if c["ST"] == 5 else 5.0 + c["KS"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(lab.WG_LAB_CANDIDATES), sorted(set(new) ^ set(lab.WG_LAB_CANDIDATES))
    assert all(c["kind"] == "wg" and c["ST"] == 6 and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_wg":
            raise RuntimeError("Error building extension 'okc_proj_wg' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.wg_setup(state)
    cpp.load_inline = good
    for (layer, m) in new:
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": lab.lab_n(layer, m), "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:lab.lab_n(layer, m)]), layer)
    print(f"ok: lab_e13.py dry run; planted ST=5 bug rejected, {len(new)} WG shapes moved to ST 6 with fallbacks; failed build falls back")


if __name__ == "__main__":
    main()

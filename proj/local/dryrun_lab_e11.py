"""CPU dry run of proj/lab_e11.py (v5 + WG section + lab/sweep.py + lab/e10.py + lab/e11.py). The WG
module is mocked by x @ W.T (its kernel math is checked by check_wg.py); planted: WG instances with
KS = 8 drop their last 64 k (fastest by the fake timer, must be rejected). Triton (router) is emulated,
P9/F12V mocked exactly. Every WG shape with a correct candidate moves; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e11.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def mock_wg(calls, planted=True):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_wg":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "wgmma.mma_async.sync.aligned.m64n128k16" in cuda_sources and "@@" not in cuda_sources
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda: None)
                continue
            if fn.endswith("_info"):
                setattr(mod, fn, lambda: 120 * 1000000)
                continue
            ks, st, tt = (int(v) for v in fn.split("_")[2:])

            def run(w, x, out, n_rows, ks=ks, tt=tt, fn=fn):
                calls[fn] = calls.get(fn, 0) + 1
                m, k = x.shape
                assert w.dtype == x.dtype == out.dtype == torch.bfloat16 and w.is_contiguous() and x.is_contiguous()
                assert n_rows % 128 == 0 and k % 64 == 0 and 1 <= m <= 128 * tt and out.shape == (m, n_rows) and w.shape[1] == k
                xd = x.double()
                if planted and ks == 8:
                    xd = xd.clone()
                    xd[:, -64:] = 0
                out.copy_((xd @ w[:n_rows].double().T).to(torch.bfloat16))
            setattr(mod, fn, run)
        return mod
    cpp = types.ModuleType("torch.utils.cpp_extension")
    cpp.load_inline = load_inline
    sys.modules["torch.utils.cpp_extension"] = cpp


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
    mock_wg(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e11.py", "lab_e11_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or any(c is v for v in base_choice.values()):
            return 20.0
        if c["kind"] == "wg":
            return 1.0 if c["KS"] == 8 else 5.0 + c["ST"] / 100.0
        if c["kind"] == "gemm":
            return 6.0
        return 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    real_cands = lab.router_wide_cands
    lab.router_wide_cands = lambda m: real_cands(m)[:1]
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {k for k, cands in lab.WG_LAB_CANDIDATES.items() if any(c["KS"] != 8 for c in cands)}
    want |= {("router", 32), ("router", 128), ("router", 256)}
    assert set(new) == want, sorted(set(new) ^ want)
    for key, c in new.items():
        if c["kind"] == "wg":
            assert c["KS"] != 8 and c["fallback"] == base_choice.get(key), (key, c)
    assert any(n.startswith("wg_run_8_") for n in calls), "no planted instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    # failed nvcc: no module, every WG shape falls back
    cpp = sys.modules["torch.utils.cpp_extension"]
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_wg":
            raise RuntimeError("Error building extension 'okc_proj_wg' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.wg_setup(state)
    cpp.load_inline = good
    assert lab.WG_MOD is None
    for (layer, m), c in new.items():
        if c["kind"] != "wg":
            continue
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": lab.lab_n(layer, m), "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:lab.lab_n(layer, m)]), layer), (layer, m)
    print(f"ok: lab_e11.py dry run; planted KS=8 bug rejected, {len(new)} shapes moved (WG m128/256, router wide), "
          f"a failed build falls back")


if __name__ == "__main__":
    main()

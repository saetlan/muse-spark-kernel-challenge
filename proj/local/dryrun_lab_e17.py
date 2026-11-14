"""CPU dry run of proj/lab_e17.py (v7a's choices with the CS-capable F12M + 22 split-K instances + lab/sweep.py +
lab/e17.py). The F12M module is mocked (dryrun_lab_e9.mock_f12m: decodes the planes, x @ W.T in FP64 -> BF16;
the kernel's math, CS included, is checked by check_f12m.py). Planted: CS = 8 instances drop their last 64 k
(fastest by the fake timer; the gate must reject them); every shape with a correct split-K candidate moves to
it with the base kernel as fallback; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e17.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: __import__("torch").nn.functional.linear(x, w)
    import dryrun_lab_e3
    import dryrun_lab_e4
    import dryrun_lab_e8
    import dryrun_lab_e9
    dryrun_lab_e3.mock_modules({}, planted=False)
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    dryrun_lab_e4.mock_p9({}, k_perm, planted=False)
    dryrun_lab_e8.mock_f12v({}, planted=False)
    calls = {}
    dryrun_lab_e9.mock_f12m(calls, planted="cs8")
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e17.py", "lab_e17_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "f12m" or c.get("CS", 1) == 1:
            return 20.0
        return 1.0 if c["CS"] == 8 else 5.0 + c["D"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {key for key, cands in lab.F12M_CS_CANDIDATES.items() if any(c["CS"] != 8 for c in cands)}
    assert set(new) == want, (sorted(new), sorted(want))
    assert all(c["kind"] == "f12m" and c["CS"] in (2, 4) and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert any(fn.endswith("_c8") for fn in calls), "no planted CS = 8 instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    good = sys.modules["torch.utils.cpp_extension"].load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_f12m":
            raise RuntimeError("Error building extension 'okc_proj_f12m' (planted)")
        return good(name, *a, **k)
    sys.modules["torch.utils.cpp_extension"].load_inline = failing
    lab.f12m_setup(state)
    sys.modules["torch.utils.cpp_extension"].load_inline = good
    assert lab.F12M_MOD is None and state["f12m"] == {}
    for (layer, m) in new:
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m)
    print(f"ok: lab_e17.py dry run; planted CS=8 bug rejected, {len(new)} shapes moved to split-K with fallbacks "
          f"({', '.join(f'{k[0]} m{k[1]}' for k in sorted(new, key=lambda k: (k[1], k[0])))}); failed build falls back")


if __name__ == "__main__":
    main()

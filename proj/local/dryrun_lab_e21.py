"""CPU dry run of proj/lab_e21.py (v10's kernels with the TG-capable F12M + 19 token-group instances + lab/sweep.py +
lab/e21.py). F12M mocked (dryrun_lab_e9.mock_f12m decodes the planes; the kernel's token groups are checked by
check_f12m.py), router hi/lo Triton kernels emulated (dryrun_lab_e18). Planted: TG = 4 instances drop their last 64 k
(fastest by the fake timer); the gate must reject them; every shape with a correct TG candidate moves.

Run: .venv/bin/python proj/local/dryrun_lab_e21.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402


def main():
    d.install_stubs()
    e18.PLANT["on"] = False
    d.EMULATE["_hl_gemm_kernel"] = e18._hl_gemm
    d.EMULATE["_hl_swap_kernel"] = e18._hl_swap
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
    calls = {}
    dryrun_lab_e9.mock_f12m(calls, planted="tg4")
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e21.py", "lab_e21_cpu")
    base_choice = dict(lab.CHOICE)
    assert {k for k, c in base_choice.items() if c["kind"].startswith("hl_")} == {("router", m) for m in (16, 32, 128, 256)}

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or any(c is v for v in base_choice.values()) or c["kind"] != "f12m" or c.get("TG", 1) == 1:
            return 20.0
        return 1.0 if c["TG"] == 4 else 5.0 + c["D"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {key for key, cands in lab.F12M_TG_CANDIDATES.items() if any(c["TG"] != 4 for c in cands)}
    assert set(new) == want, (sorted(new), sorted(want))
    assert all(c["kind"] == "f12m" and c["TG"] == 2 and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert any(fn.endswith("_t4") for fn in calls), "no planted TG = 4 instance ran"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m), c in sorted(lab.CHOICE.items()):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 500 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, c, msg)
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
        data = reference.generate_input(m, layer, 700 + m, device="cpu")
        ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
        assert ok, (layer, m, "fallback", msg)
    print(f"ok: lab_e21.py dry run; planted TG=4 bug rejected, {len(new)} shapes moved to TG 2 "
          f"({', '.join(f'{k[0]} m{k[1]}' for k in sorted(new, key=lambda k: (k[1], k[0])))}); judge's check passes; failed build falls back")


if __name__ == "__main__":
    main()

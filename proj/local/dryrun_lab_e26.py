"""CPU dry run of proj/lab_e26.py (v10 + DG12 section + lab/sweep.py + lab/e24.py). Each per-instance module okc_proj_dg12_*
is mocked: it rebuilds W from the planes the lab passes (parsed CUDA decode, escape list ent8/offs8, column scale) and
computes x @ W.T in FP64 -> BF16, with the C++ wrapper's checks; base modules and Triton as in dryrun_lab_e22. Planted:
the qkv m128 instance (BN 48) drops its last 64 k (fastest by the fake timer: the gate must reject it) and the o m32
instance fails to build. Every chosen shape must pass the judge's check.

Run: .venv/bin/python proj/local/dryrun_lab_e26.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402
import dryrun_lab_e24 as e24  # noqa: E402
import check_dg12  # noqa: E402


mock_dg12 = e24.mock_dg12


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
    dryrun_lab_e9.mock_f12m({}, planted=False)
    calls = {}
    mock_dg12(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e26.py", "lab_e26_cpu")
    lab.dg12_include_dirs = lambda: ["/fake/deep_gemm/include"]
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dg12":
            return 20.0
        return 1.0 if c["BN"] == 48 else 5.0 + c["BN"] / 1000.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = set(lab.DG12_TARGETS) - {("qkv_gate", 128), ("o", 32)}
    assert set(new) == want, (sorted(new), sorted(want))
    assert all(c["kind"] == "dg12" and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 1100 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e26.py dry run; planes rebuilt W in every mock call, planted wrong instance rejected, failed build "
          f"skipped, {len(new)} shapes moved to dg12; judge's check passes on every shape ({len(calls)} instances ran)")


if __name__ == "__main__":
    main()

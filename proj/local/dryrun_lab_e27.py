"""CPU dry run of proj/lab_e27.py (DG12 DIAG timings): mocks as dryrun_lab_e24 (DIAG instances compute the normal result);
checks that both diagnostic lines are printed with every DIAG variant timed and DIAG 0/4 checked, and that kernel_fn is
v10's (nothing enters CHOICE).

Run: .venv/bin/python proj/local/dryrun_lab_e27.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402
import dryrun_lab_e24 as e24  # noqa: E402


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
    e24.mock_dg12(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e27.py", "lab_e27_cpu")
    lab.dg12_include_dirs = lambda: ["/fake"]
    base_choice = dict(lab.CHOICE)
    lab.Lab.time_graph = lambda self, fn, reps=20, cold=True: (fn(), 7.0)[1]
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    buf = io.StringIO()
    with redirect_stdout(buf):
        lab.prepare(w)
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] diag ") and "(base " in ln]
    assert len(lines) == 2, out[-2000:]
    for ln in lines:
        for dd in (0, 1, 3, 4):
            assert f"d{dd}:7.00" in ln, ln
        assert "d0:7.00(ok)" in ln and "d4:7.00(ok)" in ln, ln
    assert lab.CHOICE == base_choice
    print("\n".join(lines))
    print("ok: lab_e27.py dry run; both diagnostic lines printed, every DIAG variant timed, DIAG 0/4 checked; CHOICE unchanged")


if __name__ == "__main__":
    main()

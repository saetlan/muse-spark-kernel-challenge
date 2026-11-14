"""CPU dry run of proj/lab_e20.py (v9 + lab/sweep.py + lab/e20.py: router hi/lo swap kernel at m = 128/256 and
re-timed near-winners). Triton emulated launch by launch (dryrun_lab_e18's emulators), base modules mocked.
Planted: router configs with 2 stages drop their last K slice (fastest by the fake timer); the gate must reject
them; router m256/m128/m16 move to correct candidates; the judge's check passes on the chosen kernels.

Run: .venv/bin/python proj/local/dryrun_lab_e20.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402


def planted(real, x_pos):
    def run(grid, *a, **k):
        if k.get("num_stages") == 2:
            a = list(a)
            kps = a[6]
            a[x_pos] = a[x_pos].clone()
            a[x_pos][:, -kps:] = 0
        return real(grid, *a, **k)
    return run


def main():
    d.install_stubs()
    e18.PLANT["on"] = False
    d.EMULATE["_hl_gemm_kernel"] = planted(e18._hl_gemm, 0)
    d.EMULATE["_hl_swap_kernel"] = planted(e18._hl_swap, 2)
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
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e20.py", "lab_e20_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or any(c is v for v in base_choice.values()) or c["kind"] not in ("hl_gemm", "hl_swap"):
            return 20.0
        return 1.0 if c["stages"] == 2 else 5.0 + c["BK"] / 1e4
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    full = {m: lab.swap_cands(m) for m in (128, 256)}
    for m, cs in full.items():
        assert any(c["stages"] == 2 for c in cs), m
        print(f"  router m{m}: {len(cs)} swap candidates, BN {sorted({c['BN'] for c in cs})}, splits {sorted({c['split'] for c in cs})}")
    lab.swap_cands = lambda m: [c for c in full[m] if c["stages"] != 2][:3] + [c for c in full[m] if c["stages"] == 2][:1]
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == {("router", 16), ("router", 128), ("router", 256)}, sorted(new)
    assert all(c["stages"] != 2 for c in new.values()) and new[("router", 256)]["kind"] == "hl_swap", new
    sys.path.insert(0, str(d.PROJ))
    import reference
    for m in (16, 128, 256):
        data = reference.generate_input(m, "router", 300 + m, device="cpu")
        ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
        assert ok, (m, msg)
    print("ok: lab_e20.py dry run; planted 2-stage bug rejected, router m16/128/256 moved; judge's check passes")


if __name__ == "__main__":
    main()

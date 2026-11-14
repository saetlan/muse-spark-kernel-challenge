"""CPU dry run of a lab built as <base> + lab/sweep.py + lab/e10.py (router wide-tile sweep): Triton's
_gemm_kernel emulated launch by launch (dryrun_lab_e1), the base's P9/F12V/F12M modules and libraries
mocked exactly. Planted: router configs with split 32 lose a K slice in the emulation (the fake timer
makes them fastest); the gate must reject them, and every router shape with a correct candidate moves.

Run: .venv/bin/python proj/local/dryrun_lab_e10.py [lab_e10.py]
"""

import sys
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
    real_gemm = d.EMULATE["_gemm_kernel"]

    def planted_gemm(grid, x, w, out, M, N, kps, *a, **k):
        if grid[2] == 32:   # split 32: drop the last K slice
            x = x.clone()
            x[:, -kps:] = 0
        return real_gemm(grid, x, w, out, M, N, kps, *a, **k)
    d.EMULATE["_gemm_kernel"] = planted_gemm
    w = d.weights()
    name = sys.argv[1] if len(sys.argv) > 1 else "lab_e10.py"
    lab = d.load(d.PROJ / name, name.replace(".py", "") + "_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c is base_choice.get(("router", 128)) or c is base_choice.get(("router", 256)) \
                or c is base_choice.get(("router", 32)) or c["kind"] != "gemm" or "BM" not in c:
            return 20.0
        return 1.0 if c["split"] == 32 else 5.0 + c["BN"] / 1000.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    real_cands = lab.router_wide_cands
    lab.router_wide_cands = lambda m: [c for c in real_cands(m) if c["split"] != 32][:2] + [c for c in real_cands(m) if c["split"] == 32][:1]
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == {("router", 32), ("router", 128), ("router", 256)}, sorted(new)
    assert all(c["split"] != 32 for c in new.values()), new
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: {name} dry run; planted split-32 router bug rejected, router m32/128/256 moved to wide tiles")


if __name__ == "__main__":
    main()

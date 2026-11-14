"""CPU dry run of proj/lab_e19.py (v8's kernels with the CS/XD-capable F12M + 15 x-ring instances + lab/sweep.py +
lab/e19.py). F12M is mocked (dryrun_lab_e9.mock_f12m decodes the planes; the kernel's x ring is checked by
check_f12m.py), the router hi/lo Triton kernels are emulated launch by launch (dryrun_lab_e18). Planted: x-ring
instances with XD = 2 and router configs with 6 stages drop their last K slice (fastest by the fake timer); the
gate must reject both, and every shape with a correct candidate moves.

Run: .venv/bin/python proj/local/dryrun_lab_e19.py
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
    real_g, real_s = e18._hl_gemm, e18._hl_swap

    def planted(real):
        def run(grid, *a, **k):
            if k.get("num_stages") == 6:
                e18.PLANT["on"], saved = True, None
                try:   # same planted loss as e18 (last K slice), keyed on stages here: force split 64 semantics
                    a = list(a)
                    x = a[0] if real is real_g else a[2]
                    kps = a[6]
                    x = x.clone()
                    x[:, -kps:] = 0
                    if real is real_g:
                        a[0] = x
                    else:
                        a[2] = x
                finally:
                    e18.PLANT["on"] = False
            return real(grid, *a, **k)
        return run
    d.EMULATE["_hl_gemm_kernel"] = planted(real_g)
    d.EMULATE["_hl_swap_kernel"] = planted(real_s)
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
    dryrun_lab_e9.mock_f12m(calls, planted="xd2")
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e19.py", "lab_e19_cpu")
    base_choice = dict(lab.CHOICE)
    assert {k for k, c in base_choice.items() if c["kind"].startswith("hl_")} == {("router", 32), ("router", 128), ("router", 256)}

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or any(c is v for v in base_choice.values()):
            return 20.0
        if c["kind"] == "f12m" and "XD" in c:
            return 1.0 if c["XD"] == 2 else 5.0 + c["D"] / 100.0
        if c["kind"] in ("hl_gemm", "hl_swap"):
            seeds = [sd for ss in lab.HL_SEEDS.values() for sd in ss]
            return 1.0 if c["stages"] == 6 else (5.5 if c in seeds else 5.0 + c["BK"] / 1e4 + c["split"] / 1e5)
        return 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    full = {m: lab.hl_refine(m) for m in (8, 16, 32, 128, 256)}
    for m, cs in full.items():
        assert any(c["stages"] == 6 for c in cs), (m, "no planted 6-stage candidate")
        print(f"  router m{m}: {len(cs)} refinement candidates")
    lab.hl_refine = lambda m: [c for c in full[m] if c["stages"] != 6][:3] + [c for c in full[m] if c["stages"] == 6][:1]
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {key for key, cands in lab.F12M_XD_CANDIDATES.items() if any(c["XD"] != 2 for c in cands)}
    want |= {("router", m) for m in (8, 16, 32, 128, 256)}
    assert set(new) == want, (sorted(new), sorted(want))
    for k, c in new.items():
        if c["kind"] == "f12m":
            assert c["XD"] != 2 and c["fallback"] == base_choice.get(k), (k, c)
        else:
            assert c["stages"] != 6, (k, c)
    assert any(fn.endswith("_x2") for fn in calls), "no planted XD = 2 instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: lab_e19.py dry run; planted XD=2 and 6-stage bugs rejected, {len(new)} shapes moved "
          f"({', '.join(f'{k[0]} m{k[1]}' for k in sorted(new, key=lambda k: (k[1], k[0])))})")


if __name__ == "__main__":
    main()

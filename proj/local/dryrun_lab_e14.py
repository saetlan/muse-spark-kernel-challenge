"""CPU dry run of proj/lab_e14.py (v7a + RG section + lab/sweep.py + lab/e14.py). The RG module is mocked:
it unpacks the hi/lo planes it is given (inverse of rg_frag) and computes x @ (hi + lo).T in FP64 -> FP32
(the kernel's math itself is checked by check_rg.py). Planted: RG instances with D = 1 drop their last
64 k (fastest by the fake timer; the gate must reject them); every router shape moves to a D = 2 config
with the base kernel as fallback; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e14.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def unpack_hilo(wp):
    """[16, 512, 2, 32, 8] bf16 -> hi + lo as FP64 [256, 8192] (inverse of rg_frag per half)."""
    rt, ks = wp.shape[0], wp.shape[1]
    out = 0
    for hh in range(2):
        v = wp[:, :, hh].reshape(rt, ks, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(rt * 16, ks * 16)
        out = out + v.double()
    return out


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
    calls, cache = {}, {}
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_rg":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda: None)
            elif fn.endswith("_info"):
                setattr(mod, fn, lambda: 120 * 1000000)
            else:
                rtc, ntw, cs, xs, dd = (int(v) for v in fn.split("_")[2:])

                def run(wp, x, out, rtc=rtc, ntw=ntw, dd=dd, fn=fn):
                    calls[fn] = calls.get(fn, 0) + 1
                    m = x.shape[0]
                    assert x.dtype == torch.bfloat16 and out.dtype == torch.float32 and out.shape == (m, 256)
                    assert m <= (4 // rtc) * 8 * ntw and x.is_contiguous() and wp.shape == (16, 512, 2, 32, 8)
                    if id(wp) not in cache:
                        cache[id(wp)] = unpack_hilo(wp)
                    xd = x.double()
                    if dd == 1:
                        xd = xd.clone()
                        xd[:, -64:] = 0
                    out.copy_((xd @ cache[id(wp)].T).float())
                setattr(mod, fn, run)
        return mod
    cpp = types.ModuleType("torch.utils.cpp_extension")
    cpp.load_inline = load_inline
    sys.modules["torch.utils.cpp_extension"] = cpp
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e14.py", "lab_e14_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "rg":
            return 20.0
        return 1.0 if c["D"] == 1 else 5.0 + c["CS"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    for key, cands in list(lab.RG_LAB_CANDIDATES.items()):
        lab.RG_LAB_CANDIDATES[key] = [c for c in cands if c["D"] == 2][:2] + [c for c in cands if c["D"] == 1][:1]
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == {("router", 32), ("router", 128), ("router", 256)}, sorted(new)
    assert all(c["kind"] == "rg" and c["D"] == 2 and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert any(n.endswith("_1") for n in calls), "no planted instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_rg":
            raise RuntimeError("Error building extension 'okc_proj_rg' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.rg_setup(state)
    cpp.load_inline = good
    for (layer, m) in new:
        x = lab.lab_input(m, 8192, 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": 256, "k": 8192, "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer]), layer), (layer, m)
    print(f"ok: lab_e14.py dry run; planted D=1 bug rejected, router m32/128/256 moved to RG with fallbacks; failed build falls back")


if __name__ == "__main__":
    main()

"""CPU dry run of proj/lab_e8.py (v4 + F12V section + lab/sweep.py + lab/e8.py). The F12V CUDA module is
mocked by check_f12v.emulate, which runs the kernel's own parsed CUDA helpers (decode, x conversion,
scale, indices) at full size, so packing, dispatch, the lab gate, the fallback and the output dtype
are exercised on the CPU. P9 (v4's router) and the libraries use dryrun_lab_e4/e3's exact mocks.
Planted: F12V instances with D = 2 drop their last k block (the fake timer makes them fastest; the
gate must reject every one), and a layer whose planes are missing must run its fallback.

Run: .venv/bin/python proj/local/dryrun_lab_e8.py   (after make_lab.py has built proj/lab_e8.py)
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12v  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402


def mock_f12v(calls, planted=True):
    """Wrap the installed cpp_extension mock so okc_proj_f12v builds into emulated kernels."""
    h = check_f12v.helpers()
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_f12v":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_info"):
                setattr(mod, fn, lambda: 64 * 1000000)
                continue
            inst = tuple(int(v) for v in fn.split("_")[2:])

            def run(wp, rec, x, out, n_rows, inst=inst, fn=fn):
                calls[fn] = calls.get(fn, 0) + 1
                r, rg, ksw, ns, dd = inst
                assert x.is_contiguous() and x.dtype == torch.bfloat16 and x.numel() == 1024 * ns * ksw
                assert out.shape == (1, n_rows) and out.dtype == torch.bfloat16 and n_rows % (r * rg) == 0
                xv = x.view(-1)
                if planted and dd == 2:          # planted bug: the last k block is lost
                    xv = xv.clone()
                    xv[-1024:] = 0
                got = check_f12v.emulate(inst, {"p": wp, "rec": rec}, xv, n_rows, h)
                out.copy_(got.view(1, -1).to(torch.bfloat16))
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
    dryrun_lab_e3.mock_modules({}, planted=False)
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    dryrun_lab_e4.mock_p9({}, k_perm, planted=False)
    calls = {}
    mock_f12v(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e8.py", "lab_e8_cpu")
    v4_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "f12v":
            return 20.0
        return 1.0 if c["D"] == 2 else 5.0 + c["R"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    # Keep the CPU run short: per layer two candidates that are correct (D != 2) and one planted (D = 2).
    for layer, cands in list(lab.F12V_LAB_CANDIDATES.items()):
        keep = [c for c in cands if c["D"] != 2][:2] + [c for c in cands if c["D"] == 2][:1]
        lab.F12V_LAB_CANDIDATES[layer] = keep
    state = lab.prepare(w)
    assert set(state["f12v"]) == {"qkv_gate", "o", "pooler", "unpooler"}, set(state["f12v"])
    new = {k: c for k, c in lab.CHOICE.items() if v4_choice.get(k) != c}
    assert set(new) == {(layer, 1) for layer in lab.F12V_LAB_CANDIDATES}, sorted(new)
    for (layer, m), c in new.items():
        assert c["kind"] == "f12v" and c["D"] != 2, (layer, c)
        assert c["fallback"] == v4_choice.get((layer, m)), (layer, "fallback is not v4's choice")
    planted = [n for n in calls if n.endswith("_2")]
    assert planted, "no planted instance was ever run"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    # A layer without planes (more than 7 escapes in a row) runs its fallback, still correct.
    used, real = [], lab.run_choice

    def recording(c, x, layer, n, st):
        used.append(c["kind"])
        return real(c, x, layer, n, st)
    lab.run_choice = recording
    del state["f12v"]["o"]
    x = lab.lab_input(1, 1024, 4)
    out = lab.kernel_fn({"x": x, "config": {"m": 1, "n": 8192, "k": 1024, "layer": "o"}}, state)
    assert used == ["f12v", v4_choice[("o", 1)]["kind"]], used
    assert lab.lab_ok(lab.lab_errors(out, x, w["o"]), "o")
    # nvcc failing on the judge: f12v_setup logs it and drops the planes, so every F12V shape falls back.
    cpp = sys.modules["torch.utils.cpp_extension"]
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_f12v":
            raise RuntimeError("Error building extension 'okc_proj_f12v' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.f12v_setup(state)
    cpp.load_inline = good
    assert state["f12v"] == {}, "planes kept after a failed build"
    used.clear()
    x = lab.lab_input(1, 8192, 6)
    out = lab.kernel_fn({"x": x, "config": {"m": 1, "n": 4096, "k": 8192, "layer": "pooler"}}, state)
    assert used == ["f12v", v4_choice[("pooler", 1)]["kind"]], used
    assert lab.lab_ok(lab.lab_errors(out, x, w["pooler"]), "pooler")
    print(f"ok: lab_e8.py dry run; {len(calls)} F12V instances called, planted D=2 bug rejected on every layer, "
          f"{len(new)} shapes moved to F12V with v4's kernel as fallback; a layer without planes falls back")


if __name__ == "__main__":
    main()

"""CPU dry run of proj/lab_e9.py (base with P9 + F12V sections, + F12M section + lab/sweep.py + lab/e8.py +
lab/e9.py). The F12M module is mocked by decoding the planes it is given (check_f12m.decode_weights: the
parsed CUDA decode, inverse fragment order, unscale, escapes) or P9's BF16 fragments for the F12 = False
twins, then x @ W.T; the kernel's own indexing is proven by check_f12m.py. F12V, P9 and the libraries
use the e8/e4/e3 mocks. Planted: F12M instances with D = 4 lose their last 64 k (fastest by the fake
timer; the gate must reject them), and a failed nvcc must leave every F12M shape on its fallback.

Run: .venv/bin/python proj/local/dryrun_lab_e9.py   (after make_lab.py has built proj/lab_e9.py)
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402


def mock_f12m(calls, planted=True):
    h = check_f12m.helpers()
    import dryrun_lab_e4
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_f12m":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_info"):
                setattr(mod, fn, lambda: 96 * 1000000)
                continue
            parts = fn.split("_")[2:]
            tgn = int(parts.pop()[1:]) if parts[-1].startswith("t") else 1      # token-group suffix _tN (PE21)
            xring = int(parts.pop()[1:]) if parts[-1].startswith("x") else None    # x ring suffix _xN (PE19)
            cs = int(parts.pop()[1:]) if parts[-1].startswith("c") else 1     # split-K suffix _cN (P36)
            nt, rtw, rg, ksw, ns, dd, f12 = (int(v) for v in parts)
            assert xring is None or (xring < dd and dd % xring == 0), fn

            def run(wp, rec, x, out, m, n_rows, nt=nt, rb=16 * rtw * rg, k=64 * ns * ksw * cs, dd=dd, f12=f12, fn=fn, cs=cs, xring=xring, tgn=tgn):
                calls[fn] = calls.get(fn, 0) + 1
                assert x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (m, k) and 1 <= m <= 8 * nt * tgn
                assert out.shape == (m, n_rows) and out.dtype == torch.bfloat16 and n_rows % rb == 0
                key = (id(wp), f12)
                if key not in cache:
                    if f12:
                        cache[key] = check_f12m.decode_weights({"p": wp, "rec": rec}, k, h["decode"])
                    else:
                        cache[key] = dryrun_lab_e4.unpack(wp, wp.shape[0] * 16, k_perm).double()
                w = cache[key][:n_rows]
                xd = x.double()
                if (planted is True and dd == 4) or (planted == "cs8" and cs == 8) or (planted == "xd2" and xring == 2) or (planted == "tg4" and tgn == 4):
                    xd = xd.clone()
                    xd[:, -64:] = 0
                out.copy_((xd @ w.T).to(torch.bfloat16))
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
    dryrun_lab_e3.mock_modules({}, planted=False)
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    dryrun_lab_e4.mock_p9({}, k_perm, planted=False)
    dryrun_lab_e8.mock_f12v({}, planted=False)
    calls = {}
    mock_f12m(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e9.py", "lab_e9_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] not in ("f12m", "f12v"):
            return 20.0
        if c["kind"] == "f12v":
            return 6.0
        return 1.0 if c["D"] == 4 else 5.0 + c["RTW"] / 100.0 + (0.0 if c["F12"] else 0.5)
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    for key, cands in list(lab.F12M_LAB_CANDIDATES.items()):
        keep = [c for c in cands if c["D"] != 4][:2] + [c for c in cands if c["D"] == 4][:1]
        keep += [c for c in cands if not c["F12"] and c not in keep][:1]
        lab.F12M_LAB_CANDIDATES[key] = keep
    for layer, cands in list(lab.F12V_LAB_CANDIDATES.items()):
        lab.F12V_LAB_CANDIDATES[layer] = [c for c in cands if c["D"] != 2][:1]
    state = lab.prepare(w)
    assert set(state["f12m"]) == {"qkv_gate", "o", "pooler", "unpooler"}, set(state["f12m"])
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {key for key in lab.F12M_LAB_CANDIDATES} | {(layer, 1) for layer in lab.F12V_LAB_CANDIDATES}
    assert set(new) == want, sorted(set(new) ^ want)
    for (layer, m), c in new.items():
        assert c["fallback"] == base_choice.get((layer, m)), (layer, m, "fallback is not the base's choice")
        if c["kind"] == "f12m":
            assert c["D"] != 4 and c["F12"] and c["NT"] == m // 8, (layer, m, c)
    assert any(n.endswith("_4_1") or n.endswith("_4_0") for n in calls), "no planted instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        for dist in ("normal", "outliers"):
            x = lab.lab_input(m, w[layer].shape[1], 9, dist=dist)
            out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
            assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c, dist)
    # a failed F12M build: every F12M shape runs its fallback, still correct
    cpp = sys.modules["torch.utils.cpp_extension"]
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_f12m":
            raise RuntimeError("Error building extension 'okc_proj_f12m' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.f12m_setup(state)
    cpp.load_inline = good
    assert state["f12m"] == {} and lab.F12M_MOD is None
    used, real = [], lab.run_choice

    def recording(c, x, layer, n, st):
        used.append(c["kind"])
        return real(c, x, layer, n, st)
    lab.run_choice = recording
    for (layer, m), c in new.items():
        if c["kind"] != "f12m":
            continue
        used.clear()
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        fb = base_choice.get((layer, m))
        assert used[:2] == ["f12m"] + ([fb["kind"]] if fb else []), (layer, m, used)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m)
    print(f"ok: lab_e9.py dry run; {len(calls)} F12M instances called, planted D=4 bug rejected, "
          f"{len(new)} shapes moved (F12M at m 8/16/32, F12V at m 1) with fallbacks; a failed build falls back")


if __name__ == "__main__":
    main()

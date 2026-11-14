"""CPU dry run of proj/lab_e30.py (v10 + lab/f12mlv.py + lab/e30.py) plus a clang syntax check of the module source it
generates. The okc_proj_f12mlv module is mocked through dryrun_lab_e9.mock_f12m (x @ W from the F12 planes): the lv_
wrappers take the extra cluster size, which must be 0, 1 or 2; planted: LVC 1 drops the last 64 k, which the sweep's gate
must report as wrong. CHOICE must be unchanged (lab only) and the judge's check must pass on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e30.py
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_cuda_syntax  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402


def mock_lv(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_f12mlv":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "@@INSTANCES@@" not in cuda_sources and cuda_sources.count("cudaLaunchAttributeClusterDimension") >= 2
        inner = prev("okc_proj_f12m", "", "", [f[3:] if f.startswith("lv_") else f for f in functions])
        mod = types.SimpleNamespace()
        for fn in functions:
            if not fn.startswith("lv_"):
                setattr(mod, fn, getattr(inner, fn))
                continue
            base = getattr(inner, fn[3:])

            def run(wp, rec, x, out, m, n_rows, cdim, base=base, fn=fn):
                assert cdim in (0, 1, 2)
                calls[(fn, cdim)] = calls.get((fn, cdim), 0) + 1
                if cdim == 1:
                    xx = x.clone()
                    xx[:, -64:] = 0
                    return base(wp, rec, xx, out, m, n_rows)
                return base(wp, rec, x, out, m, n_rows)
            setattr(mod, fn, run)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


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
    f12m_calls = {}
    dryrun_lab_e9.mock_f12m(f12m_calls, planted=False)
    calls = {}
    mock_lv(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e30.py", "lab_e30_cpu")
    base_choice = {k: dict(v) for k, v in lab.CHOICE.items()}
    lab.Lab.cold_l2 = lambda self: None
    lab.Lab.time_graph = lambda self, fn, reps=20, cold=True: (fn(), 10.0)[1]
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] f12mlv ") and "(sglang" in ln]
    assert len(lines) == 3 and "f12mlv build ok (2 + 1 split-K instances)" in out, out[-2000:]
    for ln in lines:
        planted = "qkv_gate m8 " not in ln          # qkv m8 has no LVC 1 entry
        assert ("1 wrong" if planted else "0 wrong") in ln and "0 failed" in ln and ("lvc1" not in ln), ln
    assert {("lv_f12m_run_2_1_1_8_16_2_1", 2), ("lv_f12m_run_1_1_1_8_16_4_1", 2)} <= set(calls) and f12m_calls.get("f12m_run_2_1_1_8_8_2_1_c2"), (calls, f12m_calls)
    assert lab.CHOICE == base_choice, "the lab changed CHOICE"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        data = reference.generate_input(m, layer, 1900 + m, dist="normal", device="cpu")
        ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
        assert ok, (layer, m, msg)
    every = [(c, lab.FULL[layer][1]) for (layer, m), cs in lab.F12MLV_TARGETS.items() for c in cs]
    insts = tuple(dict.fromkeys(lab.f12mlv_inst(c, k) for c, k in every if c["LVC"] != "cs2"))
    cs_insts = tuple(dict.fromkeys(lab.f12mlv_inst(c, k) for c, k in every if c["LVC"] == "cs2"))
    src = lab.f12mlv_source(insts, cs_insts)
    res = check_cuda_syntax.check("f12mlv", src)
    print(f"syntax: {res}")
    print("ok: lab_e30.py dry run; 3 shapes timed with LVC 0/1/2 (+ split-K at qkv m16), planted LVC 1 caught as wrong on each, "
          "CHOICE unchanged, judge's check passes on every shape")
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()

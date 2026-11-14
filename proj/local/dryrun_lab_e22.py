"""CPU dry run of proj/lab_e22.py (v10 + DGX section + lab/sweep.py + lab/e22.py). Each per-instance load_inline module
okc_proj_dgx_* is mocked (x @ W[:n].T in FP64 -> BF16; shape checks as the C++ wrapper's), the base's modules and Triton
kernels as in dryrun_lab_e21. Planted: the qkv m128 instance (BN 48) drops its last 64 k (fastest by the fake timer:
the gate must reject it) and the o m128 instance fails to build (partial build: the others still run). Also checks
the thread-pool build path and that every chosen shape passes the judge's check.

Run: .venv/bin/python proj/local/dryrun_lab_e22.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402


def mock_dgx(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if not name.startswith("okc_proj_dgx_"):
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert kw.get("no_implicit_headers") is True and "#include <torch" not in cuda_sources
        assert "deep_gemm/impls/sm90_bf16_gemm.cuh" in cuda_sources and "@@" not in cuda_sources + cpp_sources
        parts = name[len("okc_proj_dgx_"):].split("_")
        bm, bn, st, swd, nmath, cl, n, k = (int(v) for v in parts)
        if bn == 128 and k == 1024:
            raise RuntimeError(f"Error building extension '{name}': cuda.cu(12): error: planted failure")
        mod = types.SimpleNamespace()
        fn_name = functions[0]

        def run(x, w, out, n=n, k=k, bn=bn):
            calls[fn_name] = calls.get(fn_name, 0) + 1
            assert x.dtype == w.dtype == out.dtype == torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
            assert x.shape[1] == k and w.shape[1] == k and w.shape[0] >= n and out.shape == (x.shape[0], n)
            xd = x.double()
            if bn == 48:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ w[:n].double().T).to(torch.bfloat16))
        setattr(mod, fn_name, run)
        setattr(mod, fn_name + "_attr", lambda: None)
        setattr(mod, fn_name + "_info", lambda: 168 * 1000000)
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
    dryrun_lab_e9.mock_f12m({}, planted=False)
    calls = {}
    mock_dgx(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e22.py", "lab_e22_cpu")
    lab.dgx_include_dirs = lambda: ["/fake/deep_gemm/include"]
    lab.dgx_recon = lambda: "fake"
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgx":
            return 20.0
        return 1.0 if c["BN"] == 48 else 5.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {("pooler", 128), ("unpooler", 128), ("pooler", 256), ("pooler", 32)}
    assert set(new) == want, (sorted(new), sorted(want))
    assert all(c["kind"] == "dgx" and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert len(lab.DGX_OK) == 5 and any(n.endswith("_2304_8192") for n in calls), (lab.DGX_OK, calls)
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        data = reference.generate_input(m, layer, 900 + m, device="cpu")
        ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
        assert ok, (layer, m, msg)
    print(f"ok: lab_e22.py dry run; planted wrong instance rejected, failed build skipped, {len(new)} shapes moved to dgx "
          f"({', '.join(f'{k[0]} m{k[1]}' for k in sorted(new))}); judge's check passes on every shape")


if __name__ == "__main__":
    main()

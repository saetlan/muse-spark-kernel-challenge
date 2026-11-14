"""CPU dry run of proj/lab_e24.py (v10 + DG12 section + lab/sweep.py + lab/e24.py). Each per-instance module okc_proj_dg12_*
is mocked: it rebuilds W from the planes the lab passes (parsed CUDA decode, escape list ent8/offs8, column scale) and
computes x @ W.T in FP64 -> BF16, with the C++ wrapper's checks; base modules and Triton as in dryrun_lab_e22. Planted:
the qkv m128 instance (BN 48) drops its last 64 k (fastest by the fake timer: the gate must reject it) and the o m32
instance fails to build. Every chosen shape must pass the judge's check.

Run: .venv/bin/python proj/local/dryrun_lab_e24.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402
import check_dg12  # noqa: E402


def mock_dg12(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_dg12.dg12_helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if not name.startswith("okc_proj_dg12_"):
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert kw.get("no_implicit_headers") is True and "#include <torch" not in cuda_sources
        assert "dg12_gemm_impl<N, K, BM, BN, SWD, ST, NMATH, NUM_SMS" in cuda_sources and "@@" not in cuda_sources + cpp_sources
        parts = name[len("okc_proj_dg12_"):].split("_")
        diag = int(parts.pop()[1:]) if parts[-1].startswith("d") else 0          # DIAG suffix _dN (PE27)
        bm, bn, st, swd, nmath, n, k = (int(v) for v in parts)
        if bm == 32 and bn == 64 and k == 1024:
            raise RuntimeError(f"Error building extension '{name}': cuda.cu(40): error: planted failure")
        mod = types.SimpleNamespace()
        fn_name = functions[0]

        def run(x, q, l, scale, offs8, ent8, out, n=n, k=k, bn=bn):
            calls[fn_name] = calls.get(fn_name, 0) + 1
            assert x.dtype == out.dtype == torch.bfloat16 and q.dtype == l.dtype == offs8.dtype == ent8.dtype == torch.int32
            assert scale.dtype == torch.float32 and x.shape[1] == k and out.shape == (x.shape[0], n)
            assert q.shape == (k // 64, n, 8) and l.shape == (k // 64, n, 8, 2) and offs8.shape == ((n // bn) * 4 + 1,)
            key = (id(q), bn)
            if key not in cache:
                qr = q.permute(1, 0, 2).reshape(n, k // 8)                  # chunk-major -> row-major
                lr = l.permute(1, 0, 2, 3).reshape(n, k // 8, 2)
                w = check_dg12.dg12_unpack_reference({"q": qr, "l": lr}, n, k, dec)
                for b in range(n // bn):                           # per-(n-block, lane class) table (PF45)
                    for t in range(4):
                        for e in ent8[int(offs8[4 * b + t]):int(offs8[4 * b + t + 1])].tolist():
                            assert ((e[0] & 7) >> 1) == t and 0 <= e[0] < bn
                            w[b * bn + e[0], e[1]] = torch.tensor([e[2]], dtype=torch.int32).view(torch.float32).double()
                cache[key] = w * scale.double().view(-1, 1)
            xd = x.double()
            if bn == 48:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[key].T).to(torch.bfloat16))
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
    mock_dg12(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e24.py", "lab_e24_cpu")
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
    print(f"ok: lab_e24.py dry run; planes rebuilt W in every mock call, planted wrong instance rejected, failed build "
          f"skipped, {len(new)} shapes moved to dg12; judge's check passes on every shape ({len(calls)} instances ran)")


if __name__ == "__main__":
    main()

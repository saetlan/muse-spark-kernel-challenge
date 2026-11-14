"""CPU dry run of proj/lab_e29.py (v10 + DGRS section + lab/sweep.py + lab/e29.py). The dgrs_ext module is mocked: dgrs_gemm
rebuilds W from the chunk-major planes it is given (permuted back to natural order, decoded by dryrun_lab_e15.decode_natural
with the parsed F12M decode, escapes from rec) and writes x @ W.T in FP64 -> BF16, checking the buffers' sizes like the
kernel's TORCH_CHECKs. Planted: configs with 3 splits drop their last 64 k (fastest by the fake timer: the gate must
reject them). Every chosen shape must pass the judge's check; the error-flag line must print.

Run: .venv/bin/python proj/local/dryrun_lab_e29.py
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e15  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402


def mock_dgrs(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "dgrs_ext":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "rs_kernel<BM>" in cuda_sources and "DGRS_CASE(32)" in cuda_sources and "@@" not in cuda_sources
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax):
            calls[(bm, splits)] = calls.get((bm, splits), 0) + 1
            kc = K // 64
            per = -(-kc // splits)
            tiles = -(-M // bm)
            assert (splits - 1) * per < kc and part.numel() >= splits * tiles * bm * N and cnt.numel() >= (N // 128) * tiles
            assert wp.shape[0] == kc and x.shape == (M, K) and out.shape == (M, N)
            key = id(wp)
            if key not in cache:
                cache[key] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            xd = x.double()
            if splits == 3:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[key][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        mod.dgrs_attrs = lambda bm: torch.tensor([140, 0])
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
    mock_dgrs(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e29.py", "lab_e29_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgrs":
            return 20.0
        return 1.0 if c["S"] == 3 else 5.0 + c["S"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    assert "dgrs error flags: 0 of" in out and "dgrs build ok" in out, out[-1500:]
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {k for k, cs in lab.DGRS_TARGETS.items() if any(c["S"] != 3 for c in cs)}   # S == 3 is planted wrong
    assert set(new) == want, (sorted(new), sorted(want))
    assert all(c["kind"] == "dgrs" and c["S"] != 3 and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert (32, 3) in calls or (64, 3) in calls or (128, 3) in calls, "no planted 3-split config ran"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 1300 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e29.py dry run; planted 3-split bug rejected, {len(new)} shapes moved to dgrs with fallbacks; "
          f"judge's check passes on every shape; error-flag line printed")


if __name__ == "__main__":
    main()

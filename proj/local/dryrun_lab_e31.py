"""CPU dry run of proj/lab_e31.py (v10 + DGRS v3 section + lab/sweep.py + lab/e31.py). dgrs_ext is mocked with v3's API:
dgrs_gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red) rebuilds W from the chunk-major planes
(dryrun_lab_e15.decode_natural with the parsed F12M decode) and writes x @ W.T, checking the instance was built, the split
rule, red 1's buffers and the CTA count; diag != 0 writes nothing. Planted: CPS 2 with red 1 drops its last 64 k (fastest by
the fake timer: the gate must reject it). The build runs in the worker thread; every shape must move with its fallback; the
judge's check must pass on every shape; the DIAG and error-flag lines must print.

Run: .venv/bin/python proj/local/dryrun_lab_e31.py
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


def mock_dgrs(calls, built):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "dgrs_ext":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "rs_kernel<BM, CPS, DIAG>" in cuda_sources and "@@" not in cuda_sources
        for b, cps, dg in [(int(a), int(b_), int(c_)) for a, b_, c_ in
                           __import__("re").findall(r"DGRS_CASE\((\d+), (\d+), (\d+)\)", cuda_sources)]:
            built.add((b, cps, dg))
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red):
            assert (bm, cps, diag) in built, (bm, cps, diag)
            calls[(bm, splits, cps, red, diag)] = calls.get((bm, splits, cps, red, diag), 0) + 1
            kc = K // 64
            per = -(-kc // splits)
            tiles = -(-M // bm)
            assert 1 <= splits <= 8 and (splits - 1) * per < kc and x.shape == (M, K) and out.shape == (M, N) and x.is_contiguous()
            assert N // 128 * tiles * splits <= 132 * cps and (cps == 1 or bm <= 64)
            assert red == 0 or splits == 1 or (part.numel() >= splits * tiles * bm * N and cnt.numel() >= N // 128 * tiles)
            if diag:
                return
            key = id(wp)
            if key not in cache:
                cache[key] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            xd = x.double()
            if cps == 2 and red == 1:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[key][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        mod.dgrs_attrs = lambda bm, cps, diag: torch.tensor([90 if cps == 2 else 120, 0])
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
    calls, built = {}, set()
    mock_dgrs(calls, built)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e31.py", "lab_e31_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgrs":
            return 20.0
        return 1.0 if (c.get("CPS") == 2 and c.get("RED") == 1) else 5.0 + c["S"] / 100.0 + c["BM"] / 1000.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    assert "dgrs error flags: 0 of" in out and "dgrs build ok" in out and "dgrs diag unpooler m32 bm32s2c1r0: d0:" in out, out[-2500:]
    diag_line = next(ln for ln in out.splitlines() if "dgrs diag unpooler" in ln)
    assert all(f"d{b}:" in diag_line and f"d{b}:ERR" not in diag_line for b in (0, 1, 2, 4, 8)), diag_line
    assert built == set(lab.dgrs_cfg_insts()) and {(32, 1, 1), (32, 1, 2), (32, 1, 4), (32, 1, 8), (64, 2, 0)} <= built, built
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(lab.DGRS_TARGETS), (sorted(new), sorted(lab.DGRS_TARGETS))
    assert all(c["kind"] == "dgrs" and not (c["CPS"] == 2 and c["RED"] == 1) and c["fallback"] == base_choice.get(k)
               for k, c in new.items()), new
    assert any(k[2] == 2 and k[3] == 1 for k in calls), "no planted CPS 2 / red 1 config ran"
    assert sum(1 for ln in out.splitlines() if "wrong)" in ln and " 0 wrong)" not in ln) == 4, "planted configs not all rejected"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 2100 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e31.py dry run; {len(built)} instances built in the worker thread; planted CPS 2 / red 1 rejected on 4 shapes; "
          f"{len(new)} shapes moved to dgrs with fallbacks; DIAG line complete; judge's check passes on every shape")
    print("  " + diag_line)


if __name__ == "__main__":
    main()

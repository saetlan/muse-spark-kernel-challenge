"""CPU dry run of proj/lab_e34.py (v10 + DGRS v6 section + lab/sweep.py + lab/dgrs6.py + lab/e34.py). dgrs_ext is mocked with
v6's API (red 0 / 1 / 2) (dgrs_gemm / dgrs_attrs / dgrs_max_clusters / dgrs_stages): x @ W from the chunk-major planes, the instance must be
built, the split rule, CTA count, MATH 4 limits and red 1's buffers are checked, diag != 0 writes nothing. Planted: MATH 4
with red 1 drops its last 64 k (fastest by the fake timer: the gate must reject it, 2 shapes). Every shape must move with
v10's entry as fallback; both DIAG lines (5 runs each), the stages, occupancy and error-flag lines must print; the judge's
check must pass on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e34.py
"""

import io
import re
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


def mock_dgrs5(calls, built):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "dgrs_ext":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "rs_kernel<BM, CPS, DIAG, MATH>" in cuda_sources and "pre_corr" in cuda_sources and "@@" not in cuda_sources
        assert functions == ["dgrs_gemm", "dgrs_attrs", "dgrs_max_clusters", "dgrs_stages"], functions
        for t in re.findall(r"DGRS_CASE\((\d+), (\d+), (\d+), (\d+)\)", cuda_sources):
            built.add(tuple(int(v) for v in t))
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red, math):
            assert (bm, cps, diag, math) in built, (bm, cps, diag, math)
            calls[(bm, splits, red, math, diag)] = calls.get((bm, splits, red, math, diag), 0) + 1
            kc = K // 64
            per = -(-kc // splits)
            tiles = -(-M // bm)
            assert 1 <= splits <= 8 and (splits - 1) * per < kc and x.shape == (M, K) and out.shape == (M, N) and x.is_contiguous()
            assert N // 128 * tiles * splits <= 132 * cps and (math == 2 or (bm <= 64 and cps == 1))
            assert red in (0, 1, 2) and (red != 1 or splits == 1 or (part.numel() >= splits * tiles * bm * N and cnt.numel() >= N // 128 * tiles))
            if diag:
                return
            if id(wp) not in cache:
                cache[id(wp)] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            xd = x.double()
            if math == 4 and red == 1 and splits > 1:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[id(wp)][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        mod.dgrs_attrs = lambda bm, cps, diag, math: torch.tensor([112 if math == 4 else 90, 0])
        mod.dgrs_max_clusters = lambda bm, cps, diag, math, splits: 0 if splits < 2 else 132 // splits
        mod.dgrs_stages = lambda bm, cps, math, splits, red, smax: 8 if bm <= 64 else 6
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
    mock_dgrs5(calls, built)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e34.py", "lab_e34_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgrs":
            return 20.0
        return 1.0 if (c["MATH"] == 4 and c["RED"] == 1) else 5.0 + c["S"] / 100.0 + c["MATH"] / 1000.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    for key in ("dgrs build ok", "dgrs max active clusters BM/CPS/MATH/S:", "dgrs ring stages (smax 8):", "dgrs error flags: 0 of"):
        assert key in out, (key, out[-2500:])
    diags = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs diag ") and "bm32s2:" in ln]
    assert len(diags) == 2, diags
    for ln in diags:
        assert all(f"{t}:" in ln and f"{t}:ERR" not in ln for t in ("r0m4d0", "r2m4d0", "r0m4d16", "r0m4d32", "r2m4d32")), ln
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(lab.DGRS_TARGETS), sorted(new)
    assert all(c["kind"] == "dgrs" and not (c["MATH"] == 4 and c["RED"] == 1) and c["fallback"] == base_choice.get(k)
               for k, c in new.items()), new
    want = {(32, 1, 0, 4), (32, 1, 16, 4), (32, 1, 32, 4), (64, 1, 0, 4)}
    assert built == set(lab.dgrs_cfg_insts()) == want, (built, want)
    dlines = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs ") and "(sglang" in ln]
    assert len(dlines) == 6 and sum(1 for ln in dlines if "1 wrong" in ln) == 2, dlines
    assert any(k[2] == 2 for k in calls), "no red 2 config ran"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 2900 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e34.py dry run; {len(built)} instances built in the worker thread; planted MATH 4 / red 1 rejected on 2 shapes; "
          f"{len(new)} shapes moved with fallbacks; 2 DIAG lines complete; stages / occupancy / error lines; judge's check passes")
    for ln in diags:
        print("  " + ln)


if __name__ == "__main__":
    main()

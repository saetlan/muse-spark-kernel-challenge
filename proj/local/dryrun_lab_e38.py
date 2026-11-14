"""CPU dry run of proj/lab_e38.py (v10 + pinned DGRS v9 section + lab/sweep.py + lab/dgrs9.py + lab/e38.py). Both modules are
mocked with v9's API: "dgrs_ext" (generic instances, RED -1 / SPL 0) and "dgrs_ext_spec" (6-tuple instances); a call must match
a generic instance by (BM, CPS, DIAG, MATH) or, in the spec module, a specialized one by (BM, CPS, DIAG, MATH, RED, SPL = splits);
x @ W from the chunk-major planes; DIAG 64 writes synthetic stamps; DIAG 512 leaves the values unchanged. Planted: the
specialized pooler S3 red 0 drops its last 64 k (fastest by the fake timer: rejected; pooler still moves). Both DIAG lines (6
runs), both stamp lines (G and S, planted values recovered), the stages line, every sweep shape and the judge's check.

Run: .venv/bin/python proj/local/dryrun_lab_e38.py
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

FUNCS = ["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"]


def mock_dgrs9(calls, built, strict=True):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name not in ("dgrs_ext", "dgrs_ext_spec"):
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert functions == FUNCS and "pre_ring" in cuda_sources and "@@" not in cuda_sources, functions
        insts = {tuple(int(v) for v in t) for t in re.findall(r"DGRS_CASE\((\d+), (\d+), (\d+), (\d+), (-?\d+), (\d+)\)", cuda_sources)}
        assert insts, "no instances"
        assert not strict or all((t[4] >= 0) == (name == "dgrs_ext_spec") for t in insts), (name, insts)
        built[name] = insts
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red, math):
            spec = any(t[:4] == (bm, cps, diag, math) and t[4] == red and t[5] == splits for t in insts if t[4] >= 0)
            gen = any(t[:4] == (bm, cps, diag, math) for t in insts if t[4] < 0)
            assert spec or gen, (name, bm, cps, diag, math, red, splits)
            calls[(name, bm, splits, red, math, diag)] = calls.get((name, bm, splits, red, math, diag), 0) + 1
            kc = K // 64
            per = -(-kc // splits)
            tiles = -(-M // bm)
            assert 1 <= splits <= 8 and (splits - 1) * per < kc and x.shape == (M, K) and out.shape == (M, N) and x.is_contiguous()
            assert N // 128 * tiles * splits <= 132 * cps and (math == 2 or (bm <= 64 and cps == 1))
            assert not (diag & 512) or per <= 256
            if diag & 64:
                ctas = N // 128 * tiles * splits
                cyc = torch.tensor([0, 0, 1755, 2 * 1755, 3 * 1755, 4 * 1755, 4 * 1755 + 100, 4 * 1755 + 600, 5 * 1755, 0, 0,
                                    3 * 1755 + 50, 0, 3 * 1755 + 351, 4 * 1755 + 200, 0], dtype=torch.int64)
                st = cyc.repeat(ctas, 1) + 1000
                st[:, 0] = 10 ** 6 + torch.arange(ctas) * 2
                st[:, 9] = st[:, 0] + 5000
                st[:, 10] = torch.arange(ctas)
                part.view(torch.int32)[: 16 * ctas].copy_(st.flatten().to(torch.int32))
            if diag & ~(512 | 64):
                return
            if id(wp) not in cache:
                cache[id(wp)] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            xd = x.double()
            if spec and red == 0 and splits == 3:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[id(wp)][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        mod.dgrs_attrs_s = lambda bm, cps, diag, math, red, spl: torch.tensor([96, 0 if red >= 0 else 32])
        mod.dgrs_attrs = lambda bm, cps, diag, math: torch.tensor([96, 32])
        mod.dgrs_max_clusters = lambda bm, cps, diag, math, splits: 0 if splits < 2 else 132 // splits
        mod.dgrs_stages = lambda bm, cps, math, splits, red, smax: 8
        mod.dgrs_stages_d = lambda bm, cps, math, splits, red, smax, diag: 8
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
    calls, built = {}, {}
    mock_dgrs9(calls, built)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e38.py", "lab_e38_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgrs":
            return 20.0
        return 3.0 if c.get("SPEC") else (4.0 if c.get("DIAG") == 512 else 5.0)
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    for key in ("dgrs gen build ok", "dgrs spec build ok", "dgrs ring stages (smax 8):", "dgrs error flags: 0 of"):
        assert key in out, (key, out[-2500:])
    gen, spec = lab.dgrs_cfg_insts()
    assert built["dgrs_ext"] == {t + (-1, 0) for t in gen} and built["dgrs_ext_spec"] == set(spec), (built, gen, spec)
    assert set(spec) == {(32, 1, 0, 4, 2, 2), (32, 1, 512, 4, 2, 2), (32, 1, 576, 4, 2, 2), (32, 1, 512, 4, 0, 3),
                         (32, 1, 512, 4, 2, 3), (32, 1, 512, 4, 2, 4)}, spec
    diags = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs diag ") and "r2m4:" in ln]
    assert len(diags) == 2 and all(all(f"{t}:" in ln and f"{t}:ERR" not in ln and f"{t}:not built" not in ln
                                       for t in ("d0G", "d512G", "d576G", "d0S", "d512S", "d576S")) for ln in diags), diags
    stamps = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs stamps ") and "(med/max" in ln]
    assert len(stamps) == 2, stamps
    for ln in stamps:
        assert ln.count("loop0 3.00/3.00") == 2 and ln.count("loads 0.06/0.06") == 2 and ln.count("push-issue 0.20/0.20") == 2, ln
        assert "d576G (128 CTAs)" in ln and "d576S (128 CTAs)" in ln, ln
    assert any(k[0] == "dgrs_ext_spec" and k[5] == 576 for k in calls) and any(k[0] == "dgrs_ext" and k[5] == 576 for k in calls)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(lab.DGRS_TARGETS), sorted(new)
    assert new[("pooler", 32)] != dict(new[("pooler", 32)], RED=0, SPEC=True), new[("pooler", 32)]
    assert all(c["kind"] == "dgrs" and c["fallback"] == base_choice.get(k) and not (c["SPEC"] and c["RED"] == 0 and c["S"] == 3)
               for k, c in new.items()), new
    dlines = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs ") and "(sglang" in ln]
    assert len(dlines) == 4 and [("pooler" in ln) for ln in dlines if " 1 wrong" in ln] == [True], dlines
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 3700 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e38.py dry run; generic ({len(gen)}) and specialized ({len(spec)}) modules built in worker threads and "
          f"dispatched by their own rules; planted wrong spec pooler S3 r0 rejected; {len(new)} shapes moved with fallbacks; "
          f"both DIAG and stamp lines complete; judge's check passes on every shape")
    for ln in diags:
        print("  " + ln)


if __name__ == "__main__":
    main()

"""CPU dry run of proj/lab_e41.py (v10 + pinned DGRS v10 section + lab/sweep.py + lab/dgrs10b.py + lab/e41.py): only the
"dgrs_ext_full" module (every config fully specialized, each called with its own smax = SMX), mocked with v10's 10-field
dispatch. Both modules are
mocked with v9's API: "dgrs_ext" (generic instances, RED -1 / SPL 0) and "dgrs_ext_spec" (6-tuple instances); a call must match
a generic instance by (BM, CPS, DIAG, MATH) or, in the spec module, a specialized one by (BM, CPS, DIAG, MATH, RED, SPL = splits);
x @ W from the chunk-major planes; DIAG 64 writes synthetic stamps; DIAG 512 leaves the values unchanged. Planted: the
specialized pooler S3 red 0 drops its last 64 k (fastest by the fake timer: rejected; pooler still moves). Both DIAG lines (6
runs), both stamp lines (G and S, planted values recovered), the stages line, every sweep shape and the judge's check.

Run: .venv/bin/python proj/local/dryrun_lab_e41.py
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

FUNCS10 = ["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_attrs_x", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"]


def mock_dgrs10(calls, built):
    import re
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name not in ("dgrs_ext_spec", "dgrs_ext_full"):
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert functions == FUNCS10 and "@@" not in cuda_sources, functions
        insts = {tuple(int(v) for v in t) for t in
                 re.findall(r"DGRS_CASE\((\d+), (\d+), (\d+), (\d+), (-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\)", cuda_sources)}
        assert insts and all((t[6] > 0) == (name == "dgrs_ext_full") for t in insts), (name, insts)
        built[name] = insts
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red, math):
            hit = [t for t in insts if t[:4] == (bm, cps, diag, math) and (t[4] < 0 or t[4] == red) and (t[5] in (0, splits))
                   and (t[6] in (0, smax)) and (t[7] in (0, M)) and (t[8] in (0, N)) and (t[9] in (0, K))]
            assert hit, (name, bm, cps, diag, math, red, splits, smax, M, N, K)
            calls[(name, bm, splits, red, math, diag, N)] = calls.get((name, bm, splits, red, math, diag, N), 0) + 1
            if id(wp) not in cache:
                cache[id(wp)] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            xd = x.double()
            if name == "dgrs_ext_full" and red == 0 and splits == 3:
                xd = xd.clone()
                xd[:, -64:] = 0
            out.copy_((xd @ cache[id(wp)][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        mod.dgrs_attrs_x = lambda *a: torch.tensor([96, 24])
        mod.dgrs_stages_d = lambda bm, cps, math, splits, red, smax, diag: 8
        mod.dgrs_max_clusters = lambda bm, cps, diag, math, splits: 0 if splits < 2 else 132 // splits
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline

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
    mock_dgrs10(calls, built)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e41.py", "lab_e41_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "dgrs":
            return 20.0
        return 3.0 if c.get("SMX", 8) > 8 else 4.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    for key in ("dgrs full build ok", "dgrs ring stages (each config's smax):", "dgrs error flags: 0 of"):
        assert key in out, (key, out[-2500:])
    spec, full = lab.dgrs_cfg_insts()
    assert not spec and "dgrs_ext_spec" not in built and built["dgrs_ext_full"] == set(full) and len(full) == 11, (built, spec, full)
    assert (32, 1, 0, 4, 2, 2, 10, 32, 8192, 4096) in full and (32, 1, 0, 4, 0, 3, 12, 32, 4096, 8192) in full, full
    assert (32, 1, 0, 2, 2, 2, 8, 32, 8192, 1024) in full, full
    assert {k[6] for k in calls if k[0] == "dgrs_ext_full"} == {8192, 4096, 2304}, calls
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(lab.DGRS_TARGETS), sorted(new)
    assert not (new[("pooler", 32)]["RED"] == 0 and new[("pooler", 32)]["S"] == 3), new[("pooler", 32)]
    assert all(c["kind"] == "dgrs" and c["fallback"] == base_choice.get(k) and not (c["MOD"] == "full" and c["RED"] == 0 and c["S"] == 3)
               for k, c in new.items()), new
    assert all(c["MOD"] == "full" and (c["SMX"] > 8 or k == ("o", 32)) for k, c in new.items()), new
    dlines = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs ") and "(sglang" in ln]
    assert len(dlines) == 4 and [("pooler m32" in ln) for ln in dlines if " 0 wrong" not in ln] == [True] and any(" 2 wrong" in ln for ln in dlines), dlines
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 4700 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e41.py dry run; {len(full)} fully specialized instances (ring depth 8 / 10 / 12) in one worker-thread "
          f"module, each called with its own smax; planted wrong pooler S3 r0 (both depths) rejected; {len(new)} shapes moved with "
          f"fallbacks; judge's check passes on every shape")


if __name__ == "__main__":
    main()

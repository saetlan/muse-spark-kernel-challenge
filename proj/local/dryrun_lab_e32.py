"""CPU dry run of proj/lab_e32.py (v10 + DGRS v4 section + lab/hlp.py + lab/dgrs4.py + lab/e32.py). Triton is the dry-run stub; the HLP kernels are emulated
following the launch exactly: the zero kernel zeroes out over its grid (must cover m * n), the GEMM kernels reuse
dryrun_lab_e18's HL emulation with atomics (which asserts the output was zeroed first) and must be launched with
launch_pdl=True. Planted: "hlpt" with half the splits drops its last split (fastest by the fake timer: the gate must reject
it). DG12-RS v4 is mocked (v4 API; x @ W from the chunk-major planes; diag != 0 writes nothing); planted: MATH 4 with red 1
drops its last 64 k (fastest by the fake timer: rejected). Every router and DGRS shape must move with v10's entry as
fallback; the DIAG, cluster-occupancy and error-flag lines must print; the judge's check must pass on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e32.py
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

LAUNCHES = []
BASE_SPLIT = {16: 32, 32: 32, 128: 16, 256: 16}        # v10's HL split per m (HL_CHOICE)


def zero(grid, out, numel, BLOCK, num_warps):
    assert grid[0] * BLOCK >= numel == out.numel() and out.dtype == torch.float32
    out.zero_()
    LAUNCHES.append("zero")


def hlp_gemm(grid, x, hi, lo, out, M, N, kps, sx, sw, so, BM, BN, BK, num_warps, num_stages, launch_pdl):
    assert launch_pdl is True and sx == x.stride(0) and sw == hi.stride(0) == lo.stride(0) and so == out.stride(0)
    planted = grid[2] == BASE_SPLIT[M] // 2 and len(LAUNCHES) and LAUNCHES[-1] == "torch.zeros"
    _hl_planted(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, False, planted, grid[2])
    LAUNCHES.append("gemm")


def hlp_swap(grid, hi, lo, x, out, M, N, kps, sw, sx, so, BM, BN, BK, num_warps, num_stages, launch_pdl):
    assert launch_pdl is True and sx == x.stride(0) and sw == hi.stride(0) == lo.stride(0) and so == out.stride(0)
    planted = grid[1] == BASE_SPLIT[M] // 2 and len(LAUNCHES) and LAUNCHES[-1] == "torch.zeros"
    _hl_planted(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, True, planted, grid[1])
    LAUNCHES.append("swap")


def _hl_planted(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, swap, planted, splits):
    if not planted:
        return e18._hl(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, True, swap)
    assert torch.count_nonzero(out) == 0
    for s in range(splits - 1):                    # planted: the last split is dropped
        sl = slice(s * kps, (s + 1) * kps)
        out += x[:, sl].float() @ hi[:N, sl].float().T + x[:, sl].float() @ lo[:N, sl].float().T


def mock_dgrs4(calls, built):
    import re
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "dgrs_ext":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "rs_kernel<BM, CPS, DIAG, MATH>" in cuda_sources and "@@" not in cuda_sources
        assert functions == ["dgrs_gemm", "dgrs_attrs", "dgrs_max_clusters"], functions
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
            assert red == 0 or splits == 1 or (part.numel() >= splits * tiles * bm * N and cnt.numel() >= N // 128 * tiles)
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
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    d.install_stubs()
    extra = types.ModuleType("triton.language.extra")
    cuda = types.ModuleType("triton.language.extra.cuda")
    cuda.gdc_wait = cuda.gdc_launch_dependents = lambda *a, **k: None
    extra.cuda = cuda
    sys.modules["triton.language"].extra = extra
    sys.modules["triton.language.extra"], sys.modules["triton.language.extra.cuda"] = extra, cuda
    e18.PLANT["on"] = False
    d.EMULATE["_hl_gemm_kernel"] = e18._hl_gemm
    d.EMULATE["_hl_swap_kernel"] = e18._hl_swap
    d.EMULATE["_hlp_zero_kernel"] = zero
    d.EMULATE["_hlp_gemm_kernel"] = hlp_gemm
    d.EMULATE["_hlp_swap_kernel"] = hlp_swap
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
    dcalls, built = {}, set()
    mock_dgrs4(dcalls, built)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e32.py", "lab_e32_cpu")
    zeros = torch.zeros

    def zeros_marked(*a, **k):
        LAUNCHES.append("torch.zeros")
        return zeros(*a, **k)
    lab.torch.zeros = zeros_marked
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is not None and c["kind"] == "dgrs":
            return 1.0 if (c["MATH"] == 4 and c["RED"] == 1) else 5.0 + c["S"] / 100.0 + c["MATH"] / 1000.0
        if c is None or c["kind"] not in ("hlp", "hlpt"):
            return 20.0
        b = c["base"]
        half = b["split"] == HALF[(b["kind"], b["BM"], b["BK"])] // 2
        return 1.0 if (c["kind"] == "hlpt" and half) else 5.0 + (0.5 if c["kind"] == "hlpt" else 0.0) + (0.2 if half else 0.0)
    HALF = {("hl_gemm", 16, 128): 32, ("hl_swap", 32, 128): 32, ("hl_gemm", 128, 128): 16, ("hl_gemm", 128, 64): 16}
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    lab.torch.zeros = zeros
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] hlp router") and "(sglang" in ln]
    assert len(lines) == 4, out[-2000:]
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == {("router", m) for m in (16, 32, 128, 256)} | set(lab.DGRS_TARGETS), sorted(new)
    for k, c in new.items():
        if k[0] == "router":
            assert c["kind"] == "hlp" and c["base"] == base_choice[k] and c["fallback"] == base_choice[k], (k, c)
        else:
            assert c["kind"] == "dgrs" and not (c["MATH"] == 4 and c["RED"] == 1) and c["fallback"] == base_choice.get(k), (k, c)
    assert built == set(lab.dgrs_cfg_insts()) and (32, 1, 16, 4) in built and (64, 1, 0, 4) in built, built
    assert any(k[3] == 4 and k[2] == 1 for k in dcalls), "no planted MATH 4 / red 1 config ran"
    diag = next(ln for ln in out.splitlines() if "dgrs diag unpooler" in ln)
    assert all(f"m{mm}d{b}:" in diag and f"m{mm}d{b}:ERR" not in diag for b, mm in ((0, 2), (0, 4), (16, 2), (1, 2), (16, 4), (1, 4))), diag
    assert "dgrs max active clusters BM/CPS/MATH/S:" in out and "dgrs error flags: 0 of" in out and "dgrs build ok" in out, out[-2500:]
    dlines = [ln for ln in out.splitlines() if ln.startswith("[lab] dgrs ") and "(sglang" in ln]
    assert len(dlines) == 12 and sum(1 for ln in dlines if "1 wrong" in ln) == 3, dlines
    assert sum(1 for ln in lines if "1 wrong" in ln) == 4, lines
    assert "zero" in LAUNCHES and "gemm" in LAUNCHES and "swap" in LAUNCHES
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 2300 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print("ok: lab_e32.py dry run; planted half-split hlpt rejected on all 4 router shapes, planted MATH 4 / red 1 on 3 DGRS "
          f"shapes; {len(new)} shapes moved (v10 entry as fallback); zero kernel ran before every PDL GEMM's atomics; "
          f"{len(built)} DGRS instances built in the worker thread; DIAG line complete; judge's check passes on every shape")
    print("  " + diag)
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()

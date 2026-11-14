"""CPU dry run of proj/lab_e18.py (<base> + lab/sweep.py + lab/e18.py: the router with pre-split BF16 hi/lo read by
_hl_gemm_kernel / _hl_swap_kernel). Triton is emulated launch by launch (dryrun_lab_e1's stubs; the two new
kernels here: grid, split-K ranges, strides, atomics into zeros vs stores, FP32 out, x @ hi.T + x @ lo.T in
FP32 per K slice), the base's modules and libraries mocked. Checks hl_setup's hi/lo against SGLang's split.
Planted: configs with split 64 lose their last K slice (fastest by the fake timer); the gate must reject
them, and every router shape with a correct candidate moves to the new kinds.

Run: .venv/bin/python proj/local/dryrun_lab_e18.py [lab_e18.py]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402

PLANT = {"on": True}


def _hl(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, ATOMIC, swap):
    assert hi.dtype == lo.dtype == torch.bfloat16 and hi.is_contiguous() and lo.is_contiguous() and hi.shape == lo.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and out.dtype == torch.float32 and out.shape == (M, N)
    assert kps % BK == 0 and BK >= 16 and BM >= 16 and BN >= 16, (kps, BK, BM, BN)
    tiles_n = grid[0] if swap else grid[1]
    splits = grid[1] if swap else grid[2]
    assert tiles_n * BN >= N and (tiles_n - 1) * BN < N, (tiles_n, BN, N)
    assert splits * kps == x.shape[1] == hi.shape[1], (splits, kps, x.shape)
    if swap:
        assert M <= BM, (M, BM)
    else:
        assert grid[0] * BM >= M and (grid[0] - 1) * BM < M, (grid, BM, M)
    assert ATOMIC == (splits > 1)
    if ATOMIC:
        assert torch.count_nonzero(out) == 0, "split-K atomics need a zeroed output"
    for s in range(splits):
        if PLANT["on"] and splits == 64 and s == splits - 1:
            continue
        sl = slice(s * kps, (s + 1) * kps)
        xs = x[:, sl].float()
        part = xs @ hi[:N, sl].float().T + xs @ lo[:N, sl].float().T
        if ATOMIC:
            out += part
        else:
            out.copy_(part)


def _hl_gemm(grid, x, hi, lo, out, M, N, kps, sx, sw, so, BM, BN, BK, ATOMIC, num_warps, num_stages):
    assert sx == x.stride(0) and sw == hi.stride(0) == lo.stride(0) and so == out.stride(0)
    _hl(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, ATOMIC, False)


def _hl_swap(grid, hi, lo, x, out, M, N, kps, sw, sx, so, BM, BN, BK, ATOMIC, num_warps, num_stages):
    assert sx == x.stride(0) and sw == hi.stride(0) == lo.stride(0) and so == out.stride(0)
    _hl(grid, x, hi, lo, out, M, N, kps, BM, BN, BK, ATOMIC, True)


def main():
    d.install_stubs()
    d.EMULATE["_hl_gemm_kernel"] = _hl_gemm
    d.EMULATE["_hl_swap_kernel"] = _hl_swap
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
    w = d.weights()
    name = sys.argv[1] if len(sys.argv) > 1 else "lab_e18.py"
    lab = d.load(d.PROJ / name, name.replace(".py", "") + "_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] not in ("hl_gemm", "hl_swap"):
            return 20.0
        return 1.0 if c["split"] == 64 else 5.0 + c["BN"] / 1000.0 + c["BK"] / 1e5
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    full = {m: lab.hl_cands(m) for m in (8, 16, 32, 128, 256)}
    for m, cs in full.items():
        assert cs and any(c["split"] == 64 for c in cs) or m >= 128, (m, "no planted split-64 candidate")
        print(f"  router m{m}: {len(cs)} candidates, kinds {sorted({c['kind'] for c in cs})}, "
              f"splits {sorted({c['split'] for c in cs})}, BM {sorted({c['BM'] for c in cs})}")
    lab.hl_cands = lambda m: ([c for c in full[m] if c["split"] != 64][:3] + [c for c in full[m] if c["split"] == 64][:1])
    state = lab.prepare(w)
    hi = w["router"].to(torch.bfloat16)
    assert torch.equal(state["router_hi"], hi) and torch.equal(state["router_lo"], (w["router"] - hi.float()).to(torch.bfloat16))
    assert torch.equal(state["router_hi"].t(), state["router_hi_t"]) and torch.equal(state["router_lo"].t(), state["router_lo_t"])
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == {("router", m) for m in (8, 16, 32, 128, 256)}, sorted(new)
    assert all(c["kind"] in ("hl_gemm", "hl_swap") and c["split"] != 64 for c in new.values()), new
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        for seed, dist in ((9, "outliers"), (10, "normal")):
            x = lab.lab_input(m, w[layer].shape[1], seed, dist=dist)
            out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
            assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    # the judge's own check on the new kinds (reference.check_implementation, FP64 reference, probe rows)
    sys.path.insert(0, str(d.PROJ))
    import reference
    for m in (8, 9, 16, 32, 128, 256):
        data = reference.generate_input(m, "router", 77 + m, device="cpu")
        c = lab.CHOICE.get(("router", m))
        out = lab.kernel_fn(data, state)
        ok, msg = reference.check_implementation(data, out, w)
        assert ok, (m, c, msg)
    print(f"ok: {name} dry run; planted split-64 bug rejected, router m8/16/32/128/256 moved to "
          f"{sorted({c['kind'] for c in new.values()})}; the judge's check passes on the chosen kinds")


if __name__ == "__main__":
    main()

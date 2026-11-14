"""CPU dry run of proj/lab_e15.py (v7a + SX section + lab/sweep.py + lab/e15.py). The SX module is mocked: it
decodes the natural-order 12-bit planes it is given (parsed decode, inverse fragment order, per-row scale,
escapes from the records), checks the flags against the records, and computes x @ W.T in FP64 -> BF16 (the
kernel's math itself is checked by check_sx.py). Planted: SX instances with CS = 8 drop their last 64 k
(fastest by the fake timer; the gate must reject them); every SX shape with a correct candidate moves to SX
with the base kernel as fallback; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e15.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402

sys.path.insert(0, str(d.PROJ / "sx"))
from sx_layout import sx_flags  # noqa: E402


def decode_natural(p, rec, k, dec):
    """FP64 W [N, K] from SX planes (F12M format, natural k)."""
    p = check_f12m.u32(p.to(torch.int64))
    rt = p.shape[0]
    n = 16 * rt
    regs = check_f12m.regs_of(p, dec)
    el = torch.stack([(regs[..., e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    wb = el.view(rt, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
    r = check_f12m.u32(rec.to(torch.int64)).view(n, 8)
    base = r[:, 0] & 0xFF
    w = check_f12m.bf16_bits_to_f64(wb) * torch.pow(2.0, (base - 112).double()).view(-1, 1)
    count = (r[:, 0] >> 8) & 0xFF
    for j in range(7):
        rows = (count > j).nonzero().flatten()
        e = r[rows, 1 + j]
        w[rows, e & 0xFFFF] = check_f12m.bf16_bits_to_f64(e >> 16)
    return w


def mock_sx(calls, planted=True):
    h = check_f12m.helpers()
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_sx":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda: None)
                continue
            if fn.endswith("_info"):
                setattr(mod, fn, lambda: 120 * 1000000)
                continue
            if fn.startswith("sxw_run_"):
                nb, wgs, ksw, ns, pp, cs = (int(v) for v in fn.split("_")[2:])

                def run_w(wp, rec, x, out, n_rows, mt=128 * nb, rows=64 * wgs, k=64 * ns * ksw * cs, pp=pp, fn=fn):
                    calls[fn] = calls.get(fn, 0) + 1
                    m = x.shape[0]
                    assert x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (m, k) and 1 <= m <= mt
                    assert out.shape == (m, n_rows) and out.dtype == torch.bfloat16 and n_rows % rows == 0
                    assert wp.dtype == torch.int32 and wp.shape[1:] == (k // 64, 3, 32, 4) and wp.shape[0] * 16 >= n_rows
                    key = id(wp)
                    if key not in cache:
                        cache[key] = decode_natural(wp, rec, k, h["decode"])
                    w = cache[key][:n_rows]
                    xd = x.double()
                    if planted and pp == 1:
                        xd = xd.clone()
                        xd[:, -64:] = 0
                    out.copy_((xd @ w.T).to(torch.bfloat16))
                setattr(mod, fn, run_w)
                continue
            nt8, tg, rtw, rgn, ksw, ns, dd, xs, f12, cs = (int(v) for v in fn.split("_")[2:])

            def run(wp, rec, flags, x, out, n_rows, mt=8 * nt8 * tg, rows=16 * rtw * rgn, k=64 * ns * ksw * cs,
                    cs=cs, fn=fn):
                calls[fn] = calls.get(fn, 0) + 1
                m = x.shape[0]
                assert x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (m, k) and 1 <= m <= mt
                assert out.shape == (m, n_rows) and out.dtype == torch.bfloat16 and n_rows % rows == 0
                assert wp.dtype == torch.int32 and wp.shape[1:] == (k // 64, 3, 32, 4) and wp.shape[0] * 16 >= n_rows
                key = id(wp)
                if key not in cache:
                    nn = wp.shape[0] * 16
                    assert torch.equal(flags, sx_flags(rec, nn, k)), "flags do not match the records"
                    cache[key] = decode_natural(wp, rec, k, h["decode"])
                w = cache[key][:n_rows]
                xd = x.double()
                if planted and cs == 8:
                    xd = xd.clone()
                    xd[:, -64:] = 0
                out.copy_((xd @ w.T).to(torch.bfloat16))
            setattr(mod, fn, run)
        return mod
    cpp = types.ModuleType("torch.utils.cpp_extension")
    cpp.load_inline = load_inline
    sys.modules["torch.utils.cpp_extension"] = cpp


def main():
    d.install_stubs()
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
    mock_sx(calls)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e15.py", "lab_e15_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "sx":
            return 20.0
        return 1.0 if c["CS"] == 8 else 5.0 + c["NS"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = {key for key, cands in lab.SX_LAB_CANDIDATES.items() if any(c["CS"] != 8 for c in cands)}
    assert set(new) == want, (sorted(new), sorted(want))
    assert ("qkv_gate", 128) not in new, "a planted CS = 8 instance was chosen"
    assert all(c["kind"] == "sx" and c["CS"] != 8 and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert any(fn.endswith("_8") for fn in calls), "no planted instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    good = sys.modules["torch.utils.cpp_extension"].load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_sx":
            raise RuntimeError("Error building extension 'okc_proj_sx' (planted)")
        return good(name, *a, **k)
    sys.modules["torch.utils.cpp_extension"].load_inline = failing
    lab.sx_setup(state)
    sys.modules["torch.utils.cpp_extension"].load_inline = good
    assert lab.SX_MOD is None and state["sx"] == {}
    for (layer, m) in new:
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m)
    print(f"ok: lab_e15.py dry run; planted CS=8 bug rejected, {len(new)} shapes moved to SX with fallbacks "
          f"({', '.join(f'{k[0]} m{k[1]}' for k in sorted(new, key=lambda k: (k[1], k[0])))}); failed build falls back")


if __name__ == "__main__":
    main()

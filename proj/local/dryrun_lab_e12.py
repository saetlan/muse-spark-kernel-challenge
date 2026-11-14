"""CPU dry run of a WGF lab (base with F12M + WG sections + lab/sweep.py + lab/e10.py + lab/e11.py +
lab/e12.py). The WG module is mocked: wg_run_* = x @ W.T, wgf_run_* = x @ decode(natural F12M planes).T
(check_f12m.decode_weights with the natural k order); WGF's kernel math itself is checked by check_wg.py.
Planted: WGF instances with D = 1 drop their last 64 k (fastest by the fake timer; the gate must reject
them); every WGF shape with a correct candidate moves; a failed nvcc falls back.

Run: .venv/bin/python proj/local/dryrun_lab_e12.py [lab file]
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402


def decode_natural(planes, k, dec):
    """FP64 W from natural-order F12M planes (check_f12m.decode_weights without the k_perm)."""
    p = check_f12m.u32(planes["p"].to(torch.int64))
    rt = p.shape[0]
    n = 16 * rt
    regs = check_f12m.regs_of(p, dec)
    el = torch.stack([(regs[..., e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    wb = el.view(rt, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
    rec = check_f12m.u32(planes["rec"].to(torch.int64)).view(n, 8)
    base = rec[:, 0] & 0xFF
    w = check_f12m.bf16_bits_to_f64(wb) * torch.pow(2.0, (base - 112).double()).view(-1, 1)
    count = (rec[:, 0] >> 8) & 0xFF
    for j in range(7):
        rows = (count > j).nonzero().flatten()
        e = rec[rows, 1 + j]
        w[rows, e & 0xFFFF] = check_f12m.bf16_bits_to_f64(e >> 16)
    return w


def mock_wg_all(calls, planted=True):
    h = check_f12m.helpers()
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        if name != "okc_proj_wg":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "@@" not in cuda_sources
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda: None)
                continue
            if fn.endswith("_info"):
                setattr(mod, fn, lambda: 140 * 1000000)
                continue
            parts = [int(v) for v in fn.split("_")[2:]]
            if fn.startswith("wgf_"):
                ks, st, tt, dd = parts

                def run(wp, rec, x, out, n_rows, tt=tt, dd=dd, fn=fn):
                    calls[fn] = calls.get(fn, 0) + 1
                    m, k = x.shape
                    assert x.dtype == out.dtype == torch.bfloat16 and x.is_contiguous() and 1 <= m <= 128 * tt and out.shape == (m, n_rows)
                    key = id(wp)
                    if key not in cache:
                        cache[key] = decode_natural({"p": wp, "rec": rec}, k, h["decode"])
                    xd = x.double()
                    if planted and dd == 1:
                        xd = xd.clone()
                        xd[:, -64:] = 0
                    out.copy_((xd @ cache[key][:n_rows].T).to(torch.bfloat16))
            else:
                ks, st, tt = parts

                def run(w, x, out, n_rows, tt=tt, fn=fn):
                    calls[fn] = calls.get(fn, 0) + 1
                    out.copy_((x.double() @ w[:n_rows].double().T).to(torch.bfloat16))
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
    mock_wg_all(calls)
    w = d.weights()
    name = sys.argv[1] if len(sys.argv) > 1 else "lab_e12.py"
    lab = d.load(d.PROJ / name, name.replace(".py", "") + "_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or any(c is v for v in base_choice.values()) or c["kind"] not in ("wgf", "wg"):
            return 20.0
        if c["kind"] == "wgf":
            return 1.0 if c["D"] == 1 else 5.0 + c["ST"] / 100.0
        return 8.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    for key, cands in list(lab.WGF_LAB_CANDIDATES.items()):
        lab.WGF_LAB_CANDIDATES[key] = [c for c in cands if c["D"] != 1][:1] + [c for c in cands if c["D"] == 1][:1]
    state = lab.prepare(w)
    assert set(state["wgf"]) == {"qkv_gate", "o", "pooler", "unpooler"}
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    want = set(lab.WGF_LAB_CANDIDATES)
    assert set(new) >= want, sorted(want - set(new))
    for key in want:
        c = new[key]
        assert c["kind"] == "wgf" and c["D"] != 1 and c["fallback"] == base_choice.get(key), (key, c)
    assert any(n.startswith("wgf_run_") and n.endswith("_1") for n in calls), "no planted instance ran"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9, dist="outliers")
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    cpp = sys.modules["torch.utils.cpp_extension"]
    good = cpp.load_inline

    def failing(name, *a, **k):
        if name == "okc_proj_wg":
            raise RuntimeError("Error building extension 'okc_proj_wg' (planted)")
        return good(name, *a, **k)
    cpp.load_inline = failing
    lab.wg_setup(state)
    cpp.load_inline = good
    assert lab.WG_MOD is None
    for key in want:
        layer, m = key
        x = lab.lab_input(m, w[layer].shape[1], 3)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": lab.lab_n(layer, m), "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:lab.lab_n(layer, m)]), layer), key
    print(f"ok: {name} dry run; planted D=1 bug rejected, {len(want)} WGF shapes moved with fallbacks; a failed build falls back")


if __name__ == "__main__":
    main()

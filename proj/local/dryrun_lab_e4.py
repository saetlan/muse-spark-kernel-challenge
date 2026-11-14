"""CPU dry run of proj/lab_e4.py (v2 + P9 section + lab/sweep.py + lab/e4.py). The P9 CUDA module is
mocked: each instance unpacks the packed weights it is given (the inverse of p9_pack, vectorized)
and computes x @ W.T (hi + lo for the router), so packing, the unpooler's row-tile slice, dispatch,
output dtype and the lab gate are exercised; the kernel's own index math is checked by check_p9.py.
Planted: CS = 8 instances drop their last K slice (must be rejected by the gate even though the
fake timer makes them fastest).

Run: .venv/bin/python proj/local/dryrun_lab_e4.py   (after make_lab.py has built proj/lab_e4.py)
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def unpack(wp, n, k_perm):
    """Inverse of p9_pack for the first n rows: [N/16, K/16, 32, 8] -> [n, K] (undoes k_perm)."""
    rt, ks = n // 16, wp.shape[1]
    v = wp[:rt].reshape(rt, ks, 8, 4, 2, 2, 2)            # rt, ks, g, t, c, h, p
    wperm = v.permute(0, 5, 2, 1, 4, 3, 6).reshape(n, ks * 16)
    w = torch.empty_like(wperm)
    w[:, k_perm(ks * 16)] = wperm
    return w


def unpack_f12(qp, rp, tops, erow, ecol, ebits, n, k, kp):
    """W [n, K] from P22 planes: decode (the kernel's ops), undo the fragment order and k_perm, then put
    the escaped weights back from the CSR list."""
    from p22_layout import p22_decode_all
    rt = n // 16
    enc = {"q": qp[:rt].to(torch.int64) & 0xFFFFFFFF, "r": rp[:rt].to(torch.int64) & 0xFFFFFFFF,
           "tops": tops[:rt].to(torch.int64)}
    dec = p22_decode_all(enc).to(torch.int16).view(torch.bfloat16)
    w = torch.empty(n, k, dtype=torch.bfloat16)
    w[:, kp(k)] = unpack(dec, n, lambda kk: torch.arange(kk))
    for row in range(n):
        for e in range(int(erow[row]), int(erow[row + 1])):
            w[row, int(ecol[e])] = ebits[e:e + 1].view(torch.bfloat16)[0]
    return w


def mock_p9(calls, kp, planted=True, exempt=()):
    def load_inline(name, cpp_sources, cuda_sources, functions, **kw):
        assert name == "okc_proj_p9", name
        mod = types.SimpleNamespace()
        for fn in functions:
            if fn.endswith("_attr"):
                setattr(mod, fn, lambda smem: None)
            elif fn.endswith("_info"):
                setattr(mod, fn, lambda: 64 * 1000000)
            else:
                parts = fn.split("_")[2:]               # nt rtw rg ksw d cs hilo f32
                cs, hilo = int(parts[5]), parts[6] == "1"

                f12 = fn.endswith("_f12")

                def run(wp, qp, rp, tops, erow, ecol, ebits, x, out, m, n, k, smem, cs=cs, hilo=hilo, fn=fn, f12=f12):
                    calls[fn] = calls.get(fn, 0) + 1
                    assert x.is_contiguous() and x.shape == (m, k) and out.shape == (m, n)
                    assert out.dtype == (torch.float32 if hilo else torch.bfloat16)
                    if f12:
                        w = unpack_f12(qp, rp, tops, erow, ecol, ebits, n, k, kp).double()
                    elif hilo:
                        w = unpack(wp[:, :, 0], n, kp).double() + unpack(wp[:, :, 1], n, kp).double()
                    else:
                        w = unpack(wp, n, kp).double()
                    xd = x.double()
                    if planted and cs == 8 and fn not in exempt:  # planted bug (not in the base's own choices)
                        xd = xd.clone()
                        xd[:, k - k // 8:] = 0
                    out.copy_((xd @ w.T).to(out.dtype))
                setattr(mod, fn, run)
        return mod
    cpp = types.ModuleType("torch.utils.cpp_extension")
    cpp.load_inline = load_inline
    sys.modules["torch.utils.cpp_extension"] = cpp


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: torch.nn.functional.linear(x, w)
    calls = {}
    import dryrun_lab_e3
    dryrun_lab_e3.mock_modules({}, planted=False)  # the v3 base calls FlashInfer and DeepGEMM
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    exempt = set()   # P9 instances the base file itself chooses: never planted
    mock_p9(calls, k_perm, exempt=exempt)
    w = d.weights()
    lab_name = sys.argv[1] if len(sys.argv) > 1 else "lab_e4.py"
    lab = d.load(d.PROJ / lab_name, lab_name.replace(".py", "") + "_cpu")
    v2_choice = dict(lab.CHOICE)
    for (layer, m), c in v2_choice.items():
        if c["kind"] == "p9":
            hilo = layer == "router"
            inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], hilo, hilo) + ((True,) if c.get("F12") else ())
            exempt.add(lab.p9_name(inst))

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "p9":
            return 20.0
        return 1.0 if c["CS"] == 8 else 5.0 + c["D"] / 100.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    lab.LAB_MAX_S = 1e9
    lab.PREPARE_CAP_S = 1e9
    # Keep the CPU run short: per shape two non-planted candidates (CS != 8) and one planted (CS = 8)
    # when present, so every shape is expected to move and every planted one must be rejected.
    for key, cands in list(lab.P9_LAB_CANDIDATES.items()):
        keep = [c for c in cands if c["CS"] != 8][:2] + [c for c in cands if c["CS"] == 8][:1]
        lab.P9_LAB_CANDIDATES[key] = keep
    state = lab.prepare(w)
    new = {k: c for k, c in lab.CHOICE.items() if v2_choice.get(k) != c}
    swept = set(lab.P9_LAB_CANDIDATES)
    def inst_name(layer, c):
        hilo = layer == "router"
        return lab.p9_name((c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], hilo, hilo) + ((True,) if c.get("F12") else ()))
    non_planted = {k for k, cands in lab.P9_LAB_CANDIDATES.items()
                   if any((c["CS"] != 8 or inst_name(k[0], c) in exempt) and dict(kind="p9", **c) != v2_choice.get(k)
                          for c in cands)}
    assert set(new) == non_planted, sorted(set(new) ^ non_planted)
    assert all(c["kind"] == "p9" and (c["CS"] != 8 or inst_name(k[0], c) in exempt) for k, c in new.items()), new
    assert all(k in lab.CHOICE for k in v2_choice)
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: {lab_name} dry run; {len(calls)} P9 instances called, planted CS=8 bug rejected, "
          f"{len(new)} of {len(swept)} swept shapes moved to P9")


if __name__ == "__main__":
    main()

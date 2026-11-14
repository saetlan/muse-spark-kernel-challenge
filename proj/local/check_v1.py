"""CPU check of proj/submission_v1.py (or the file given as argument) under the dry run's stubs (Triton kernels emulated in
PyTorch, image kernels as F.linear): prepare() with its warm-up, every CHOICE shape against FP64
within half the judge's bounds (probe rows included), and off-grid shapes equal to baseline.py.

Run: .venv/bin/python proj/local/check_v1.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: F.linear(x, w)
    import dryrun_lab_e3
    dryrun_lab_e3.mock_modules({}, planted=False)  # FlashInfer / DeepGEMM / SGLang router, exact
    import dryrun_lab_e4
    sys.path.insert(0, str(d.PROJ / "p9"))
    from p9_layout import k_perm
    dryrun_lab_e4.mock_p9({}, k_perm, planted=False)  # P9 module (v4+), exact
    import dryrun_lab_e8
    dryrun_lab_e8.mock_f12v({}, planted=False)  # F12V module (v5+): the parsed CUDA helpers, exact
    import dryrun_lab_e9
    dryrun_lab_e9.mock_f12m({}, planted=False)  # F12M module (v6+): decoded planes, exact
    import dryrun_lab_e12
    dryrun_lab_e12.mock_wg_all({}, planted=False)  # WG/WGF module (v8+): x @ W.T, decoded natural planes
    import dryrun_lab_e15
    dryrun_lab_e15.mock_sx({}, planted=False)  # SX module (v9+): decoded natural planes, flags checked
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e1.py", "lab_e1_cpu_v1")  # for lab_input / lab_errors / lab_ok only
    name = sys.argv[1] if len(sys.argv) > 1 else "submission_v1.py"
    v1 = d.load(d.PROJ / name, name.replace(".py", "") + "_cpu")
    base = d.load(d.PROJ / "baseline.py", "baseline_cpu_v1")
    state, bstate = v1.prepare(w), base.prepare(w)
    # Record which config kernel_fn dispatches to (the stubs compute the same math, so a
    # dispatch bug would not show in the outputs).
    used, real_run_choice = [], v1.run_choice

    def recording_run_choice(c, x, layer, n, st):
        used.append(c)
        return real_run_choice(c, x, layer, n, st)
    v1.run_choice = recording_run_choice
    for (layer, m), c in v1.CHOICE.items():
        n = lab.lab_n(layer, m)
        for dist in ("normal", "outliers"):
            x = lab.lab_input(m, w[layer].shape[1], 11 + m, dist=dist)
            used.clear()
            out = v1.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
            assert len(used) == 1 and used[0] is c, (layer, m, "dispatched to", used)
            errs = lab.lab_errors(out, x, w[layer][:n])
            assert lab.lab_ok(errs, layer), (layer, m, c, errs)
    for layer, m in (("qkv_gate", 512), ("router", 9), ("unpooler", 5), ("pooler", 7), ("o", 33), ("unpooler", 130)):
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 5)
        data = {"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}
        used.clear()
        assert torch.equal(v1.kernel_fn(data, state), base.kernel_fn(data, bstate)), (layer, m)
        assert not used, (layer, m, "off-grid shape dispatched to", used)
    print(f"ok: {name}: {len(v1.CHOICE)} chosen shapes correct on normal and outlier inputs; off-grid = baseline")


if __name__ == "__main__":
    main()

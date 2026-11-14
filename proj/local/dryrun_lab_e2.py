"""CPU dry run of proj/lab_e2.py (v1 + lab/sweep.py + lab/e2.py) with Triton emulated
(dryrun_lab_e1's stubs) and a fake timer under which fma configs are faster than v1.

Per shape it uses a subset of the real candidates that covers every launch path (split 1,
split 2, split > 2, every BN), plus two planted bad kernels that the fake timer makes fastest:
  WRONG  BN 4, 4 warps, split 2: 1% error on every output;
  PROBE  BN 8, 4 warps, split 1: 1% error only on probe rows (single-nonzero inputs), which the
         judge's relative-L2 bound would let through on BF16 layers.
Checks: every non-planted config in the subsets is correct when run directly; both planted ones
are rejected by the lab's gate; every swept shape moves to fma; v1's other choices are kept;
every final choice is correct.

Run: .venv/bin/python proj/local/dryrun_lab_e2.py   (after make_lab.py has built proj/lab_e2.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402

SWEPT = [("router", m) for m in (1, 8, 16, 32)] + [(lay, m) for m in (1, 8)
                                                    for lay in ("unpooler", "o", "qkv_gate", "pooler")]


def planted_kind(bn, warps, split):
    if (bn, warps, split) == (4, 4, 2):
        return "WRONG"
    if (bn, warps, split) == (8, 4, 1):
        return "PROBE"
    return None


def subset(cands):
    seen, out = set(), []
    for c in cands:
        key = (min(c["split"], 3), c["BN"])
        if key not in seen or planted_kind(c["BN"], c["warps"], c["split"]):
            seen.add(key)
            out.append(c)
    return out


def main():
    d.install_stubs()
    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = lambda x, w, max_m=16: F.linear(x, w)
    fma = d.EMULATE["_fma_kernel"]

    def fma_planted(grid, w, x, out, N, sw, sx, so, M, BN, BK, ATOMIC, OUT_BF16, num_warps):
        fma(grid, w, x, out, N, sw, sx, so, M, BN, BK, ATOMIC, OUT_BF16, num_warps)
        kind = planted_kind(BN, num_warps, grid[1])
        if kind == "WRONG":
            out.mul_(1.01 if out.dtype == torch.float32 else 1.0).add_(0.01 * out.float().abs().mean())
        elif kind == "PROBE":
            rows = ((x != 0).sum(-1) == 1)
            out[rows] = (out[rows].float() * 1.01).to(out.dtype)
    d.EMULATE["_fma_kernel"] = fma_planted
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e2.py", "lab_e2_cpu")
    v1_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        for cell in fn.__closure__ or ():
            c = cell.cell_contents
            if isinstance(c, dict) and c.get("kind") == "fma":
                return 1.0 if planted_kind(c["BN"], c["warps"], c["split"]) else 5.0 + c["BN"] / 100
        return 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    full = lab.fma_candidates
    lab.fma_candidates = lambda layer, m, n, k: subset(full(layer, m, n, k))

    # 1. Every config in the subsets, run directly: correct unless planted (catches host-side bugs
    #    the lab would otherwise just skip as "failed").
    state0 = {k: v.contiguous() for k, v in w.items()}
    paths, planted_seen = set(), set()
    for layer, m in SWEPT:
        n, k = lab.lab_n(layer, m), w[layer].shape[1]
        x = lab.lab_input(m, k, 21 + m)
        xp = lab.lab_input(m, k, 22 + m, probe_all=True)
        for c in lab.fma_candidates(layer, m, n, k):
            ok = all(lab.lab_ok(lab.lab_errors(lab.run_choice(c, xi, layer, n, state0), xi, w[layer][:n]), layer)
                     for xi in (x, xp))
            kind = planted_kind(c["BN"], c["warps"], c["split"])
            if kind:
                assert not ok, (layer, m, c, "planted bug not caught by the gate")
                planted_seen.add(kind)
            elif kind is None:
                assert ok, (layer, m, c, "correct config rejected")
                paths.add((layer == "router", min(c["split"], 3)))
    assert planted_seen == {"WRONG", "PROBE"}, planted_seen
    assert paths == {(r, s) for r in (False, True) for s in (1, 2, 3)}, paths

    # 2. The lab itself.
    state = lab.prepare(w)
    new = {key: c for key, c in lab.CHOICE.items() if v1_choice.get(key) != c}
    assert set(new) == set(SWEPT), sorted(set(SWEPT) ^ set(new))
    assert all(c["kind"] == "fma" and not planted_kind(c["BN"], c["warps"], c["split"]) for c in new.values()), new
    assert all(key in lab.CHOICE for key in v1_choice), "a v1 choice was dropped"
    for (layer, m), c in lab.CHOICE.items():
        n = lab.lab_n(layer, m)
        x = lab.lab_input(m, w[layer].shape[1], 9)
        out = lab.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": x.shape[1], "layer": layer}}, state)
        assert lab.lab_ok(lab.lab_errors(out, x, w[layer][:n]), layer), (layer, m, c)
    print(f"ok: lab_e2 dry run; all launch paths correct, both planted kernels rejected; "
          f"{len(new)} shapes moved to fma")


if __name__ == "__main__":
    main()

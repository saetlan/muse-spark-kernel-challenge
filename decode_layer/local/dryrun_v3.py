"""CPU dry run of submission_v3 + the DE5 lab (lab_de5.py): v3's prepare (self-checks), its kernel_fn paths
per token count, and the lab's choices (expert configs per class, prefetch sizes per shape) installed in the
state, with every GPU kernel replaced by a shape-checking stand-in and a fake timer.

Run: .venv/bin/python decode_layer/local/dryrun_v3.py [decode_layer/lab_de5.py]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v2  # noqa: E402

DECODE = base.DECODE


def install_v3():
    real = base._Kernel.__getitem__

    def getitem(self, grid):
        name = self.fn.__name__
        if name == "_v3_router_kernel":
            def launch(x, w, out, T, N, K, TB, num_warps):
                base.hit("v3 router")
                assert grid == (N // 4,) and x.dtype == torch.float32 and TB >= T and TB in (1, 2, 4, 8)
                out.copy_(x[:T] @ w.T)
            return launch
        if name == "_v3_local_ids_kernel":
            def launch(ids, start, out, n, BLOCK, num_warps):
                base.hit("v3 lid")
                assert ids.dtype == torch.int32 and out.dtype == torch.int32 and BLOCK >= n and BLOCK & (BLOCK - 1) == 0
                out.copy_(((ids.long() - start.long()) % 256).to(torch.int32))
            return launch
        if name == "_v3_rmsnorm_kernel":
            def launch(x, w, out, eps, H, num_warps):
                base.hit("v3 norm")
                xf = x.float()
                out.copy_((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(out.dtype))
            return launch
        return real(self, grid)

    base._Kernel.__getitem__ = getitem

    class Ext:
        def m11_gemm(self, w, sf, c, ids, a, out, n_pairs, start, rt, a_per_pair, out_cols, gated, tt, nt, depth, ks, grid, pf=0):
            base.hit(f"m11 g{gated}")
            assert ids.dtype == torch.int32 and start == 0 and n_pairs == ids.numel() and grid > 0
            assert pf in (0, 1, 2, 0x11, 0x12, 0x21, 0x22)
            assert out.shape == (n_pairs, out_cols) and a.dtype == torch.bfloat16
            return out.zero_()

        def m11p_gemm(self, w, sf, c, ids, a, out, n_pairs, start, rt, a_per_pair, out_cols, gated, tt, nt, depth, ks, grid):
            # parts/m11_pdl.py: GEMM2 PDL-launched; only the M11P instances exist
            base.hit("m11p g0")
            assert gated == 0 and (tt, nt, depth, ks) in ((1, 1, 4, 4), (2, 1, 4, 4)), (gated, tt, nt, depth, ks)
            assert ids.dtype == torch.int32 and start == 0 and n_pairs == ids.numel() and grid > 0
            assert out.shape == (n_pairs, out_cols) and a.dtype == torch.bfloat16
            return out.zero_()

        def m11_post_norm_cuda(self, y, nw, tw, ids, out, tokens, start, eps, pdl):
            # parts/m11_pdl.py: M13's CUDA post-norm (zeros, as the Triton post-norm's stand-in in dryrun_v2)
            base.hit(f"m11 post cuda pdl{pdl}")
            assert pdl in (0, 1) and start == 0 and ids.dtype == torch.int32 and tw.dtype == torch.float32
            assert y.shape == (ids.numel(), 4096) and out.shape == (tokens, 4096) and nw.numel() == 4096
            return out.zero_()

        def m12_setup(self):
            raise AssertionError("v3 must not call m12_setup")

        def v3_l2_prefetch(self, a, na, b, nb):
            base.hit("prefetch")
            assert 0 <= na <= a.numel() * a.element_size() and 0 <= nb <= b.numel() * b.element_size()

    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **kw: (base.hit(f"nvcc {kw['name']}"), Ext())[1]

    # MuseSpark1xRMSNorm with SGLang's semantics (F.rms_norm with the bf16 effective weight 1 + w), so the
    # pre-expert norm self-check compares against the real formula, not an identity stub.
    norm_cls = sys.modules["sglang.srt.layers.muse_spark_layers"].MuseSpark1xRMSNorm

    def forward(self, x):
        if self.weight is None:
            return x
        w = (self.weight.float() + 1).to(torch.bfloat16)
        return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), w.float(), 1e-5).to(x.dtype)

    norm_cls.forward = forward


def main():
    base.install_stubs()
    dryrun_v2.install_more()
    install_v3()
    base.patch_torch()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de5.py"
    lab = base.load_lab(path)
    lab.ROPE_POSITIONS = 1 << 10
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    n = [0]

    def fake_time(self, fn, reps=15, cold=True):
        before = dict(base.CALLS)
        fn()
        n[0] += 1
        ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
        m11_only = any(k.startswith("m11") for k in ran) and not any(k.startswith("marlin") for k in ran)
        return (5.0 if m11_only else 20.0) + (n[0] * 7) % 13 * 0.1

    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    text = "\n".join(lines)
    v3lines = [p for p in printed if p.startswith("[v3]")]
    # paths per token count through the real kernel_fn, with the lab's tables installed
    base.CALLS.clear()
    paths = {}
    for t, (b, q, layer) in {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"), 32: (32, 1, "full"),
                             128: (16, 8, "sliding"), 256: (32, 8, "full")}.items():
        base.CALLS.clear()
        data = lab.lab_input(b, q, layer, seed=t, device="cpu")
        out = lab.kernel_fn(data, state)
        assert out[0].shape == (t, 8192) and out[2].dtype == torch.int32
        paths[t] = sorted(base.CALLS)
    for t, p in paths.items():
        print(f"T={t}: {p}")
    m11_classes = {below for below, cfg in state["m11_configs"] if cfg is not None}
    checks = {
        "all v3 self-checks on": all(": on (" in v for v in v3lines) and len(v3lines) == 6,
        "v3 builds m11 without M12 and its own ext": base.CALLS is not None and "m11" in state and "v3_ext" in state,
        "T1: router kernel, lid, norm, tiny": {"v3 router", "v3 lid", "v3 norm", "tiny"} <= set(paths[1]),
        "T16/32: no router kernel, no tiny": all("v3 router" not in paths[t] and "tiny" not in paths[t] for t in (16, 32)),
        "T128/256: Marlin": all(any(k.startswith("marlin") for k in paths[t]) and not any(k.startswith("m11") for k in paths[t])
                                for t in (128, 256)),
        "M11 classes follow the lab's table": all((any(k.startswith("m11") for k in paths[t]) ==
                                                   (min(b for b in (5, 9, 17, 33) if t < b) in m11_classes)) for t in (1, 8, 16, 32)),
        "prefetch only where the lab kept it": all(("prefetch" in paths[t]) == (state["v3_prefetch"].get(k, 0) > 0)
                                                   for t, k in ((1, (1, True)), (8, (8, False)), (16, (16, True)), (32, (32, False)))),
        "config table printed": "V3_M11_CONFIGS = (" in text and "V3_PREFETCH = {" in text,
        "state holds the printed expert table": ("V3_M11_CONFIGS = " + repr(tuple(state["m11_configs"]))) in text,
        "prefetch size per key = the lab's table": all(
            lab.v3_prefetch_bytes(state, {"tokens": k[0], "window": 2048 if k[1] else 0}) == v
            for k, v in eval(text.split("V3_PREFETCH = ")[1].split("\n")[0]).items()),
        "4 expert lines": sum(f"experts T{t} local" in text for t in (1, 8, 16, 32)) == 4,
        "summary + router lines": "layer v3/D1" in text and "router v3/F.linear" in text,
        "no failures": "failures 0" in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("router v3/F.linear"),
        "log under 12K": len(text) < 12000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v3: all checks passed")


if __name__ == "__main__":
    main()

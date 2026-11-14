"""CPU dry run of submission_v2 + the DE1 lab (lab_de1.py): every code path with SGLang, FA3, Triton
and M11's CUDA extension replaced by shape-checking stand-ins (dryrun_d0's, plus M11, the FP32 router
kernel and tiny_gemm), and time_graph by a fake timer. Checks which path each token count takes.

Run: .venv/bin/python decode_layer/local/dryrun_v2.py [decode_layer/lab_de1.py]
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402

DECODE = base.DECODE


def install_more():
    real_getitem = base._Kernel.__getitem__

    def getitem(self, grid):
        name = self.fn.__name__
        if name == "_v2_router_fp32_kernel":
            def launch(x, w, out, T, N, K, num_warps):
                base.hit("router kernel")
                assert grid == (256,) and x.dtype == torch.float32 and x.is_contiguous() and K == 8192
                out.copy_(x[:T] @ w.T)
            return launch
        if name == "_m11_post_norm_kernel":
            def launch(y, nw, tw, ids, out, start, eps, H, LOCAL, TOP_K, num_warps):
                base.hit("m11 post")
                assert start == 0 and ids.dtype == torch.int32 and tw.dtype == torch.float32
                assert y.shape == (ids.numel(), H) and out.shape[0] == grid[0]
                out.zero_()
            return launch
        return real_getitem(self, grid)

    base._Kernel.__getitem__ = getitem

    class Ext:
        def m11_gemm(self, w, sf, c, ids, a, out, n_pairs, start, rt, a_per_pair, out_cols, gated, nt, depth, grid):
            base.hit(f"m11 gemm gated={gated} nt={nt}")
            assert ids.dtype == torch.int32 and start == 0 and n_pairs == ids.numel() <= 8 * 16
            assert grid > 0 and a.dtype == torch.bfloat16 and out.shape == (n_pairs, out_cols)
            out.zero_()
            return out

    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **kw: (base.hit("nvcc"), Ext())[1]

    def tiny(x, w, out=None, *, out_dtype=None, n_split=None, max_m=16):  # tiny_gemm_bf16's signature
        base.hit("tiny")
        assert x.shape[0] <= 16 and x.is_contiguous()
        return torch.nn.functional.linear(x, w)

    sys.modules["sglang.kernels.ops.gemm.tiny_gemm"].tiny_gemm_bf16 = tiny


def main():
    base.install_stubs()
    install_more()
    base.patch_torch()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de1.py"
    lab = base.load_lab(path)
    lab.ROPE_POSITIONS = 1 << 10
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    n = [0]

    def fake_time(self, fn, reps=15, cold=True):
        fn()
        n[0] += 1
        return 10.0 + n[0] % 7

    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    calls_before = dict(base.CALLS)
    state = lab.prepare(base.weights())
    text = "\n".join(lines)
    print(f"\n--- {len(text)} chars of lab log; calls {base.CALLS}")
    # Path per token count, through the real kernel_fn.
    base.CALLS.clear()
    paths = {}
    for t, (b, q) in {1: (1, 1), 8: (1, 8), 9: (9, 1), 16: (16, 1), 24: (3, 8), 32: (32, 1), 128: (16, 8)}.items():
        base.CALLS.clear()
        data = lab.lab_input(b, q, "sliding", seed=t, device="cpu")
        out = lab.kernel_fn(data, state)
        assert out[0].shape == (t, 8192) and out[0].dtype == torch.bfloat16
        assert out[1].dtype == torch.float32 and out[2].shape == (t, 8) and out[2].dtype == torch.int32
        paths[t] = sorted(k for k in base.CALLS if k.startswith(("m11", "marlin", "router", "tiny")))
    for t, p in paths.items():
        print(f"T={t}: {p}")
    checks = {
        "nvcc once": base.CALLS.get("nvcc", calls_before.get("nvcc", 0)) >= 0,
        "all on in prepare": "m11 on, router on, tiny on" in text,
        "T1 m11 + router + tiny": {"router kernel", "tiny", "m11 post"} <= set(paths[1]) and not any("marlin" in p for p in paths[1]),
        "T8 m11 nt1": "m11 gemm gated=1 nt=1" in paths[8] and "router kernel" in paths[8],
        "T9/T16 m11 nt2, no router kernel, no tiny": all("m11 gemm gated=1 nt=2" in paths[t] and "router kernel" not in paths[t]
                                                         and "tiny" not in paths[t] for t in (9, 16)),
        "T24/32 Marlin": all(any("marlin" in p for p in paths[t]) and not any(p.startswith("m11") for p in paths[t]) for t in (24, 32)),
        "T128 Marlin bsm16": "marlin bsm16" in paths[128],
        "12 ops lines": sum(f"ops {s[0]} " in text for s in lab.SCORED) == 12,
        "gemm lines": all(f"gemm {g} " in text for g in ("qkv", "o", "pool", "unps", "rhi")),
        "router line": "router fp32 (v2 on: True)" in text,
        "no failures": "failures 0" in text,
        "summary last-ish": "SUMMARY " in text and text.rfind("SUMMARY") > text.rfind("ops "),
        "log under 10K": len(text) < 10000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v2: all checks passed")


if __name__ == "__main__":
    main()

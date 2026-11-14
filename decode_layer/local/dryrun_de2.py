"""CPU dry run of the X5 labs (default lab_de4.py = submission_v2 + parts/x5.py + lab/de4.py): v2's prepare with
dryrun_v2's stand-ins, plus an x5_gemv stand-in that rebuilds W from the packed plain fragment array
(inverse of x5_frag_maps) and accumulates x @ W^T, so the correctness gate, the timing loop and the
log are all exercised. Fake timer.

Run: .venv/bin/python decode_layer/local/dryrun_de2.py [decode_layer/lab_de4.py]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v2  # noqa: E402

DECODE = base.DECODE


def main():
    base.install_stubs()
    dryrun_v2.install_more()
    base.patch_torch()
    import torch.utils.cpp_extension as ce
    lab = None

    class Ext:
        def m11_gemm(self, *a):
            out = a[5]
            return out.zero_()

        def x5_gemv(self, wa, wi, bases, wraw, x, ws, cnt, out, N, K, T, enc, nc, nt, grid, mode=0):
            base.hit(f"x5 enc{enc} nc{nc} nt{nt} m{mode}")
            assert T <= 8 * nt <= 16 and x.shape == (T, K) and out.shape == (T, N) and out.dtype == torch.bfloat16
            assert ws.shape == (16, N) and ws.dtype == torch.float32 and cnt.shape == (N // 16,) and cnt.dtype == torch.int32
            assert wraw.shape == (N // 16, K // 16, 32, 8) and grid in (132, 264)
            assert (enc, nt, mode, nc) in lab.X5_INSTANCES, (enc, nt, mode, nc)
            if mode == 0:
                rows, cols = lab.x5_frag_maps("cpu")
                W = torch.zeros((N // 16, K // 16, 16, 16), dtype=torch.bfloat16)
                W[:, :, rows, cols] = wraw.view(torch.bfloat16)
                W = W.permute(0, 2, 1, 3).reshape(N, K)
                out.copy_((x.float() @ W.float().T).to(torch.bfloat16))
            return out

    ce.load_inline = lambda **kw: (base.hit("nvcc"), Ext())[1]
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de4.py"
    lab = base.load_lab(path)
    lab.load_inline = ce.load_inline
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
    state = lab.prepare(base.weights())
    text = "\n".join(lines)
    print(f"\n--- {len(text)} chars of lab log; calls { {k: v for k, v in base.CALLS.items() if k.startswith('x5')} }")
    checks = {
        "build line": "x5 build" in text,
        "frost line": "frost:" in text,
        "packs line": "x5 packs: qkv raw" in text and "unp raw" in text,
        "correct yes": "x5 correct: worst rel err" in text and "BAD" not in text and "left dirty 0 times" in text,
        "5 timing lines, 3 sizes each": all(f"x5 {w} [" in text for w in ("qkv", "o", "pool", "unps", "unp"))
                                         and text.count(" T16 cub ") == 5 and text.count(" load-only ") == 5,
        "every compiled instance ran": all(f"x5 enc{e} nc{nc} nt{nt} m{m}" in base.CALLS for e, nt, m, nc in lab.X5_INSTANCES),
        "no failures": "failures 0" in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("x5 unp ["),
        "log under 12K": len(text) < 12000,
        "state returned": isinstance(state, dict) and "m11" in state,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run de2: all checks passed")


if __name__ == "__main__":
    main()

"""CPU check of proj/submission_v12.py (v11b + F12M part[] padded to MT + 3): prepare()
with Triton's dry-run stub and dgrs_ext mocked with v10's 10-field dispatch (a call must match a built instance exactly on
every set field), then the judge's check_implementation on every chosen shape and two input distributions; the two m32 shapes
must really go through dgrs_ext.

Run: .venv/bin/python proj/local/check_v11b.py
"""

import io
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import check_make_dgrs as cm  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e15  # noqa: E402


def mock_v10(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "dgrs_ext":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        insts = {tuple(int(v) for v in t) for t in
                 re.findall(r"DGRS_CASE\((\d+), (\d+), (\d+), (\d+), (-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\)", cuda_sources)}
        assert insts == {(32, 1, 0, 4, 2, 2, 8, 32, 8192, 4096), (32, 1, 0, 4, 1, 4, 8, 32, 4096, 8192)}, insts
        mod = types.SimpleNamespace()

        def gemm(wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red, math):
            hit = [t for t in insts if t[:4] == (bm, cps, diag, math) and (t[4] < 0 or t[4] == red) and t[5] in (0, splits)
                   and t[6] in (0, smax) and t[7] in (0, M) and t[8] in (0, N) and t[9] in (0, K)]
            assert hit, (bm, cps, diag, math, red, splits, smax, M, N, K)
            assert red != 1 or (part.numel() >= splits * bm * N and cnt.numel() >= N // 128), "red 1 buffers"
            calls[(N, K)] = calls.get((N, K), 0) + 1
            if id(wp) not in cache:
                cache[id(wp)] = dryrun_lab_e15.decode_natural(wp.permute(1, 0, 2, 3, 4).contiguous(), rec, K, dec)
            out.copy_((x.double() @ cache[id(wp)][:N].T).to(torch.bfloat16))
        mod.dgrs_gemm = gemm
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    cm.setup()
    calls = {}
    mock_v10(calls)
    w = d.weights()
    mod = d.load(d.PROJ / "submission_v12.py", "v12_cpu")
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    assert mod.DGRS_EXT is not None
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(mod.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 4500 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    assert calls.get((8192, 4096), 0) >= 2 and calls.get((4096, 8192), 0) >= 2, calls
    print(f"ok: v12 CPU check, judge's check on all {len(mod.CHOICE)} chosen shapes x 2 inputs; unpooler m32 and pooler m32 "
          f"ran the fully specialized DG12-RS instances ({calls})")


if __name__ == "__main__":
    main()

"""CPU test of proj/local/make_dgrs.py: build a scored file from scored/submission_v10.py with a few DG12-RS shapes (BM 32,
64 and 128), run its prepare() with dgrs_ext mocked (dryrun_lab_e38.mock_dgrs9, v9 API incl. specialized instances: x @ W from the chunk-major planes; any
specialized red 0 / 3-split config drops its last 64 k), and check kernel_fn with the judge's check on every chosen shape: the dgrs shapes must
go through the mock (no silent fallback) and pass. A second file with a specialized pooler S3 red 0 shape must FAIL that shape's check.

Run: .venv/bin/python proj/local/check_make_dgrs.py <scratch dir>
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402
import dryrun_lab_e38 as e38  # noqa: E402
import make_dgrs  # noqa: E402

GOOD = {("unpooler", 32): {"kind": "dgrs", "BM": 32, "S": 2, "CPS": 1, "RED": 2, "MATH": 4, "DIAG": 512, "SPEC": True},
        ("pooler", 32): {"kind": "dgrs", "BM": 32, "S": 3, "CPS": 1, "RED": 2, "MATH": 4, "DIAG": 512},
        ("o", 32): {"kind": "dgrs", "BM": 32, "S": 2, "CPS": 1, "RED": 0, "MATH": 4, "DIAG": 0}}
BAD = {**GOOD, ("pooler", 32): {"kind": "dgrs", "BM": 32, "S": 3, "CPS": 1, "RED": 0, "MATH": 4, "DIAG": 512, "SPEC": True}}


def setup():
    d.install_stubs()
    e18.PLANT["on"] = False
    d.EMULATE["_hl_gemm_kernel"] = e18._hl_gemm
    d.EMULATE["_hl_swap_kernel"] = e18._hl_swap
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
    calls, built = {}, {}
    e38.mock_dgrs9(calls, built, strict=False)
    return calls


def run(path, name, chosen, calls, w):
    make_dgrs.build(path, "scored/submission_v10.py", chosen)
    mod = d.load(Path(path), name)
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    assert mod.DGRS_EXT is not None and set(mod.DGRS_CHOICE) == set(chosen)
    sys.path.insert(0, str(d.PROJ))
    import reference
    bad = []
    for (layer, m) in sorted(mod.CHOICE):
        before = dict(calls)
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 1700 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
            if not ok:
                bad.append((layer, m, dist))
        if (layer, m) in chosen:
            c = chosen[(layer, m)]
            key = ("dgrs_ext", c["BM"], c["S"], c["RED"], c["MATH"], c.get("DIAG", 0))
            assert calls.get(key, 0) > before.get(key, 0), f"{layer} m{m} did not run dgrs {key}"
    return bad


def main():
    scratch = Path(sys.argv[1])
    calls = setup()
    w = d.weights()
    bad = run(scratch / "dgrs_good.py", "dgrs_good", GOOD, calls, w)
    assert not bad, bad
    bad = run(scratch / "dgrs_bad.py", "dgrs_bad", BAD, calls, w)
    assert bad and {(layer, m) for layer, m, _ in bad} == {("pooler", 32)}, bad
    print(f"ok: make_dgrs scored file passes the judge's check on every shape ({len(GOOD)} through dgrs); "
          f"the planted specialized pooler S3 red 0 shape fails it ({len(bad)} of 2 inputs)")


if __name__ == "__main__":
    main()

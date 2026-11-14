"""CPU test of proj/local/make_hlp.py: build a scored file from scored/submission_v10.py with the 4 router shapes on hlp (v10's
tiles), run its prepare() with Triton's dry-run stub (zero kernel + PDL GEMM emulations from dryrun_lab_e32), check that the
router shapes really go through the PDL kernels (zero kernel before each GEMM, launch_pdl=True) and that the judge's check
passes on every chosen shape; a second file with hlpt at half the splits (planted wrong by the emulator) must fail them.

Run: .venv/bin/python proj/local/check_make_hlp.py <scratch dir>
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e18 as e18  # noqa: E402
import dryrun_lab_e32 as e32  # noqa: E402
import make_hlp  # noqa: E402

V10_HL = {16: {'kind': 'hl_gemm', 'BM': 16, 'BN': 32, 'BK': 128, 'split': 32, 'warps': 2, 'stages': 3},
          32: {'kind': 'hl_swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'warps': 4, 'stages': 4},
          128: {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'warps': 8, 'stages': 3},
          256: {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 64, 'split': 16, 'warps': 8, 'stages': 3}}
GOOD = {("router", m): {"kind": "hlp", "base": dict(b)} for m, b in V10_HL.items()}
BAD = {("router", m): {"kind": "hlpt", "base": dict(b, split=b["split"] // 2)} for m, b in V10_HL.items()}


def setup():
    d.install_stubs()
    extra = types.ModuleType("triton.language.extra")
    cuda = types.ModuleType("triton.language.extra.cuda")
    cuda.gdc_wait = cuda.gdc_launch_dependents = lambda *a, **k: None
    extra.cuda = cuda
    sys.modules["triton.language"].extra = extra
    sys.modules["triton.language.extra"], sys.modules["triton.language.extra.cuda"] = extra, cuda
    e18.PLANT["on"] = False
    d.EMULATE["_hl_gemm_kernel"] = e18._hl_gemm
    d.EMULATE["_hl_swap_kernel"] = e18._hl_swap
    d.EMULATE["_hlp_zero_kernel"] = e32.zero
    d.EMULATE["_hlp_gemm_kernel"] = e32.hlp_gemm
    d.EMULATE["_hlp_swap_kernel"] = e32.hlp_swap
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


def run(path, name, chosen, w):
    make_hlp.build(path, "scored/submission_v10.py", chosen)
    mod = d.load(Path(path), name)
    zeros = mod.torch.zeros

    def zeros_marked(*a, **k):
        e32.LAUNCHES.append("torch.zeros")
        return zeros(*a, **k)
    mod.torch.zeros = zeros_marked
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    for k in chosen:
        assert mod.CHOICE[k]["kind"] in ("hlp", "hlpt") and mod.CHOICE[k]["fallback"] == V10_HL[k[1]], (k, mod.CHOICE[k])
    sys.path.insert(0, str(d.PROJ))
    import reference
    bad = []
    for (layer, m) in sorted(mod.CHOICE):
        e32.LAUNCHES.clear()
        data = reference.generate_input(m, layer, 2500 + m, dist="normal", device="cpu")
        ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
        if not ok:
            bad.append((layer, m))
        if (layer, m) in chosen:
            want = ["zero" if chosen[(layer, m)]["kind"] == "hlp" else "torch.zeros",
                    "swap" if chosen[(layer, m)]["base"]["kind"] == "hl_swap" else "gemm"]
            assert e32.LAUNCHES[-2:] == want, ((layer, m), e32.LAUNCHES)
    mod.torch.zeros = zeros
    return bad


def main():
    scratch = Path(sys.argv[1])
    setup()
    w = d.weights()
    bad = run(scratch / "hlp_good.py", "hlp_good", GOOD, w)
    assert not bad, bad
    bad = run(scratch / "hlp_bad.py", "hlp_bad", BAD, w)
    assert set(bad) == set(BAD), bad
    print(f"ok: make_hlp scored file passes the judge's check on every shape ({len(GOOD)} router shapes through the zero kernel "
          f"+ PDL GEMM); the planted half-split hlpt file fails exactly its {len(bad)} router shapes")


if __name__ == "__main__":
    main()

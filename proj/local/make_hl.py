"""Build a scored file with the router hi/lo Triton kernels (P37): a base file byte for byte (e.g.
scored/submission_v7a.py) plus the HL section: lab/e18.py's kernels, hl_setup, run_hl and run_choice wrapper
(the text between the first @triton.jit and def hl_cands, unchanged), the chosen configs put into CHOICE, and
_warm wrapped so hl_setup runs first and the chosen hl kernels compile in prepare(), before any capture.

Run: .venv/bin/python proj/local/make_hl.py --base scored/submission_v7a.py out.py   (configs: HL_CHOSEN below)
"""

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " HL (proj/lab/e18.py): router with pre-split BF16 hi/lo in Triton\n\n"

# PE18's "V14 CHOICE" lines (job ce3ec579c4c743ff).
HL_CHOSEN = {
    ('router', 32): {'kind': 'hl_swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'warps': 4, 'stages': 4},
    ('router', 128): {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'warps': 8, 'stages': 3},
    ('router', 256): {'kind': 'hl_gemm', 'BM': 256, 'BN': 32, 'BK': 64, 'split': 16, 'warps': 8, 'stages': 3},
}

TAIL = '''

for _key, _c in HL_CHOICE.items():
    CHOICE[_key] = dict(_c)

_warm_before_hl = _warm


def _warm(state):
    """prepare() ends with _warm(state): the router's hi/lo copies first, then every chosen kernel once, so the hl
    kernels compile here, before any capture (prepare itself stays the base's, untouched)."""
    hl_setup(state)
    _warm_before_hl(state)
'''


def section_text():
    src = (PROJ / "lab" / "e18.py").read_text()
    start = src.index('@triton.jit(do_not_specialize=["M"])\ndef _hl_gemm_kernel')
    end = src.index("\ndef hl_cands(")
    return src[start:end].rstrip() + "\n"


def build(out, base, chosen):
    assert chosen and all(k[0] == "router" and c["kind"] in ("hl_gemm", "hl_swap") for k, c in chosen.items()), chosen
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert src.count("\ndef prepare(weights") == 1 and "\ndef _warm(state)" in src and "\ndef run_choice(" in src
    assert "_hl_gemm_kernel" not in src, f"{base} already has the HL section"
    body = section_text() + f"\n\nHL_CHOICE = {dict(chosen)!r}\n" + TAIL
    Path(out).write_text(src + SEPARATOR + body)
    print(f"wrote {out} ({len(chosen)} router shapes on hl kernels: {sorted(m for _, m in chosen)})")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = None
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    assert base and args, __doc__
    build(args[0], base, HL_CHOSEN)

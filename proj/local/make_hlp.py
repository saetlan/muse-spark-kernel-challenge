"""Build a scored file with the router's PDL path (P42, lab/hlp.py) on chosen shapes: a base file byte for byte (e.g.
scored/submission_v10.py) plus the HLP section (lab/hlp.py: the zero kernel with gdc_launch_dependents, the PDL GEMMs
with gdc_wait before their atomics, run_hlp and its run_choice hook) and the chosen entries (each with the base's entry
as fallback). The base's _warm already launches every chosen kernel once in prepare(), before any capture.

Run: .venv/bin/python proj/local/make_hlp.py --base scored/submission_v10.py out.py   (configs: HLP_CHOSEN below, from
PE32's "V25 CHOICE" lines; each {"kind": "hlp" | "hlpt", "base": <HL config with split > 1>})
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " HLP (proj/lab/hlp.py): router split-K with programmatic dependent launch\n\n"

# Filled from PE32's "V25 CHOICE" lines.
HLP_CHOSEN = {}


def build(out, base, chosen):
    assert chosen, "HLP_CHOSEN is empty"
    for (layer, m), c in chosen.items():
        b = c["base"]
        assert layer == "router" and c["kind"] in ("hlp", "hlpt") and b["kind"] in ("hl_gemm", "hl_swap") and b["split"] > 1, c
        assert 8192 % (b["split"] * b["BK"]) == 0, c
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\ndef run_hl(c, x, layer, n, state)" in src and "\ndef run_choice(" in src and "_hlp_zero_kernel" not in src
    hlp = (PROJ / "lab" / "hlp.py").read_text()
    assert "\nimport triton.language.extra.cuda as tlc" in hlp
    section = ("import triton.language.extra.cuda as tlc   # a module: Triton 3.8 global_lookup accepts it (F24)\n\n"
               + body(PROJ / "lab" / "hlp.py")
               + "\n\nHLP_CHOICE = " + repr({k: {kk: vv for kk, vv in c.items() if kk != "fallback"} for k, c in chosen.items()})
               + "\n\nfor _key, _c in HLP_CHOICE.items():\n    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))\n")
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(chosen)} router shapes on HLP)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = None
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    assert base and args, __doc__
    build(args[0], base, HLP_CHOSEN)

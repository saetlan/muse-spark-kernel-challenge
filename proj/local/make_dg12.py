"""Build a scored file with DG12 (the 12-bit fork of DeepGEMM's sm90 BF16 GEMM, P40) on chosen shapes: a base file byte for
byte (e.g. scored/submission_v10.py) plus the DG12 section: proj/dg/dg12_layout.py and proj/dg/dg12.py (code only), the
chosen configs (each with the base's kernel as fallback), and _warm wrapped so prepare() packs the layers' 12-bit planes
and builds the chosen instances (in parallel, before any capture) ahead of the base's warm-up. A layer that can't be
packed, or an instance that fails to build, falls back to the base's kernel for its shapes.

Run: .venv/bin/python proj/local/make_dg12.py --base scored/submission_v10.py out.py   (configs: DG12_CHOSEN below)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " DG12 (proj/dg): 12-bit weights in DeepGEMM's sm90 GEMM, m >= 32\n\n"

# Filled from PE24's "V19 CHOICE" lines.
DG12_CHOSEN = {}

TAIL = '''

DG12_MOD = None
DG12_OK = set()
DG12_BUILD_TIMEOUT_S = 150.0


def dg12_inst(c, n, k):
    return (c["BM"], c["BN"], c["ST"], c["SWD"], c["NMATH"], n, k)


def dg12_setup(state):
    """Pack the 12-bit planes of every layer with a DG12 shape, build the chosen instances; failures fall back."""
    global DG12_MOD
    state["dg12"] = {}
    for layer in sorted({layer for (layer, m) in DG12_CHOICE}):
        try:
            pk = dg12_pack(state[layer])
        except Exception as exc:
            print(f"dg12 pack {layer} FAILED: {type(exc).__name__}", flush=True)
            pk = None
        if pk is not None:
            state["dg12"][layer] = {"qc": pk["qc"], "lc": pk["lc"], "scale": pk["scale"], "et": pk["et"]}
    insts = tuple(dict.fromkeys(dg12_inst(c, state[layer].shape[0], state[layer].shape[1])
                                for (layer, m), c in DG12_CHOICE.items() if layer in state["dg12"]))
    if not insts:
        return
    try:
        mod, errors = dg12_build(insts, timeout=DG12_BUILD_TIMEOUT_S)
    except Exception as exc:
        print(f"dg12 build FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-300:]}", flush=True)
        return
    for name, exc in errors.items():
        print(f"dg12 {name} FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-300:]}", flush=True)
    ok = set()
    for inst in insts:
        name = dg12_name(inst)
        if name in errors:
            continue
        try:
            getattr(mod, name + "_attr")()
            ok.add(name)
        except Exception as exc:
            print(f"dg12 {name} attr FAILED: {type(exc).__name__}", flush=True)
    if ok:
        DG12_MOD = mod
        DG12_OK.update(ok)


def run_dg12(c, x, layer, n, state):
    m, k = x.shape
    pk = state.get("dg12", {}).get(layer)
    if DG12_MOD is None or pk is None or dg12_name(dg12_inst(c, n, k)) not in DG12_OK or n != state[layer].shape[0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    offs, ent = pk["et"][c["BN"]]
    getattr(DG12_MOD, dg12_name(dg12_inst(c, n, k)))(x.contiguous(), pk["qc"], pk["lc"], pk["scale"], offs, ent, out)
    return out


_run_choice_pre_dg12 = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dg12":
        return run_dg12(c, x, layer, n, state)
    return _run_choice_pre_dg12(c, x, layer, n, state)


for _key, _c in DG12_CHOICE.items():
    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))

_warm_before_dg12 = _warm


def _warm(state):
    """prepare() ends with _warm(state): DG12's planes and instances first, then every chosen kernel once."""
    dg12_setup(state)
    _warm_before_dg12(state)
'''


def build(out, base, chosen):
    assert chosen and all(c["kind"] == "dg12" and key[1] >= 32 for key, c in chosen.items()), chosen
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert src.count("\ndef prepare(weights") == 1 and "\ndef _warm(state)" in src and "\ndef run_choice(" in src
    assert "dg12_gemm_impl" not in src, f"{base} already has the DG12 section"
    assert "\nimport os\n" in src or True
    body_text = ("import os\n\n" + body(PROJ / "dg" / "dg12_layout.py") + "\n\n" + body(PROJ / "dg" / "dg12.py")
                 + "\n\nDG12_CHOICE = " + repr({k: {kk: vv for kk, vv in c.items() if kk != "fallback"} for k, c in chosen.items()})
                 + "\n" + TAIL)
    Path(out).write_text(src + SEPARATOR + body_text)
    print(f"wrote {out} ({len(chosen)} shapes on DG12)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = None
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    assert base and args, __doc__
    build(args[0], base, DG12_CHOSEN)

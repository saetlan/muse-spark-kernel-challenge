"""Build a scored file with DG12-RS (P41: shared/gemm/dg12rs/dgrs.py by muse_prefill_layer) on chosen shapes: a base file
byte for byte (e.g. scored/submission_v10.py) plus the DGRS section: dgrs.py (code only, with its load_inline import), the
chosen configs (each with the base's kernel as fallback), and _warm wrapped so prepare() makes the chunk-major 12-bit planes
of the chosen layers, builds one dgrs_ext with only the chosen token-tile widths and allocates the split-K scratch of each
chosen (layer, m) before the base's warm-up (which launches every chosen kernel once, outside any capture). A layer whose
planes can't be made, or a failed build, leaves the base's kernel on its shapes.

Run: .venv/bin/python proj/local/make_dgrs.py --base scored/submission_v10.py out.py   (configs: DGRS_CHOSEN below,
from the DG12-RS lab's CHOICE lines; each {"kind": "dgrs", "BM": .., "S": .., "CPS": 1|2, "RED": 0|1, "MATH": 2|4}; dgrs.py
v4+ API)
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DGRS = PROJ / "dg" / "pinned" / "dgrs_v9.py"   # the exact v9 PE38 / PE39 measured (sha256 c8b76b59...)
DGRS_V10 = PROJ / "dg" / "pinned" / "dgrs_v10.py"   # v10 (sha256 f4ff43fa...): needed only for MOD "full" configs (PE40)
PINNED = {"dgrs_v9.py": "c8b76b59502806b8", "dgrs_v10.py": "f4ff43fafffd5af2"}
SEPARATOR = "\n\n# " + "=" * 70 + " DG12-RS (shared/gemm/dg12rs, P41): 12-bit weights into register-A wgmma, m >= 32\n\n"
N_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192}

# Filled from PE31's "V24 CHOICE" lines.
DGRS_CHOSEN = {}

TAIL = '''

DGRS_EXT = None
DGRS_STATE = {}


def dgrs_key(c, layer, m):
    return (layer, m, c["BM"], c["S"], c.get("CPS", 1), c.get("RED", 0), c.get("MATH", 2), c.get("DIAG", 0))


def dgrs_inst(c, layer, m):
    t = (c["BM"], c.get("CPS", 1), c.get("DIAG", 0), c.get("MATH", 2))
    if c.get("MOD") == "full":
        return t + (c.get("RED", 0), c["S"], 8, c["M"], c["N"], c["K"])
    if c.get("SPEC") or c.get("MOD") == "spec":
        return t + (c.get("RED", 0), c["S"])
    return t + (-1, 0)


def dgrs_setup(state):
    """Planes of every layer with a DGRS shape, one build with the chosen BMs, scratch per chosen (layer, m)."""
    global DGRS_EXT
    planes = {}
    for layer in sorted({layer for (layer, m) in DGRS_CHOICE}):
        try:
            pl = f12m_device_planes(state[layer], natural=True)
        except Exception as exc:
            print(f"dgrs planes {layer} FAILED: {type(exc).__name__}", flush=True)
            pl = None
        if pl is not None:
            planes[layer] = {"wp": dgrs_pack(pl["p"]), "rec": pl["rec"]}
    if not planes:
        return
    try:
        ext = dgrs_build(tuple(sorted({dgrs_inst(c, layer, m) for (layer, m), c in DGRS_CHOICE.items()})))
    except Exception as exc:
        print(f"dgrs build FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-300:]}", flush=True)
        return
    bufs = {}
    for (layer, m), c in DGRS_CHOICE.items():
        if layer in planes:
            bufs[dgrs_key(c, layer, m)] = dgrs_buffers(m, state[layer].shape[0], c["BM"], c["S"], state[layer].device)
    DGRS_STATE["planes"] = planes
    DGRS_STATE["bufs"] = bufs
    DGRS_EXT = ext


def run_dgrs(c, x, layer, n, state):
    m, k = x.shape
    pl = DGRS_STATE.get("planes", {}).get(layer)
    bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
    if DGRS_EXT is None or pl is None or bufs is None or n != state[layer].shape[0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    return dgrs_run(DGRS_EXT, x.contiguous(), pl["wp"], pl["rec"], out, bufs, c["BM"], c["S"],
                    cps=c.get("CPS", 1), red=c.get("RED", 0), math=c.get("MATH", 2), diag=c.get("DIAG", 0))


_run_choice_pre_dgrs = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dgrs":
        return run_dgrs(c, x, layer, n, state)
    return _run_choice_pre_dgrs(c, x, layer, n, state)


for _key, _c in DGRS_CHOICE.items():
    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))

_warm_before_dgrs = _warm


def _warm(state):
    """prepare() ends with _warm(state): DG12-RS's planes, build and scratch first, then every chosen kernel once."""
    dgrs_setup(state)
    _warm_before_dgrs(state)
'''


def build(out, base, chosen):
    assert chosen, "DGRS_CHOSEN is empty"
    full = any(c.get("MOD") == "full" for c in chosen.values())
    src_file = DGRS_V10 if full else DGRS
    for (layer, m), c in chosen.items():
        cps, red = c.get("CPS", 1), c.get("RED", 0)
        assert c["kind"] == "dgrs" and m >= 32 and c["BM"] in (32, 64, 128) and 1 <= c["S"] <= 8, (layer, m, c)
        mth = c.get("MATH", 2)
        assert cps in (1, 2) and (cps == 1 or c["BM"] <= 64) and red in (0, 1, 2), (layer, m, c)
        assert mth in (2, 4) and (mth == 2 or (c["BM"] <= 64 and cps == 1)), (layer, m, c)
        dg = c.get("DIAG", 0)
        assert dg in (0, 512), (layer, m, c)   # v9: 512 = ring corrections (256's dry pass measured slower, PE36)
        kc = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096}[layer] // 64
        assert dg == 0 or -(-kc // c["S"]) <= 256, (layer, m, c)
        assert (c["S"] - 1) * -(-kc // c["S"]) < kc, (layer, m, c)
        assert N_ROWS[layer] // 128 * -(-m // c["BM"]) * c["S"] <= 132 * cps, (layer, m, c)
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert src.count("\ndef prepare(weights") == 1 and "\ndef _warm(state)" in src and "\ndef run_choice(" in src
    assert "\ndef f12m_device_planes(w, natural=False)" in src and "rs_kernel" not in src, f"{base}: no F12M planes or DGRS present"
    sha = hashlib.sha256(src_file.read_bytes()).hexdigest()
    assert sha.startswith(PINNED[src_file.name]), f"{src_file.name} is not the pinned copy: {sha[:16]}"
    section = ("from torch.utils.cpp_extension import load_inline\n\n" + body(src_file)
               + f"\n\n# proj/dg/pinned/{src_file.name} (shared/gemm/dg12rs) sha256 {sha[:16]}\n"
               + "DGRS_CHOICE = " + repr({k: {kk: vv for kk, vv in c.items() if kk != "fallback"} for k, c in chosen.items()})
               + "\n" + TAIL)
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(chosen)} shapes on DG12-RS; dgrs.py sha256 {sha[:16]})")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = None
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    assert base and args, __doc__
    build(args[0], base, DGRS_CHOSEN)

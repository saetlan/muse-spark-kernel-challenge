"""Build a WG submission file: a base file byte for byte plus the WG section (proj/wg/wg.py without its
docstring and imports, with the wgmma asm rendered in, the setup and run functions below) and
WG_INSTANCES set to the given instance list.

Run: .venv/bin/python proj/local/make_wg.py --base submission_v5.py [--instances LAB] [--lab] out.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " WG (proj/wg): wgmma GEMM with in-cluster split-K, m = 128/256\n\n"

SETUP = '''
WG_ENABLED = True
WG_MOD = None


def wg_setup(state):
    """Build every WG/WGF instance and allow its dynamic shared memory; WGF also packs the BF16 layers into
    F12M planes in natural k order. A failed build leaves no module, so every WG/WGF shape runs its
    fallback (slower, still correct)."""
    global WG_MOD
    state["wgf"] = {}
    if WGF_INSTANCES:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            planes = f12m_device_planes(state[name], natural=True)
            if planes is not None:
                state["wgf"][name] = planes
    try:
        WG_MOD = wg_build(WG_INSTANCES, WGF_INSTANCES)
        for inst in WG_INSTANCES:
            getattr(WG_MOD, wg_name(inst) + "_attr")()
        for inst in WGF_INSTANCES:
            getattr(WG_MOD, wgf_name(inst) + "_attr")()
    except Exception as exc:
        print(f"wg build FAILED, WG shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        WG_MOD = None


def run_wg(c, x, layer, n, state):
    """c: dict(kind="wg", KS, ST, TT, fallback). BF16 layers, m <= 128 TT."""
    m, k = x.shape
    if WG_MOD is None or m > 128 * c["TT"] or state[layer].dtype != torch.bfloat16:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(WG_MOD, wg_name((c["KS"], c["ST"], c["TT"])))(state[layer], x.contiguous(), out, n)
    return out


def run_wgf(c, x, layer, n, state):
    """c: dict(kind="wgf", KS, ST, TT, D, fallback). 12-bit planes (natural k order), m <= 128 TT."""
    m, k = x.shape
    pl = state.get("wgf", {}).get(layer)
    if WG_MOD is None or pl is None or m > 128 * c["TT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(WG_MOD, wgf_name((c["KS"], c["ST"], c["TT"], c["D"])))(pl["p"], pl["rec"], x.contiguous(), out, n)
    return out
'''


def build(out, base, instances="LAB", lab=False, wgf_instances=None):
    sys.path.insert(0, str(PROJ / "wg"))
    import wg as wgmod
    import wg_configs
    insts = getattr(wg_configs, instances)
    winsts = getattr(wg_configs, wgf_instances) if wgf_instances else ()
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert "\nWG_ENABLED = False\n" in src, f"{base} has no WG hook (rebuild it from submission.py)"
    if winsts:
        assert "\ndef f12m_device_planes(" in src, f"{base} lacks the F12M section WGF needs"
        assert 'if c["kind"] == "wgf":' in src, f"{base} has no WGF dispatch hook (rebuild it from submission.py)"
    text = body(PROJ / "wg" / "wg.py").replace("WG_INSTANCES = ()", f"WG_INSTANCES = {tuple(insts)!r}")
    text = text.replace("WGF_INSTANCES = ()", f"WGF_INSTANCES = {tuple(winsts)!r}")
    assert "@@WGMMA128@@" in text and "@@WGMMARS128@@" in text
    section = text + SETUP
    if "import tempfile" not in src:
        section = "import tempfile\n\n" + section
    if lab:
        cands = {}
        for layer in wg_configs.SHAPES:
            for m in wg_configs.MS:
                gen = wg_configs.candidates_deep if instances == "LAB_DEEP" else wg_configs.candidates
                keep = [dict(c, TT=wg_configs.tiles(m)) for c in gen(layer, m) if wg_configs.instance(c, m) in insts]
                if keep:
                    cands[(layer, m)] = keep
        section += f"\n\nWG_LAB_CANDIDATES = {cands!r}\n"
        wcands = {}
        for layer in wg_configs.SHAPES:
            for m in wg_configs.WGF_MS:
                keep = [dict(c, TT=wg_configs.tiles(m)) for c in wg_configs.wgf_candidates(layer, m)
                        if wg_configs.wgf_instance(c, m) in winsts]
                if keep:
                    wcands[(layer, m)] = keep
        section += f"\nWGF_LAB_CANDIDATES = {wcands!r}\n"
    assert wgmod.wgmma_asm(128) and wgmod.wgmma_rs_asm(128)
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(insts)} WG, {len(winsts)} WGF instances)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base, inst, lab, winst = None, "LAB", False, None
    while args and args[0].startswith("--"):
        if args[0] == "--base":
            base, args = args[1], args[2:]
        elif args[0] == "--instances":
            inst, args = args[1], args[2:]
        elif args[0] == "--wgf":
            winst, args = args[1], args[2:]
        elif args[0] == "--lab":
            lab, args = True, args[1:]
    assert base, "--base is required"
    build(args[0], base, inst, lab, winst)

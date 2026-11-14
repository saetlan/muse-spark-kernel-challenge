"""Build proj/submission_v16.py = proj/submission_v15.py byte for byte + one section: P9 with the two-phase cluster reduction
(PE52 / PE57: each CTA sums its own KSW parts in place, then the cross-CTA pass reads CS values per output instead of CS x KSW)
built from v15's own P9 source by two exact edits as module okc_proj_p9l, on router m8 and router m16 (kind p9l, fallback v15's
entries: P9 at m8, the HL Triton kernel at m16). _warm is wrapped so prepare() builds the module (and sets its shared-memory
attribute) before the base's warm-up launches every chosen kernel once. PE57 (job bd1ae73f) timed exactly this kernel_fn.

Run: .venv/bin/python proj/local/make_v16.py
"""

import hashlib
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v15.py"
SEPARATOR = "\n\n# " + "=" * 70 + " P9L (two-phase cluster reduction, PE52 / PE57): router m8 / m16\n\n"


def section():
    e57 = (PROJ / "lab" / "e57.py").read_text()
    edits = e57[e57.index("PL_SUM_OLD = ("):e57.index("P9L_MOD = None")]
    src_fn = e57[e57.index("def p9l_source(instances):"):e57.index("def p9l_build(instances):")]
    run_fn = e57[e57.index("def run_p9l("):e57.index("_cfg_name_pre_pl = cfg_name")]
    targets = e57[e57.index("PL_TARGETS = {"):e57.index("def sec_pl(")]
    assert "def p9l_decls(instances):" in src_fn and "_run_choice_pre_pl = run_choice" in run_fn
    return (edits + "P9L_MOD = None\n\n\n" + src_fn + '''def p9l_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = p9l_decls(instances)
    return load_inline(name="okc_proj_p9l", cpp_sources="\\n".join(decls), cuda_sources=p9l_source(instances),
                       functions=[d.split("(")[0].split()[-1] for d in decls],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                       build_directory=tempfile.mkdtemp(prefix="okc_proj_p9l_"), verbose=False)


''' + run_fn + targets + '''
for _key, _c in PL_TARGETS.items():
    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))

_warm_before_pl = _warm


def _warm(state):
    """prepare() ends with _warm(state): the two-phase P9 module first (a failed build leaves router m8 / m16 on their
    fallbacks), then every chosen kernel once."""
    global P9L_MOD
    if "router" in state.get("p9", {}):
        try:
            mod = p9l_build(P9_INSTANCES)
            for _inst in P9_INSTANCES:
                getattr(mod, p9_name(_inst).replace("p9_run_", "p9l_run_") + "_attr")(P9_MAX_SMEM)
            P9L_MOD = mod
        except Exception as exc:
            print(f"p9l build FAILED, router m8 / m16 fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
            P9L_MOD = None
    _warm_before_pl(state)
''')


def build(out=PROJ / "submission_v16.py"):
    src = (PROJ / BASE).read_text()
    assert hashlib.sha256(src.encode()).hexdigest().startswith("20c1d624"), "base is not v15"
    assert "p9l" not in src
    out_src = src.rstrip() + "\n" + SEPARATOR + section()
    Path(out).write_text(out_src)
    print(f"wrote {out} sha256 {hashlib.sha256(out_src.encode()).hexdigest()[:16]} (v15 + two-phase P9 on router m8 / m16)")


if __name__ == "__main__":
    build()

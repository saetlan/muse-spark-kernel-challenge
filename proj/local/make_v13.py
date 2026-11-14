"""Build proj/submission_v13.py = proj/submission_v12.py (v11b + F12M part[] padded, PE45) byte for byte + one section:
  - F12MH (lab/f12mh.py, PE46): F12M with a helper warp that precomputes the escape corrections, built as its own module with
    only the chosen configs' instances, on MH_CHOSEN shapes (fallback: v12's entry);
  - extra padded F12M instances appended to F12M_INSTANCES (built by v12's own f12m_setup) for MQ_CHOSEN shapes (PE47), which
    run as plain kind "f12m" (fallback: v12's entry, SGLang's path when it had none).
_warm is wrapped so prepare() builds the F12MH module before the base's warm-up launches every chosen kernel once.

Run: .venv/bin/python proj/local/make_v13.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "submission_v12.py"
K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096}
# PE46's "MH CHOICE" lines
MH_CHOSEN = {("o", 8): {"RTW": 2, "RG": 1, "KSW": 8, "D": 1, "F12": True, "NT": 1},
             # o m1: o m8's config (PE47: padded 9.36 vs fma 9.57 us; the helper added 3.8% at m8), same instance
             ("o", 1): {"RTW": 2, "RG": 1, "KSW": 8, "D": 1, "F12": True, "NT": 1},
             ("qkv_gate", 16): {"RTW": 1, "RG": 1, "KSW": 8, "D": 2, "F12": True, "NT": 2}}
# PE47's "MQ CHOICE" lines (kind f12m on v12's padded module)
MQ_CHOSEN = {}
SEPARATOR = "\n\n# " + "=" * 70 + " F12MH (helper-warp escape corrections, PE46) and extra padded F12M configs (PE47)\n\n"

TAIL = '''

F12MH_MOD = None


def f12mh_decls(instances):
    return [d.replace("f12m_run_", "f12mh_run_") for d in f12m_decls(instances)]


def f12mh_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12mh_decls(instances)
    return load_inline(name="okc_proj_f12mh", cpp_sources="\\n".join(decls), cuda_sources=f12mh_source(instances, f12m_source),
                       functions=[d.split("(")[0].split()[-1] for d in decls],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                       build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mh_"), verbose=False)


def f12m_inst(c, k):
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    return inst + ((cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ()))


def run_f12mh(c, x, layer, n, state):
    """run_f12m on the helper-warp module; without it (or planes) the shape's previous kernel."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MH_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MH_MOD, f12m_name(f12m_inst(c, k)).replace("f12m_run_", "f12mh_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_mh = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mh":
        return run_f12mh(c, x, layer, n, state)
    return _run_choice_pre_mh(c, x, layer, n, state)


F12M_INSTANCES = tuple(dict.fromkeys(F12M_INSTANCES + MQ_INSTANCES))
for _key, _c in MQ_CHOICE.items():
    CHOICE[_key] = dict(_c, kind="f12m", fallback=CHOICE.get(_key))
for _key, _c in MH_CHOICE.items():
    CHOICE[_key] = dict(_c, kind="f12mh", fallback=CHOICE.get(_key))

_warm_before_mh = _warm


def _warm(state):
    """prepare() ends with _warm(state): the helper-warp module first (a failed build leaves the shapes on their
    fallbacks), then every chosen kernel once."""
    global F12MH_MOD
    if MH_CHOICE and state.get("f12m"):
        try:
            F12MH_MOD = f12mh_build(F12MH_INSTANCES)
        except Exception as exc:
            print(f"f12mh build FAILED, its shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
            F12MH_MOD = None
    _warm_before_mh(state)
'''


def build(out=PROJ / "submission_v13.py", mh=MH_CHOSEN, mq=MQ_CHOSEN):
    src = (PROJ / BASE).read_text()
    assert hashlib.sha256(src.encode()).hexdigest().startswith("5ec03ce1"), "base is not v12"
    assert "__shared__ float part[KSW][RB][MT + 3];" in src and "f12mh" not in src
    mh_insts = tuple(dict.fromkeys((c["NT"], c["RTW"], c["RG"], c["KSW"], K[layer] // 64 // c["KSW"], c["D"], c["F12"])
                                   for (layer, m), c in sorted(mh.items())))
    mq_insts = tuple(dict.fromkeys((c["NT"], c["RTW"], c["RG"], c["KSW"], K[layer] // 64 // c["KSW"], c["D"], c["F12"])
                                   for (layer, m), c in sorted(mq.items())))
    for (layer, m), c in list(mh.items()) + list(mq.items()):
        assert set(c) == {"RTW", "RG", "KSW", "D", "F12", "NT"} and c["F12"] and 1 <= m <= 8 * c["NT"], (layer, m, c)
        assert 16 * c["RG"] * c["RTW"] * c["KSW"] * (8 * c["NT"] + 3) * 4 + 2 * 16 * c["RG"] * c["RTW"] * 16 <= 48 * 1024, c
    assert not set(mh) & set(mq), "a shape can't be on both"
    sec = body(PROJ / "lab" / "f12mh.py")
    section = (sec + f"\n\nF12MH_INSTANCES = {mh_insts!r}\nMQ_INSTANCES = {mq_insts!r}\n"
               + f"MH_CHOICE = {dict(sorted(mh.items()))!r}\nMQ_CHOICE = {dict(sorted(mq.items()))!r}\n" + TAIL)
    out_src = src.rstrip() + "\n" + SEPARATOR + section
    Path(out).write_text(out_src)
    print(f"wrote {out} sha256 {hashlib.sha256(out_src.encode()).hexdigest()[:16]} (helper on {sorted(mh)}, extra F12M on {sorted(mq)})")


if __name__ == "__main__":
    build()

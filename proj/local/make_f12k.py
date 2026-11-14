"""Build a clean scored file with F12K (P43, stream-K F12M) on chosen shapes: a base file byte for byte (default
scored/submission_v11b.py) + the F12K section: proj/f12m/f12k.py (code only, F12K_INSTANCES = the chosen configs' instances),
the chosen configs (each with the base's kernel as fallback), and _warm wrapped so prepare() builds the module and allocates
each chosen config's scratch (FP32 piece slots + zeroed counters) before the base's warm-up, which launches every chosen kernel
once outside any capture. No F12M planes or a failed build leaves the base's kernels on every shape.

Run: .venv/bin/python proj/local/make_f12k.py [--base scored/submission_v11b.py] out.py   (configs: F12K_CHOSEN below, from the
lab's "SK CHOICE" lines; each {"kind": "f12k", "NT": .., "KSW": .., "D": .., "G": ..})
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " F12K (proj/f12m/f12k.py, P43): stream-K F12M for the few-tile shapes\n\n"
N_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192}
K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096}

# Filled from PE42's "SK CHOICE" lines.
F12K_CHOSEN = {}

TAIL = '''

F12K_MOD = None
F12K_STATE = {"bufs": {}}


def f12k_key(c, layer, n, k):
    return (layer, n, k, c["NT"], c["KSW"], c["D"], c["G"])


def f12k_buffers(n, k, c, device):
    n_tiles, chs = n // 16, k // 64
    maxp = max(f12k_pieces(n_tiles, chs, c["G"]))
    return {"slots": torch.zeros(n_tiles * maxp * 8 * c["NT"] * 16, dtype=torch.float32, device=device),
            "cnt": torch.zeros(n_tiles, dtype=torch.int32, device=device), "maxp": maxp}


def f12k_setup(state):
    """The module with the chosen instances, then one scratch set per chosen (layer, m). Without F12M planes or with a
    failed build, nothing is set and every F12K shape runs its fallback."""
    global F12K_MOD
    if not state.get("f12m"):
        return
    try:
        mod = f12k_build(F12K_INSTANCES)
    except Exception as exc:
        print(f"f12k build FAILED, F12K shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        return
    for (layer, m), c in F12K_CHOICE.items():
        if layer in state["f12m"]:
            n = 1024 if layer == "unpooler" and m <= 8 else state[layer].shape[0]
            k = state[layer].shape[1]
            F12K_STATE["bufs"].setdefault(f12k_key(c, layer, n, k), f12k_buffers(n, k, c, state[layer].device))
    F12K_MOD = mod


def run_f12k(c, x, layer, n, state):
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    bufs = F12K_STATE["bufs"].get(f12k_key(c, layer, n, k))
    if F12K_MOD is None or pl is None or bufs is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12K_MOD, f12k_name((c["NT"], c["KSW"], c["D"])))(
        pl["p"], pl["rec"], x.contiguous(), out, bufs["slots"], bufs["cnt"], m, n, c["G"], bufs["maxp"])
    return out


_run_choice_pre_f12k = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12k":
        return run_f12k(c, x, layer, n, state)
    return _run_choice_pre_f12k(c, x, layer, n, state)


for _key, _c in F12K_CHOICE.items():
    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))

_warm_before_f12k = _warm


def _warm(state):
    """prepare() ends with _warm(state): F12K's module and scratch first, then every chosen kernel once."""
    f12k_setup(state)
    _warm_before_f12k(state)
'''


def build(out, base, chosen):
    assert chosen, "F12K_CHOSEN is empty"
    insts = tuple(sorted({(c["NT"], c["KSW"], c["D"]) for c in chosen.values()}))
    for (layer, m), c in chosen.items():
        assert c["kind"] == "f12k" and set(c) == {"kind", "NT", "KSW", "D", "G"}, (layer, m, c)
        assert 1 <= m <= 8 * c["NT"] and c["KSW"] in (4, 8, 16) and c["D"] in (1, 2, 4, 8) and c["NT"] in (1, 2, 4), (layer, m, c)
        n = 1024 if layer == "unpooler" and m <= 8 else N_ROWS[layer]
        assert 1 <= c["G"] <= n // 16 * K[layer] // 64, (layer, m, c)
    src = (PROJ / base).read_text().rstrip() + "\n"
    assert src.count("\ndef prepare(weights") == 1 and "\ndef _warm(state)" in src and "\ndef run_choice(" in src
    assert "\ndef f12m_device_planes(w, natural=False)" in src and "f12k_kernel" not in src, f"{base}: no F12M planes or F12K present"
    f12k_path = PROJ / "f12m" / "pinned" / "f12k_v1.py"   # the exact source PE42 measured (sha256 2a108416...)
    sha = hashlib.sha256(f12k_path.read_bytes()).hexdigest()
    assert sha.startswith("2a10841696bea2a0"), f"pinned f12k_v1.py changed: {sha[:16]}"
    sec = body(f12k_path)
    assert sec.count("F12K_INSTANCES = ()") == 1
    sec = sec.replace("F12K_INSTANCES = ()", f"F12K_INSTANCES = {insts!r}")
    section = (sec + f"\n\n# proj/f12m/pinned/f12k_v1.py sha256 {sha[:16]}\nF12K_CHOICE = " + repr(dict(sorted(chosen.items()))) + "\n" + TAIL)
    Path(out).write_text(src + SEPARATOR + section)
    print(f"wrote {out} ({len(chosen)} shapes on F12K, instances {insts}; f12k.py sha256 {sha[:16]})")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = "scored/submission_v11b.py"
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    assert args, __doc__
    build(args[0], base, F12K_CHOSEN)

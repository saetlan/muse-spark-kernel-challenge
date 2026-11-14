"""Build proj/submission_v1.py: submission.py byte for byte, with PE1's chosen configs as CHOICE
and a prepare() that launches every chosen kernel once (JIT compile and module load happen
before graph capture, never inside it).

Run: .venv/bin/python proj/local/make_v1.py
"""

from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def swap(bm, bn, bk, split, mode):
    return {"kind": "swap", "BM": bm, "BN": bn, "BK": bk, "split": split, "mode": mode, "warps": 4, "stages": 4}


def gemm(bm, bn, bk, split, mode, warps, stages):
    return {"kind": "gemm", "BM": bm, "BN": bn, "BK": bk, "split": split, "mode": mode, "warps": warps,
            "stages": stages}


# PE1 (job eb62bbc5e6d147fa), the lab's "CHOSEN" lines.
CHOICE = {
    ("o", 1): {"kind": "gemv"},
    ("pooler", 1): {"kind": "gemv"},
    ("unpooler", 1): {"kind": "tiny", "max_m": 16},
    ("pooler", 8): swap(16, 64, 256, 1, 0),
    ("router", 8): swap(16, 64, 64, 32, 2),
    ("unpooler", 8): {"kind": "tiny", "max_m": 16},
    ("o", 16): swap(16, 64, 128, 1, 0),
    ("pooler", 16): swap(16, 64, 256, 1, 0),
    ("router", 16): swap(16, 64, 128, 32, 1),
    ("pooler", 32): swap(32, 64, 256, 1, 0),
    ("router", 32): swap(32, 64, 128, 32, 1),
    ("router", 128): gemm(64, 64, 64, 16, 1, 4, 4),
    ("router", 256): gemm(64, 64, 64, 16, 1, 4, 4),
}

WARM = '''

def _warm(state):
    """Run every chosen kernel once on zeros, so compiling happens here and not in a capture."""
    for (layer, m), c in CHOICE.items():
        n = 1024 if layer == "unpooler" and m <= 8 else state[layer].shape[0]
        x = torch.zeros((m, state[layer].shape[1]), dtype=torch.bfloat16, device=state[layer].device)
        if c["kind"] == "fi_mm":  # FlashInfer picks its tactic for this shape now (synthetic input)
            import flashinfer
            with flashinfer.autotune(True):
                run_choice(c, x, layer, n, state)
        run_choice(c, x, layer, n, state)
    torch.cuda.synchronize()
'''


def build(choice, out_name, source_note):
    src = (PROJ / "submission.py").read_text()
    lines = ["CHOICE = {"] + [f"    {k!r}: {v!r}," for k, v in choice.items()] + ["}"]
    old = "# (layer, m) -> kernel config. Empty: every shape runs SGLang's path.\nCHOICE = {}\n"
    assert src.count(old) == 1
    src = src.replace(old, f"# (layer, m) -> kernel config, from {source_note}. Other shapes run\n"
                           "# SGLang's path.\n" + "\n".join(lines) + "\n")
    old_ret = '        p9_setup(state)\n    return state\n'
    assert src.count(old_ret) == 1
    src = src.replace(old_ret, '        p9_setup(state)\n    _warm(state)\n    return state\n')
    src = src.replace("\n\ndef prepare(weights):", WARM + "\n\ndef prepare(weights):", 1)
    (PROJ / out_name).write_text(src)
    print(f"wrote proj/{out_name} ({len(choice)} chosen shapes)")


def main():
    build(CHOICE, "submission_v1.py", "the PE1 lab (job eb62bbc5e6d147fa)")


if __name__ == "__main__":
    main()

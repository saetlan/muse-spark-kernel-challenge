"""Build submission_m2.py: submission.py (v1) unchanged except that its prepare and kernel_fn
become _v1_prepare and _v1_kernel_fn, then the M2 section (moe/m2/m2.py), which defines the new
prepare and kernel_fn and falls back to v1 wherever M2 isn't configured.

Run: .venv/bin/python moe/local/make_m2.py
"""

from pathlib import Path

MOE = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " M2 (moe/m2/m2.py)\n\n"


def build(out_path=MOE / "submission_m2.py"):
    sub = (MOE / "submission.py").read_text()
    for name in ("prepare(weights", "kernel_fn(data, state"):
        if sub.count(f"\ndef {name}") != 1:
            raise SystemExit(f"expected exactly one top-level def {name}...) in submission.py")
        sub = sub.replace(f"\ndef {name}", f"\ndef _v1_{name}", 1)
    m2 = (MOE / "m2" / "m2.py").read_text()
    if m2.startswith("# okc:"):
        raise SystemExit("m2.py must not carry the okc problem line")
    text = sub.rstrip() + "\n" + SEPARATOR + m2.strip() + "\n"
    Path(out_path).write_text(text)
    print(f"wrote {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    build()

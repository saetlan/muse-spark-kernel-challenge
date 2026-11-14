"""Build submission_m13.py: submission.py (v1) with prepare/kernel_fn renamed _v1_*, then the M11
section (moe/m11/m11.py) with its prepare/kernel_fn renamed _m11_*, then the M13 section
(moe/m13/m13.py), which defines the new prepare and kernel_fn (M13 where configured, else M11/v1).

Run: .venv/bin/python moe/local/make_m13.py
"""

from pathlib import Path

MOE = Path(__file__).resolve().parents[1]


def _rename(text, prefix, what):
    for name in ("prepare(weights", "kernel_fn(data, state"):
        if text.count(f"\ndef {name}") != 1:
            raise SystemExit(f"expected exactly one top-level def {name}...) in {what}")
        text = text.replace(f"\ndef {name}", f"\ndef {prefix}{name}", 1)
    return text


def build(out_path=MOE / "submission_m13.py", tune=False):
    sub = _rename((MOE / "submission.py").read_text(), "_v1_", "submission.py")
    m11 = _rename("\n" + (MOE / "m11" / "m11.py").read_text(), "_m11_", "m11.py")
    m13 = (MOE / "m13" / "m13.py").read_text()
    g2p = (MOE / "m13" / "g2p_moe.py").read_text()  # E23: prefill's G2P (snapshot of shared/glue/g2p_moe.py)
    if tune:  # M13's prepare becomes _m13_prepare; the self-tuning section defines prepare
        if m13.count("\ndef prepare(weights") != 1:
            raise SystemExit("expected one def prepare in m13.py")
        m13 = m13.replace("\ndef prepare(weights", "\ndef _m13_prepare(weights", 1)
        m13 += "\n\n# " + "=" * 70 + " self-tuning prepare (moe/m13/tune.py)\n\n" + (MOE / "m13" / "tune.py").read_text()
    for name, text in (("m11.py", m11), ("m13.py", m13)):
        if text.lstrip().startswith("# okc:"):
            raise SystemExit(f"{name} must not carry the okc problem line")
    sep = "\n\n# " + "=" * 70 + " {} (moe/{})\n\n"
    text = (sub.rstrip() + "\n" + sep.format("M11", "m11/m11.py") + m11.strip() + "\n"
            + sep.format("G2P", "m13/g2p_moe.py") + g2p.strip() + "\n"
            + sep.format("M13", "m13/m13.py") + m13.strip() + "\n")
    Path(out_path).write_text(text)
    print(f"wrote {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    import sys
    if "--tune" in sys.argv:
        build(MOE / "submission_m13t.py", tune=True)
    else:
        build()

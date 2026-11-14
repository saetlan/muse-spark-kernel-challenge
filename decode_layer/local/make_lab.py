"""Build a decode_layer check-run lab file: a submission file unchanged, with lab sections
appended. The submission's prepare() is renamed _submission_prepare(); the sections define
prepare(weights) that calls it, runs the lab and returns its state, so the timed kernel_fn is
exactly the submission's.

Run: .venv/bin/python decode_layer/local/make_lab.py submission_d1.py decode_layer/lab/d0.py decode_layer/lab_d0.py
(the base is relative to decode_layer/; lab/common.py is always included first)
"""

import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " lab ({name}): check runs only\n\n"


def build(base, section_paths, out_path):
    sub = (DECODE / base).read_text()
    if not sub.startswith("# okc: problem=muse_decode_layer\n"):
        raise SystemExit(f"{base}: first line must be the okc problem line")
    if sub.count("\ndef prepare(weights") != 1:
        raise SystemExit(f"expected exactly one top-level def prepare(weights...) in {base}")
    sub = sub.replace("\ndef prepare(weights", "\ndef _submission_prepare(weights", 1)
    parts = [sub.rstrip() + "\n"]
    for n, path in enumerate(section_paths):
        text = Path(path).read_text()
        if text.startswith("# okc:"):
            raise SystemExit(f"{path}: sections must not carry the okc problem line")
        if n < len(section_paths) - 1 and "\ndef prepare(weights" in text:  # only the last section's prepare runs
            text = text.replace("\ndef prepare(weights", f"\ndef _{Path(path).stem}_prepare(weights")
        parts.append(SEPARATOR.format(name=Path(path).stem) + text.strip() + "\n")
    Path(out_path).write_text("".join(parts))
    print(f"wrote {out_path} ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    base, *sections, out = sys.argv[1:]
    build(base, [DECODE / "lab" / "common.py", *sections], out)

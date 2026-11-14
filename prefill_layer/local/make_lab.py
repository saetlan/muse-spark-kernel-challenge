"""Build a check-run lab file: submission.py unchanged, with lab sections appended (prefill copy of
moe/local/make_lab.py). The submission's prepare() becomes _submission_prepare(); the sections define
prepare(weights), which calls it, runs the lab and returns its state, so kernel_fn is the submission's.

Run: .venv/bin/python prefill_layer/local/make_lab.py prefill_layer/lab/re1.py prefill_layer/lab_re1.py
"""

import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " lab ({name}): check runs only\n\n"


def build(section_paths, out_path, base="submission.py"):
    sub = (DIR / base).read_text()
    if sub.count("\ndef prepare(weights") != 1:
        raise SystemExit(f"expected exactly one top-level def prepare(weights...) in {base}")
    sub = sub.replace("\ndef prepare(weights", "\ndef _submission_prepare(weights", 1)
    parts = [sub.rstrip() + "\n"]
    for path in section_paths:
        text = Path(path).read_text()
        if text.startswith("# okc:"):
            raise SystemExit(f"{path}: sections must not carry the okc problem line")
        parts.append(SEPARATOR.format(name=Path(path).stem) + text.strip() + "\n")
    Path(out_path).write_text("".join(parts))
    print(f"wrote {out_path} ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    args = sys.argv[1:]
    base = "submission.py"
    if args[:1] == ["--base"]:
        base, args = args[1], args[2:]
    *sections, out = args
    build([DIR / "lab" / "common.py", *sections], out, base)

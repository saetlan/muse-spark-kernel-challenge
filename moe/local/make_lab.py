"""Build a check-run lab file: a submission file unchanged, with lab sections appended.

The submission's prepare() is renamed to _submission_prepare(); the sections must define
prepare(weights) that calls it, runs the lab and returns its state. So the timed kernel_fn is
exactly the submission's.

Run: .venv/bin/python moe/local/make_lab.py [--base submission_m2.py] moe/lab/e2.py moe/lab_e2.py
(the base defaults to submission.py)
"""

import sys
from pathlib import Path

MOE = Path(__file__).resolve().parents[1]
SEPARATOR = "\n\n# " + "=" * 70 + " lab ({name}): check runs only\n\n"


def build(section_paths, out_path, base="submission.py"):
    sub = (MOE / base).read_text()
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
    build([MOE / "lab" / "common.py", *sections], out, base)

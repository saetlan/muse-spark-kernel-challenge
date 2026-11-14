"""Builds a check-run lab file: the base submission byte for byte, then the named blocks appended.
Usage: .venv/bin/python attention/local/make_lab.py [--base <file in attention/>] <name> [block ...]
  base defaults to submission.py; block = a file in attention/local/ (lab_<x>_block.py by short
  name <x>, or a .py file name); with no blocks, lab_<name>_block.py. Writes attention/lab_<name>.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
args = sys.argv[1:]
base = "submission.py"
if args[:1] == ["--base"]:
    base, args = args[1], args[2:]
name, blocks = args[0], args[1:] or [args[0]]
sub = open(os.path.join(PROB, base)).read()
parts = []
for blk in blocks:
    path = os.path.join(HERE, blk if blk.endswith(".py") else f"lab_{blk}_block.py")
    parts.append(f"\n\n# ===== appended: attention/local/{os.path.basename(path)} =====\n" + open(path).read())
out = os.path.join(PROB, f"lab_{name}.py")
open(out, "w").write(sub + "".join(parts))
print(f"wrote {out}: {base} ({len(sub)} bytes) + " + ", ".join(blocks))

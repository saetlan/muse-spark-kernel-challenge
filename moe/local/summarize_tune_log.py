"""Summarize a self-tuning run's prepare log (profiles/<file>.<job>.log): recon lines, the best stage per kernel
family and batch size, the M13 prefetch grid, and each class's decision.

Run: .venv/bin/python moe/local/summarize_tune_log.py moe/profiles/<file>.<job>.log
"""

import re
import sys
from collections import defaultdict


def main(path):
    lines = [l.rstrip("\n") for l in open(path) if l.startswith("[tune]") or l.startswith("[m13]")]
    best = defaultdict(dict)  # (stage, kind) -> {bs: (t, name)}
    for l in lines:
        if l.startswith("[m13]") or "base prepare" in l or "bw TB/s" in l or "recon bs" in l or "attrs:" in l \
                or "out of budget" in l or "stage search:" in l or "FAILED" in l or "skeleton" in l:
            print(l[:400])
        m = re.match(r"\[tune\] s([12]) (\S+) \| (.*)$", l)
        if m:
            stage, name, rest = int(m.group(1)), m.group(2), m.group(3)
            kind = name.split(":")[0]
            for bs, t in re.findall(r"(\d+):(\d+)", rest):
                bs, t = int(bs), float(t)
                cur = best[(stage, kind)].get(bs)
                if cur is None or t < cur[0]:
                    best[(stage, kind)][bs] = (t, name)
    if best:
        bss = sorted({bs for v in best.values() for bs in v})
        print("\nbest stage per family (us):  " + "  ".join(f"bs{b:>4}" for b in bss))
        for (stage, kind) in sorted(best):
            row = best[(stage, kind)]
            print(f"  s{stage} {kind:4}                  " + "  ".join(f"{row[b][0]:6.0f}" if b in row else "     -" for b in bss))
        print("\nbest names:")
        for (stage, kind) in sorted(best):
            for b in bss:
                if b in best[(stage, kind)]:
                    print(f"  s{stage} {kind} bs{b}: {best[(stage, kind)][b][1]} {best[(stage, kind)][b][0]:.0f}")
    print()
    for l in lines:
        if "m13 s1(npre,nxq)" in l or re.search(r"class \[\d+,\d+\) bs\d+:", l) or "rejected" in l or "done in" in l:
            print(l[:700])


if __name__ == "__main__":
    main(sys.argv[1])

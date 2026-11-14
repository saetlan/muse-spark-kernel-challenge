"""Summarize an okc check result for the F12 labs: score, per-shape judge times, and the lab lines that
matter (CHOSEN / keep, per-candidate F12V/F12M timings, V5/V6 CHOICE configs, failures), from the
result JSON that `okc check --json` printed (its log tail) or a saved log file.

Run: .venv/bin/python proj/local/parse_lab.py <okc json or log file>
"""

import ast
import json
import re
import sys
from pathlib import Path


def load(path):
    text = Path(path).read_text()
    try:
        start = text.index("{")
        d = json.loads(text[start:])
    except (ValueError, json.JSONDecodeError):
        return None, text
    res = d.get("result", d)
    log = res.get("log_tail", "") or ""
    out = d.get("output", {}).get("log")
    if out and Path(out).exists():
        log = Path(out).read_text()
    return d, log


def main():
    d, log = load(sys.argv[1])
    if d is not None:
        res = d.get("result", d)
        print(f"job {d.get('id')} status {res.get('status')} passed {res.get('passed')} speedup {res.get('speedup')} "
              f"CI {res.get('speedup_ci')} score_us {res.get('score_us')}")
        for r in res.get("records", []):
            if r.get("event") == "setup":
                print(f"prepare {r.get('prepare_s')} s, status {r.get('status')}")
            elif r.get("status") != "pass" and r.get("event") == "test":
                print("TEST", r.get("status"), r.get("case"), r.get("message"))
    keys = ("CHOSEN", "keep v1", "FAILED", "WRONG", "CHOICE", "R/RG/KSW/D", "RTW/RG/KSW/D", "KS/ST", "regs", "escapes",
            "took", "budget", "skipped", "build FAILED")
    for line in log.splitlines():
        if any(k in line for k in keys):
            print(line[:400])
    chosen = {}
    for m in re.finditer(r"V[0-9]+ CHOICE (\w+) m(\d+): (\{.*\})", log):
        chosen[(m.group(1), int(m.group(2)))] = ast.literal_eval(m.group(3))
    if chosen:
        print("\nCHOSEN =", repr(chosen))


if __name__ == "__main__":
    main()

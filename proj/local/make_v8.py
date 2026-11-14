"""Build proj/submission_v8.py = v7a (v6 + PE10's router configs) + the WG/WGF winners of a lab result
(its "V8"/"V9" CHOICE lines), i.e. exactly the set that lab's timed kernel_fn ran.

Run: .venv/bin/python proj/local/make_v8.py <okc result json of PE12 or PE13>
"""

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chosen  # noqa: E402
import make_v7  # noqa: E402


def lab_winners(path):
    text = Path(path).read_text()
    d = json.loads(text[text.index("{"):])
    log = Path(d["output"]["log"]).read_text() if d.get("output", {}).get("log") else d["result"]["log_tail"]
    out = {}
    for mo in re.finditer(r"V[89] CHOICE (\w+) m(\d+): (\{.*\})", log):
        c = ast.literal_eval(mo.group(3))
        if c["kind"] in ("wg", "wgf"):
            out[(mo.group(1), int(mo.group(2)))] = c
    return out


if __name__ == "__main__":
    win = lab_winners(sys.argv[1])
    print("PE12 winners:", win)
    assert win, "no WG/WGF winners in that result"
    make_v7.build({**chosen.V6, **chosen.PE10, **win}, "submission_v8.py")

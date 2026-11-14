"""Build proj/lab_e43.py (PE43, P43 v2: F12K HL on the router): scored/submission_v11b.py byte for byte + lab/common.py +
proj/f12m/f12k.py v2 (code only) + lab/sweep.py + lab/f12k_lab.py + lab/e43.py with PE42_WINNERS filled in.

Run: .venv/bin/python proj/local/make_e43.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lab  # noqa: E402
from make_p9 import body  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
BASE = "scored/submission_v11b.py"
# PE42's "SK CHOICE" lines (F12 path, v1 measured; re-measured here on v2's build).
PE42_WINNERS = {}


def build(winners=PE42_WINNERS, out=PROJ / "lab_e43.py"):
    base_sha = hashlib.sha256((PROJ / BASE).read_bytes()).hexdigest()
    assert base_sha.startswith("cfe20913"), f"{BASE} is not v11b: {base_sha[:16]}"
    for (layer, m), c in winners.items():
        assert c["kind"] == "f12k" and not c.get("HL") and set(c) == {"kind", "NT", "KSW", "D", "G"}, (layer, m, c)
    sec = body(PROJ / "f12m" / "f12k.py")
    assert "bool HL" in sec and "F12K_INSTANCES = ()" in sec, "f12k.py is not v2"
    sec_path = PROJ / "lab" / "_f12k2_section.py"
    sec_path.write_text(sec)
    e43 = (PROJ / "lab" / "e43.py").read_text()
    assert e43.count("PE42_WINNERS = {}") == 1
    e43_path = PROJ / "lab" / "_e43_filled.py"
    e43_path.write_text(e43.replace("PE42_WINNERS = {}", f"PE42_WINNERS = {dict(sorted(winners.items()))!r}"))
    make_lab.build([PROJ / "lab" / "common.py", sec_path, PROJ / "lab" / "sweep.py", PROJ / "lab" / "f12k_lab.py", e43_path],
                   out, base=BASE)
    print("base", BASE, "sha256", base_sha[:16], "| f12k.py (v2) sha256",
          hashlib.sha256((PROJ / "f12m" / "f12k.py").read_bytes()).hexdigest()[:16], "| PE42 winners", len(winners))


if __name__ == "__main__":
    build()

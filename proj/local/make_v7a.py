"""Build proj/submission_v7a.py: v6 (chosen.V6) + PE10's router configs (chosen.PE10); the base of PE12.

Run: .venv/bin/python proj/local/make_v7a.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chosen  # noqa: E402
import make_v7  # noqa: E402

if __name__ == "__main__":
    make_v7.build({**chosen.V6, **chosen.PE10}, "submission_v7a.py")

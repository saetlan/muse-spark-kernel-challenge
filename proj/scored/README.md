# Scored files (muse_bf16_proj)

Exact copies of what went to `okc submit`, snapshotted right before each submit from now on.

| File | Job | Speedup | Note |
|---|---|---|---|
| `submission_v1.py` | `4df483b4c4254517` (Nov 3) | 1.117× (CI 1.100–1.134) | Reconstructed after the fact: today's `submission_v1.py` minus the three `_fma_kernel` insertions made after the submit (the only edits since). okc's `content_hash` is not a plain SHA-256 of the file, so the bytes could not be confirmed against it. |
| `submission_v2.py` | `aae716920c1d403f` (Nov 3) | 1.192× (CI 1.174–1.210) | Snapshotted before submit and submitted from this path (profiles in `scored/profiles/`). SHA-256 da1b76b358072dee… |
| `submission_v4.py` | `0d1dc429eebf455f` (Nov 4) | 1.247× (CI 1.228–1.265) | Snapshotted before submit and submitted from this path. SHA-256 4048d4cec26d1688… |

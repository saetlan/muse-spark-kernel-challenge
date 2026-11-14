"""CPU dry run of lab L3's host side: the FP32 reference (_lab_reference) must agree with the
judge's reference.py on CPU cases (full and sliding, decode and verify, after our emulated
prologue wrote the new rows), and every tune in the lab must plan cleanly."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from emulate import load_submission, prologue  # noqa: E402

import reference as ref  # noqa: E402

lab = load_submission(os.path.join(os.path.dirname(HERE), "lab_l3.py"))
worst = 0.0
for i, case in enumerate([dict(batch=2, ctx=3000, layer="full"), dict(batch=2, ctx=3000, layer="full", q_len=8),
                          dict(batch=2, ctx=5000, layer="sliding"), dict(batch=2, ctx=5000, layer="sliding", q_len=8)]):
    data = ref.generate_input(heads="tp8", page=1, dist="soft", seed=90 + i, device="cpu", **case)
    want = ref.ref_kernel(data)[0].float()          # before our prologue touches the cache
    prologue(data, lab)                             # writes the block's rows, as run() does
    got = lab._lab_reference(data)
    err = float((got - want).norm() / want.norm())
    worst = max(worst, err)
    print(f"{case}: lab reference vs reference.py {err:.1e}")
assert worst < 2e-2, worst      # the cache holds FP8 new keys either way; queries BF16 in both
for tune in [None, (16, 256, 64, 4, 3), (16, 256, 64, 2, 5), (16, 256, 64, 4, 6), (16, 256, 32, 4, 8),
             (16, 256, 128, 4, 3), (16, 256, 64, 8, 4), (4, 256, 64, 8, 2)]:
    for b, l, w in ((16, 1, 0), (32, 1, 0), (1, 1, 0), (16, 8, 0), (16, 1, 2048)):
        cfg = dict(batch=b, q_len=l, num_q_heads=16, num_kv_heads=2, page_size=1, window=w)
        lab.plan(cfg, lab.scored_max_pages(b, l), tune)
print("dryrun_lab_l3: ok")

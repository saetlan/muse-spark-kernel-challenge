"""CPU dry run of lab L2: imports the lab file (stub Triton), exercises its host-side helpers on
CPU, and runs every V-section launch setting through the emulator and the judge's checker."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from emulate import load_submission, run  # noqa: E402

import reference as ref  # noqa: E402

lab = load_submission(os.path.join(os.path.dirname(HERE), "lab_l2.py"))
gen = torch.Generator().manual_seed(0)
orders = lab._orders(1000, gen, "cpu")
assert sorted(orders) == ["rand", "rand50", "seq", "sorted50"]
assert torch.equal(torch.sort(orders["rand"]).values, orders["seq"])
assert bool((orders["sorted50"][1:] >= orders["sorted50"][:-1]).all())
assert "void gsum(" in lab._CUDA_GATHER_DECL and "gsum_kernel<16>" in lab._CUDA_GATHER
tunes = [(32, 256, 64, 4, 4), (32, 256, 128, 4, 3), (32, 128, 64, 4, 4), (16, 256, 64, 4, 5),
         (16, 256, 128, 4, 4), (8, 256, 128, 8, 3), (16, 256, 32, 4, 6), (4, 256, 64, 4, 3),
         (8, 256, 64, 4, 3), (4, 256, 32, 4, 4), (8, 256, 32, 4, 4)]
bad = 0
for i, tune in enumerate(tunes):
    for q_len in (1, 8):
        data = ref.generate_input(batch=2, ctx=6000, heads="tp8", layer="full", q_len=q_len, page=1,
                                  dist="coherent", seed=50 + i, device="cpu")
        ok, msg = ref.check_implementation(data, ref.collect(data, run(data, lab, tune)))
        bad += not ok
        if not ok:
            print("FAIL", tune, q_len, msg)
print(f"dryrun_lab_l2: {2 * len(tunes)} runs, {bad} failed")
sys.exit(1 if bad else 0)

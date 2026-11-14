"""CPU dry run of lab L5's host side: imports lab_l5.py with stub Triton, checks the appended CUDA
sources equal cuda_dec2.py / cuda_gather.py, every A5b variant's slice count is sane at the scored
shapes, and the gather sections' shared-memory sizes fit (227 KB per block)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emulate import load_submission  # noqa: E402
import cuda_dec2  # noqa: E402
import cuda_gather  # noqa: E402

lab = load_submission(os.path.join(os.path.dirname(HERE), "lab_l5.py"))
assert lab.CUDA_DEC2_SRC == cuda_dec2.CUDA_DEC2_SRC and lab.CUDA_GATHER_SRC == cuda_gather.CUDA_GATHER_SRC
assert lab.CUDA_BULK_SRC == cuda_gather.CUDA_BULK_SRC and "gbulk_kernel" not in lab.CUDA_GATHER_SRC
for b, w in ((16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)):
    cfg = dict(batch=b, q_len=1, num_q_heads=16, num_kv_heads=2, page_size=1, window=w)
    mp = lab.scored_max_pages(b, 1)
    row = []
    for cps, warps in ((2, 4), (4, 4), (8, 4), (2, 8), (4, 8), (8, 8)):
        ns = lab._dec2_nsplit(cfg, mp, cps, warps)
        keys = (min(mp, w + 1) if w else mp) / ns
        assert 1 <= ns <= lab.MAX_SPLITS and keys >= 16 * warps, (b, w, cps, warps, ns, keys)
        row.append(f"c{cps}w{warps}:n{ns}")
    print(f"{'S' if w else 'F'}d{b}: " + " ".join(row))
for stages in (2, 4, 6):
    for warps in (4, 8):
        if warps * stages * 4096 <= 227 * 1024:
            print(f"cp s{stages} w{warps}: smem {warps * stages * 4096 // 1024} KB")
for stages in (2, 4):
    for warps in (4, 8):
        sm = warps * stages * (4096 + 8)
        assert sm <= 227 * 1024, (stages, warps, sm)
        print(f"bulk s{stages} w{warps}: smem {sm / 1024:.1f} KB")
print("dryrun_lab_l5: ok")

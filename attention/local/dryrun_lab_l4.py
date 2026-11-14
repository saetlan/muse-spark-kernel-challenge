"""CPU dry run of lab L4's host side: imports lab_l4.py with stub Triton, checks the embedded CUDA
source is cuda_dec.py's, and that every variant's slice count is sane at the scored shapes (each
CTA's 4/8 warps get at least one 16-token chunk; the emulator covers these slice counts' math)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emulate import load_submission  # noqa: E402
import cuda_dec  # noqa: E402

lab = load_submission(os.path.join(os.path.dirname(HERE), "lab_l4.py"))
assert lab.CUDA_DEC_SRC == cuda_dec.CUDA_DEC_SRC and lab.CUDA_DEC_DECL == cuda_dec.CUDA_DEC_DECL
for b, w in ((16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)):
    cfg = dict(batch=b, q_len=1, num_q_heads=16, num_kv_heads=2, page_size=1, window=w)
    mp = lab.scored_max_pages(b, 1)
    row = []
    for cps, warps in ((2, 4), (4, 4), (8, 4), (16, 4), (2, 8), (4, 8), (8, 8)):
        ns = lab._dec_nsplit(cfg, mp, cps, warps)
        keys = (min(mp, w + 1) if w else mp) / ns
        assert 1 <= ns <= lab.MAX_SPLITS and keys >= 16 * warps, (b, w, cps, warps, ns, keys)
        row.append(f"c{cps}w{warps}:n{ns}({keys:.0f}k)")
    print(f"{'S' if w else 'F'}d{b}: " + " ".join(row))
print("dryrun_lab_l4: ok")

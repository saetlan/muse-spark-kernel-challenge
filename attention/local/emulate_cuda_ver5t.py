"""CPU emulation of attention/local/cuda_ver5t.py (V5t: V5 with V re-laid once per stage by the producer):
V5's emulator (emulate_cuda_ver5.py, PTX fragment formulas, judge's checker) with the V fragments taken
through the re-laid tile by the kernel's two index formulas: the producer lane L writes V5's fragment
words of chunk c, head hv as uint4 k at vt[((c*2 + hv)*4 + k)*32 + L] (k 0/1: vb0 words 0-3 / 4-7,
k 2/3: vb1); the consumer lane L reads uint4 k of block (c*2 + h) at the same formula. The rest of the
kernel is V5's text (make_ver5t.py), checked by V5's emulator and sim_ver5t_pipe.py (barrier protocol).
MUT plants layout bugs (mutation_test_cuda_ver5t.py).
Run: .venv/bin/python attention/local/emulate_cuda_ver5t.py [--quick]"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_ver5 as e5  # noqa: E402

MUT = set()


def vt_frags(sv, c, h):
    vt = torch.full((16 * 32, 16), 0x7F, dtype=torch.uint8)          # uint4 slots, NaN bytes if never written
    for cc in range(2):                                             # producer: every (chunk, head) block
        for hv in range(2):
            w0, w1 = e5.v_frags(sv, cc, hv)
            words = w0 + w1                                         # 16 x [32, 4]
            for k in range(4):
                blk = (cc * 2 + hv) * 4 + k
                vt[blk * 32: blk * 32 + 32] = torch.cat(words[4 * k: 4 * k + 4], 1)
    hh = (1 - h) if "head_swap" in MUT else h
    out = []
    for k in range(4):                                              # consumer: block (c*2 + h), uint4 k, lane
        blk = ((hh * 2 + c) if "blk_order" in MUT else (c * 2 + hh)) * 4 + (k ^ 1 if "k_order" in MUT else k)
        u = vt[blk * 32: blk * 32 + 32]
        if "lane_shift" in MUT:
            u = torch.roll(u, 1, 0)
        out += [u[:, 4 * i: 4 * i + 4] for i in range(4)]
    return out[:8], out[8:]


def run_cases(quick):
    e5.VFRAG = vt_frags
    try:
        return e5.run_cases(quick)
    finally:
        e5.VFRAG = None


if __name__ == "__main__":
    bad = run_cases("--quick" in sys.argv)
    print("emulate_cuda_ver5t:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

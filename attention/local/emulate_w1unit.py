"""CPU emulation of cuda_w1unit.py's index code under the layouts W1 assumes (emulate_cuda_w1.wgmma: CUTLASS
ALayout_64x32 / CLayout_64xN, K-major canonical B through the descriptor): for every compiled variant,
builds the A registers, the B shared-memory bytes and the descriptor exactly as the kernel does, runs
the emulated wgmma and scatters D as the kernel does. Variants that keep W1's reading (SWAP 0, AMODE 0,
any SBO / CPMODE) must reproduce A . B^T exactly (CHAIN 1: K = 64 chained; CHAIN 2: D0 + A . B^T;
CHAIN 3: e4m3(A . B^T * pscale) . V through W1's P pack and Vt key order, to e4m3 rounding); SWAP 1 and
AMODE 1 must not (they are the alternative hypotheses the GPU run tells apart). Run: .venv/bin/python attention/local/emulate_w1unit.py"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_w1 as ew  # noqa: E402
from emulate_cuda_dec import G_OF, T_OF, e4m3  # noqa: E402

ns = {}
exec(open(os.path.join(HERE, "cuda_w1unit.py")).read(), ns)
SRC = ns["CUDA_W1U_SRC"]
variants = [tuple(int(x) for x in v.split(",")) for v in re.findall(r"X\(([^)]*)\)", re.search(r"#define OKC_W1U_VARIANTS\(X\) ((?:.*\\\n)*.*)", SRC).group(1))]
gen = torch.Generator().manual_seed(5)
from emulate_cuda_ver4 import e4m3x4  # noqa: E402

PSCALE = 1.0 / 16


def dec(x):
    return e4m3(x.reshape(-1, 1)).reshape(x.shape)


def rnd(kd):
    return (torch.randn(64, kd, generator=gen) * 2).to(torch.float8_e4m3fn).view(torch.uint8)


def scatter(acc):
    Dm = torch.full((64, 64), float("nan"))
    for w in range(4):
        for i in range(32):
            j, half, c = i >> 2, (i >> 1) & 1, i & 1
            Dm[16 * w + G_OF + 8 * half, 8 * j + 2 * T_OF + c] = acc[w, :, i]
    return Dm


def gather(Dm):
    acc = torch.zeros(4, 32, 32)
    for w in range(4):
        for i in range(32):
            j, half, c = i >> 2, (i >> 1) & 1, i & 1
            acc[w, :, i] = Dm[16 * w + G_OF + 8 * half, 8 * j + 2 * T_OF + c]
    return acc


bad = 0
for sbo, swap, cpmode, amode, chain in variants:
    lbo, kd = 128, (64 if chain else 32)
    A, B, V = rnd(kd), rnd(kd), rnd(64)
    D0 = torch.randn(64, 64, generator=gen)
    S = dec(A) @ dec(B).T
    if chain == 3:
        ref = dec((S * PSCALE).to(torch.float8_e4m3fn).view(torch.uint8)) @ dec(V)
    else:
        ref = S + (D0 if chain == 2 else 0)
    vt_off = 8 * sbo
    smem = torch.full((8 * sbo + (8 * 528 if chain == 3 else 4096 if chain == 4 else 0),), 0x7F, dtype=torch.uint8)
    for tid in range(128):
        n, ch = tid >> 1, tid & 1
        for c in ((ch,) if chain == 0 else (2 * ch, 2 * ch + 1)):
            d = (n & 7) * 16 + (n >> 3) * sbo + c * lbo
            smem[d:d + 16] = B[n, 16 * c:16 * c + 16]
    if chain == 3:
        for e in range(64 * 64):
            dn, kap = e >> 6, e & 63
            p, kp = kap >> 4, kap & 15
            key = 16 * p + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1)
            smem[vt_off + (dn & 7) * 16 + (dn >> 3) * 528 + p * lbo + kp] = V[key, dn]
    if chain == 4:   # A into a K-major tile at vt_off by the kernel's loop (W2's tile_off)
        for e in range(64 * 16):
            m_, k4 = e >> 4, 4 * (e & 15)
            a_ = vt_off + ew.tile_off(m_, k4)
            smem[a_:a_ + 4] = A[m_, k4:k4 + 4]
    regs = torch.zeros(2, 4, 32, 4, 4, dtype=torch.uint8)
    for s_ in range(2 if chain else 1):
        for w in range(4):
            for r in range(4):
                m = 16 * w + G_OF + 8 * ((r & 1) if amode == 0 else (r >> 1))
                kb = 32 * s_ + 4 * T_OF + 16 * ((r >> 1) if amode == 0 else (r & 1))
                regs[s_, w, :, r] = A[m[:, None], kb[:, None] + torch.arange(4)]
    try:
        acc = gather(D0) if chain == 2 else torch.zeros(4, 32, 32)
        if chain == 0:
            desc = ew.gdesc(0, sbo, lbo) if swap else ew.gdesc(0, lbo, sbo)
            acc = ew.wgmma(acc, regs[0], desc, smem, 0)
        elif chain == 4:
            acc = ew.wgmma_ss(acc, ew.gdesc(vt_off, 128, 512), ew.gdesc(0, lbo, sbo), smem, 0)
            acc = ew.wgmma_ss(acc, ew.gdesc(vt_off + 256, 128, 512), ew.gdesc(2 * lbo, lbo, sbo), smem, 1)
        else:
            acc = ew.wgmma(acc, regs[0], ew.gdesc(0, lbo, sbo), smem, 1 if chain == 2 else 0)
            acc = ew.wgmma(acc, regs[1], ew.gdesc(2 * lbo, lbo, sbo), smem, 1)
        if chain == 3:
            pa = torch.zeros(2, 4, 32, 4, 4, dtype=torch.uint8)
            for c in range(2):
                P = lambda i: (acc[:, :, 16 * c + i] * PSCALE).reshape(-1)  # noqa: E731
                rg = [e4m3x4(P(0), P(1), P(4), P(5)), e4m3x4(P(2), P(3), P(6), P(7)),
                      e4m3x4(P(8), P(9), P(12), P(13)), e4m3x4(P(10), P(11), P(14), P(15))]
                pa[c] = torch.stack(rg, 1).view(4, 32, 4, 4)
            o = ew.wgmma(torch.zeros(4, 32, 32), pa[0], ew.gdesc(vt_off, lbo, 528), smem, 0)
            acc = ew.wgmma(o, pa[1], ew.gdesc(vt_off + 2 * lbo, lbo, 528), smem, 1)
        err = float((scatter(acc) - ref).norm() / ref.norm())
    except AssertionError:
        err = float("inf")
    want_ok = swap == 0 and amode == 0
    ok = (err < (2e-3 if chain == 3 else 1e-6)) == want_ok
    print(("ok   " if ok else "FAIL ") + f"SBO {sbo} SWAP {swap} CPMODE {cpmode} AMODE {amode} CHAIN {chain}: rel err {err:.2e} "
          f"({'must match' if want_ok else 'alternative hypothesis, must differ'})")
    bad += not ok
print("emulate_w1unit:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

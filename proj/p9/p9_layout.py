"""P9 weight layout: lossless repack of W [N, K] (BF16) into mma.m16n8k16 A-fragment order, with K
permuted inside each 64-wide chunk (k_perm) so B fragments come from two 16-byte loads of x.

packed[rt][ks][lane] is one uint4 = exactly one lane's A fragment for the 16x16 tile at rows
16*rt.., k 16*ks..: lane = 4*g + t, register j (0..3) holds W[16*rt + g + 8*(j & 1)][16*ks + 2*t + 8*(j >> 1) + p]
for p = 0 (low half) and 1 (high half), per PTX ISA 9.4 sec. 9.7.16.5.8 (Figure 82). A warp reading
consecutive ks of one rt reads consecutive 512-byte blocks.

The router (FP32) is split as SGLang does (hi = w.to(bf16), lo = (w - hi).to(bf16)) and packed per
k16 step as [hi fragment, lo fragment]: packed_hilo[rt][ks][2][lane].

These functions are used by prepare() (on the GPU) and by the CPU checks; they only reshape and
permute, so they are exact.
"""

import torch


def k_perm(k):
    """Column order of the packed weights: inside every 64-wide K chunk c, packed column 16 j + col
    (MMA step j = 0..3, A column col = 0..15) holds W column 64 c + 16 t + 4 j + 2 h + p, where
    col = 8 h + 2 t + p. Then lane (g, t)'s B values for the chunk's 4 MMAs are x[64 c + 16 t ..
    64 c + 16 t + 16): two 16-byte loads, b0 of step j = 32-bit word 2 j, b1 = word 2 j + 1."""
    idx = []
    for c in range(k // 64):
        for j in range(4):
            for col in range(16):
                h, t, p = col // 8, (col % 8) // 2, col % 2
                idx.append(64 * c + 16 * t + 4 * j + 2 * h + p)
    return torch.tensor(idx, dtype=torch.long)


def p9_pack(w):
    """W [N, K] bf16 -> [N/16, K/16, 32, 8] bf16 in A-fragment order (one lane's uint4 = 8 bf16),
    columns permuted by k_perm inside each 64-wide chunk."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0, (w.shape, w.dtype)
    w = w[:, k_perm(k).to(w.device)]
    # W[16 rt + 8 h + g][16 ks + 8 c + 2 t + p] -> [rt, ks, g, t, c, h, p]; register j = h + 2 c.
    v = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)          # rt, h, g, ks, c, t, p
    v = v.permute(0, 3, 2, 5, 4, 1, 6)                      # rt, ks, g, t, c, h, p
    return v.reshape(n // 16, k // 16, 32, 8).contiguous()


def p9_split_router(w):
    """FP32 router -> (hi, lo) bf16, exactly SGLang's MuseSpark1xRouter._build_split2."""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return hi, lo


def p9_pack_hilo(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] bf16: per k16 step the hi fragment, then the lo one."""
    hi, lo = p9_split_router(w)
    return torch.stack((p9_pack(hi), p9_pack(lo)), dim=2).contiguous()


def a_tile_from_fragments(frag):
    """Inverse view used by the checks: [32, 8] bf16 (one 16x16 tile, all lanes) -> the 16x16 A tile,
    read with the PTX formulas (row = g + 8 (j & 1), col = 2 t + 8 (j >> 1) + p)."""
    a = torch.empty(16, 16, dtype=frag.dtype)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for j in range(4):
            for p in range(2):
                a[g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + p] = frag[lane, 2 * j + p]
    return a

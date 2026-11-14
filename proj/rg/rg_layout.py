"""RG: the FP32 router as SGLang's BF16 hi + lo pair (hi = w.to(bf16), lo = (w - hi).to(bf16); allowed at any
batch size, PF22), packed in mma.sync m16n8k16 A-fragment order with natural k (B comes from shared memory
through ldmatrix, so no k permutation): per 16-row tile and k16 step, lane (g, t) holds the hi fragment then
the lo fragment, each one uint4 of 8 BF16: register j = (W[16 tile + g + 8 (j & 1)][16 step + 2 t + 8 (j >> 1)
+ p], p = 0, 1).
"""

import torch


def rg_frag(v):
    """BF16 [N, K] -> [N/16, K/16, 32, 8] in A-fragment order, natural k."""
    n, k = v.shape
    assert n % 16 == 0 and k % 16 == 0
    return v.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 32, 8)


def rg_pack(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] BF16 (hi fragment, then lo), contiguous."""
    assert w.dtype == torch.float32
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return torch.stack((rg_frag(hi), rg_frag(lo)), dim=2).contiguous()

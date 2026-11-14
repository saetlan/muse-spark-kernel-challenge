"""SX weight layouts (lossless repacks done in prepare()):
  F12 = True:  the F12M 12-bit planes in natural k order (f12m_layout.f12m_device_planes(w, natural=True),
               from the F12M section): [tile16][K/64][3][32][4] int32 + per-row records (base, escapes).
  F12 = False: BF16 A fragments in natural k order: [tile16][K/16][32][8] (rg_frag's order).
"""

import torch


def sx_bf16_frag(w):
    """BF16 W [N, K] -> [N/16, K/16, 32, 8]: lane (g, t) register j = W[16 tile + g + 8 (j & 1)][16 step + 2 t + 8 (j >> 1) + p]."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 16 == 0
    return w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 32, 8).contiguous()


def sx_flags(rec, n, k):
    """Escape flags from F12 records (int32 [N, 8]: base | count << 8, then k | bits << 16): uint8
    [N/16 * K/64], 1 where the 16-row tile has an escape in that 64-wide chunk (the kernel scans the
    records only there)."""
    r = rec.to(torch.int64).view(n, 8) & 0xFFFFFFFF
    count = (r[:, 0] >> 8) & 0xFF
    chs = k // 64
    flags = torch.zeros(n // 16 * chs, dtype=torch.uint8, device=rec.device)
    rows = torch.arange(n, device=rec.device)
    for j in range(7):
        sel = count > j
        kk = r[sel, 1 + j] & 0xFFFF
        flags[(rows[sel] // 16) * chs + kk // 64] = 1
    return flags

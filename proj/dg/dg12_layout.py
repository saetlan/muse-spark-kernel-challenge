"""DG12 weight format (P40 step 2): the F12 12-bit lossless code (per-row 16-exponent window [base, base + 15];
per weight a nibble (sign, code >> 1) and a byte (code & 1, 7 mantissa bits), decoded in 7 ops per 8 weights into BF16
scaled by 2^(112 - base)), laid out per row and per 8 consecutive k so that one decode gives one 16-byte unit of a
K-major SW128 shared-memory tile (8 BF16 in k order, element e = k0 + e):

  Q [N, K/8] int32     the nibble word of unit u (f12v_words element e = weight k = 8u + e)
  L [N, K/8, 2] int32  its two byte words (l0: k 8u..8u+3, l1: k 8u+4..8u+7)
  scale [N] float32    2^(base - 112), applied to output column n in the epilogue
  escapes              weights outside the window (zeros included) decode as the placeholder 2^-15 (scaled domain);
                       per n-block of BN rows a list of (n_local, k, w_true scaled by 2^(112 - base)) corrects them in the
                       epilogue with two FMAs (acc -= x_k 2^-15; acc += x_k w_true_scaled), as SX did

Row bytes: Q K/2 + L K = 1.5 K. The kernel reads chunk-major copies qc [K/64][N][8] and lc [K/64][N][8][2] (PF42): a
stage's BN rows are one contiguous block, TMA'd as 128-byte rows (boxes [128 B, BN/4] and [128 B, BN/2]).
"""

import torch

from f12v_layout import as_int32, f12v_fields, f12v_words


def dg12_pack(w, bn_list=(32, 48, 64, 80, 96, 112, 128, 256)):
    """BF16 W [N, K] (K % 64 == 0) -> dict(q int32 [N, K/8], l int32 [N, K/8, 2], scale f32 [N],
    esc: {BN: (offsets int32 [N/BN + 1], entries int32 [E, 4] = (n_local, k, f32 bits of w_true scaled, 0))}),
    or None when an escape's scaled value is not an exact normal FP32 (never for real weights)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % 64 == 0, (w.shape, w.dtype)
    s, c, m, esc, bits, base = f12v_fields(w)
    shape = (n, k // 8, 8)
    q, l0, l1 = f12v_words(s.view(shape), c.view(shape), m.view(shape))     # [n, k/8] int64
    rows, cols = esc.nonzero(as_tuple=True)
    true = bits[rows, cols]
    e_true = (true >> 7) & 0xFF
    scaled_e = e_true - base[rows] + 112
    zero = (true & 0x7FFF) == 0
    if bool(((~zero) & ((e_true == 0) | (scaled_e < 1) | (scaled_e > 254))).any()):
        return None
    val = torch.where(zero, true & 0x8000, ((true >> 15) << 15) | (scaled_e.clamp(1, 254) << 7) | (true & 0x7F))
    f32 = (val << 16) & 0xFFFFFFFF                                          # BF16 bits -> FP32 bits
    q32 = as_int32(q).contiguous()
    l32 = as_int32(torch.stack((l0, l1), dim=-1)).contiguous()
    out = {"q": q32, "l": l32,
           # the kernel's planes, chunk-major (PF42): [K/64][N][8 units] (32 B per row) and [K/64][N][8][2] (64 B per row),
           # so a stage's BN rows are one contiguous block
           "qc": q32.view(n, k // 64, 8).permute(1, 0, 2).contiguous(),
           "lc": l32.view(n, k // 64, 8, 2).permute(1, 0, 2, 3).contiguous(),
           "scale": torch.pow(2.0, (base - 112).double()).float().contiguous(), "esc": {}, "n_esc": int(rows.numel()),
           "base": base}
    order = torch.argsort(rows * k + cols)
    rows, cols, f32 = rows[order], cols[order], f32[order]
    # the kernel's table: entries (global row, k, f32 bits, 0) sorted by (row, k), offsets per 8-row group (= the
    # epilogue's n8 accumulator group): entries of group g are ent8[offs8[g]:offs8[g + 1]]
    g8 = rows // 8
    counts8 = torch.bincount(g8, minlength=n // 8)
    offs8 = torch.zeros(n // 8 + 1, dtype=torch.int64, device=w.device)
    offs8[1:] = counts8.cumsum(0)
    out["offs8"] = as_int32(offs8).contiguous()
    out["ent8"] = as_int32(torch.stack((rows, cols, f32, torch.zeros_like(cols)), dim=-1)).contiguous().view(-1, 4)
    # per-BN tables for the kernel's epilogue (PF45): entries of n-block b and lane class t = ((row % 8) >> 1) are
    # et[bn][1][offs[4 b + t]:offs[4 b + t + 1]] = (row in block, k, f32 bits, 0), sorted by (block, t, row, k)
    out["et"] = {}
    for bn in bn_list:
        if n % bn:
            continue
        blk = rows // bn
        rib = rows - blk * bn
        tcls = (rib & 7) >> 1
        key = blk * 4 + tcls
        order_t = torch.argsort(key * (n * k) + rib * k + cols)
        cnt = torch.bincount(key[order_t], minlength=(n // bn) * 4)
        offs_t = torch.zeros((n // bn) * 4 + 1, dtype=torch.int64, device=w.device)
        offs_t[1:] = cnt.cumsum(0)
        ent_t = torch.stack((rib[order_t], cols[order_t], f32[order_t], torch.zeros_like(cols)), dim=-1)
        out["et"][bn] = (as_int32(offs_t).contiguous(), as_int32(ent_t).contiguous().view(-1, 4))
    for bn in bn_list:
        if n % bn:
            continue
        blk = rows // bn
        counts = torch.bincount(blk, minlength=n // bn)
        offs = torch.zeros(n // bn + 1, dtype=torch.int64, device=w.device)
        offs[1:] = counts.cumsum(0)
        ent = torch.stack((rows - blk * bn, cols, f32, torch.zeros_like(cols)), dim=-1)
        out["esc"][bn] = (as_int32(offs).contiguous(), as_int32(ent).contiguous().view(-1, 4))
    return out


def dg12_unpack_reference(pack, n, k, decode):
    """FP64 W [N, K] rebuilt from the planes with a decode function (parsed from the CUDA text by the checks):
    placeholder 2^-15 at escapes in the scaled domain, then the escape corrections, then the per-row scale."""
    q = pack["q"].to(torch.int64) & 0xFFFFFFFF
    l = pack["l"].to(torch.int64) & 0xFFFFFFFF
    regs = decode(q, l[..., 0], l[..., 1])                                 # 4 x [n, k/8] uint32
    el = torch.stack([(regs[e >> 1].to(torch.int64) >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    f = ((el.view(n, k) << 16) & 0xFFFFFFFF).to(torch.int32).view(torch.float32).double()
    return f

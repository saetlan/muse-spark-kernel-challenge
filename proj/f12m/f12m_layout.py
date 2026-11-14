"""F12M: the F12V 12-bit format (per-row 16-exponent window, 4-bit H info + L byte per weight, exact
escapes from a per-row record) laid out in mma.sync m16n8k16 A-fragment order for the tensor-core
kernel at m = 2..32 (proj/f12m/f12m.py).

Order: p9_layout.p9_pack (K permuted per 64-wide chunk by k_perm so a lane's B values for a chunk are 16
consecutive x; lane (g, t) of tile rt, k16 step ks holds elements e = 0..7 with register j = e >> 1 =
(e_{2j}, e_{2j+1}) at row 16 rt + g + 8 (j & 1), packed column 16 ks + 2 t + 8 (j >> 1) + (e & 1)).
Each lane's 8 elements of a step are one F12V group (f12v_layout.f12v_words: Q, L0, L1), so the decode
to the four BF16x2 A registers is 7 integer ops:
  r4 = rotl(Q, 4); HA = (Q & 0x87878787) | 0x38383838 (H bytes of e0..e3); HB = same of r4 (e4..e7)
  reg0 = prmt(L0, HA, 0x5140), reg1 = prmt(L0, HA, 0x7362), reg2/3 = same with L1, HB
  BF16 = H << 8 | L = (-1)^s 1.m 2^(112 + c - 127) = w * 2^(112 - base(row))
so each output row is multiplied by 2^(base - 112) after the MMAs (exact), then gets its escapes.

Planes. P [RT][K/64][3][32][4] int32: per 16-row tile and 64-wide chunk, slot 0 = Q words of steps
0..3, slot 1 = L0(0), L1(0), L0(1), L1(1), slot 2 = steps 2, 3 (as F12V with group = step); each slot a
contiguous 512 bytes. Record RC [N][8] int32 as F12V (base | count << 8, then k | true bits << 16 with
k in the original, unpermuted order). More than 7 escapes in a row: no planes (other kernels serve).
"""

import torch

from f12v_layout import as_int32, f12v_bases, f12v_records, f12v_words
from p9_layout import k_perm, p9_pack


def f12m_encode(w, natural=False):
    """BF16 W [N, K] (N % 16 == 0, K % 64 == 0) -> dict: P int64 words [RT, K/64, 3, 32, 4], base [N],
    escapes (row, original k, true bits) sorted by (row, k)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0, (w.shape, w.dtype)
    bits_w = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    base = f12v_bases((bits_w >> 7) & 0xFF)                    # [N], from the unpermuted rows
    if natural:   # same fragment order without k_perm (B from shared memory reads natural k: WGF)
        frag = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 32, 8)
    else:
        frag = p9_pack(w)                                      # [RT, KS, 32, 8]
    rt_n, ks_n = frag.shape[0], frag.shape[1]
    fb = frag.view(torch.int16).to(torch.int64) & 0xFFFF
    dev = w.device
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1)
    el = torch.arange(8, device=dev).view(1, 1, 1, 8)
    g, t, j = lane >> 2, lane & 3, el >> 1
    row = 16 * torch.arange(rt_n, device=dev).view(-1, 1, 1, 1) + g + 8 * (j & 1)      # [RT, 1, 32, 8]
    col = 16 * torch.arange(ks_n, device=dev).view(1, -1, 1, 1) + 2 * t + 8 * (j >> 1) + (el & 1)
    s, e, m = (fb >> 15) & 1, (fb >> 7) & 0xFF, fb & 0x7F
    c = e - base[row]
    esc = (c < 0) | (c > 15)
    zero = torch.zeros_like(c)
    s, c, m = torch.where(esc, zero, s), torch.where(esc, zero, c), torch.where(esc, zero, m)
    q, l0, l1 = f12v_words(s, c, m)                            # [RT, KS, 32]
    ch = ks_n // 4
    qv, l0v, l1v = (v.view(rt_n, ch, 4, 32) for v in (q, l0, l1))   # rt, chunk, step, lane
    slot0 = qv.permute(0, 1, 3, 2)
    slot1 = torch.stack((l0v[:, :, 0], l1v[:, :, 0], l0v[:, :, 1], l1v[:, :, 1]), dim=-1)
    slot2 = torch.stack((l0v[:, :, 2], l1v[:, :, 2], l0v[:, :, 3], l1v[:, :, 3]), dim=-1)
    p = torch.stack((slot0, slot1, slot2), dim=2)              # [RT, ch, 3, 32, 4]
    kp = torch.arange(k, device=dev) if natural else k_perm(k).to(dev)
    rows = row.expand_as(esc)[esc]
    ks_ = kp[col.expand_as(esc)[esc]]
    tbits = fb[esc]
    order = torch.argsort(rows * k + ks_)
    return {"p": p, "base": base, "esc_row": rows[order], "esc_k": ks_[order], "esc_bits": tbits[order],
            "n_esc": int(esc.sum()), "n": n * k}


def f12m_device_planes(w, natural=False):
    """BF16 W [N, K] -> {"p": int32 [RT, K/64, 3, 32, 4], "rec": int32 [N, 8], "rate", "max_row"}, or None
    when a row has more than 7 escapes."""
    n = w.shape[0]
    enc = f12m_encode(w, natural)
    rec = f12v_records(enc, n)
    if rec is None:
        return None
    rows = enc["esc_row"]
    max_row = int(torch.bincount(rows, minlength=n).max()) if len(rows) else 0
    return {"p": as_int32(enc["p"]).contiguous(), "rec": as_int32(rec).contiguous(),
            "rate": enc["n_esc"] / enc["n"], "max_row": max_row}


"""P22 (F12): lossless 12-bit storage of BF16 weights for the P9 kernel, decoded in registers.

Starts from P9's packed A-fragment stream (p9_layout.p9_pack: K permuted per 64-chunk, one lane's
8 bf16 per k16 step, elements e0..e7 with register r = (e_{2r}, e_{2r+1})). Per weight, with bits
b = [s:1][e:8][m:7]:
  raw byte  r = s << 7 | m
  code      c = top - e if 0 <= top - e <= 14, else 15 (escape; then r = 0)
where top is chosen per 16-row tile (the 15-exponent window [top-14, top] holding the most weights,
top >= 15 so the packed subtraction below never borrows). An escape decodes to the bf16 with
exponent top - 15, sign 0, mantissa 0 (w_dec); the exact value comes back in the epilogue as
acc += x[k] * w_true - x[k] * w_dec from a per-row list (CSR, original k).

Storage per 16-row tile and 64-wide chunk (4 k16 steps), per lane:
  Q  one uint4: word j = step j's 8 codes, low nibbles of bytes 0..3 = c0..c3, high nibbles = c4..c7
  R  two uint4: uint4 h holds steps 2h and 2h+1, each as words [RA, RB] with RA = r0..r3, RB = r4..r7
  12 bytes per 8 weights (vs 16): 0.75x the bytes.
Decode of one step (all 32-bit ops; the CUDA code does exactly this):
  CA = Q & 0x0F0F0F0F, CB = (Q >> 4) & 0x0F0F0F0F, EA = TOP4 - CA, EB = TOP4 - CB
  HI = (R & 0x80808080) | ((E >> 1) & 0x7F7F7F7F), LO = ((E << 7) & 0x80808080) | (R & 0x7F7F7F7F)
  reg0 = prmt(LOA, HIA, 0x5140), reg1 = prmt(LOA, HIA, 0x7362), reg2/3 = same with B.
"""

import torch

from p9_layout import k_perm, p9_pack

ESC = 15
M32 = 0xFFFFFFFF


def choose_top(e_tile):
    """e_tile: int64 exponents of one 16-row tile -> top of the best 15-exponent window (>= 15)."""
    hist = torch.bincount(e_tile.flatten(), minlength=256).double()
    win = torch.nn.functional.conv1d(hist.view(1, 1, -1), torch.ones(1, 1, 15, dtype=torch.float64, device=hist.device)).flatten()
    start = int(win.argmax())          # window [start, start + 14]
    return max(start + 14, 15)


def p22_encode(frag, k_perm_idx):
    """frag: p9_pack output [RT, KS, 32, 8] bf16 (K % 64 == 0). Returns dict of int32 planes, tile tops,
    and the escape list in original (unpermuted) K, plus counts."""
    rt, ks = frag.shape[0], frag.shape[1]
    assert ks % 4 == 0
    b = frag.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF            # [RT, KS, 32, 8]
    e = (b >> 7) & 0xFF
    r = ((b >> 15) << 7) | (b & 0x7F)
    dev = frag.device
    tops = torch.tensor([choose_top(e[i]) for i in range(rt)], dtype=torch.int64, device=dev)
    c = tops.view(rt, 1, 1, 1) - e
    esc = (c < 0) | (c > 14)
    c = torch.where(esc, torch.full_like(c, ESC), c)
    r = torch.where(esc, torch.zeros_like(r), r)
    # Q word per (tile, step, lane): low nibbles c0..c3, high nibbles c4..c7.
    q = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        q |= c[..., i] << (8 * i)
        q |= c[..., 4 + i] << (8 * i + 4)
    ra = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    rb = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        ra |= r[..., i] << (8 * i)
        rb |= r[..., 4 + i] << (8 * i)
    chunks = ks // 4
    # Q plane [RT, chunks, 32, 4]: word j = step 4 chunk + j.
    qp = q.view(rt, chunks, 4, 32).permute(0, 1, 3, 2)
    # R plane [RT, chunks, 2, 32, 4]: uint4 h = [RA(2h), RB(2h), RA(2h+1), RB(2h+1)].
    rr = torch.stack((ra, rb), dim=-1).view(rt, chunks, 2, 2, 32, 2)          # rt, ch, h, step-in-h, lane, A/B
    rp = rr.permute(0, 1, 2, 4, 3, 5).reshape(rt, chunks, 2, 32, 4)
    # Escape list (row, original k, true bits).
    ti, si, li, ei = esc.nonzero(as_tuple=True)
    g, t = li >> 2, li & 3
    reg = ei >> 1
    row = 16 * ti + g + 8 * (reg & 1)
    col = 2 * t + 8 * (reg >> 1) + (ei & 1)
    korig = k_perm_idx.to(dev)[16 * si + col]
    true_bits = b[ti, si, li, ei]
    to32 = lambda v: (v & M32).to(torch.int64)
    return {"q": to32(qp), "r": to32(rp), "tops": tops, "esc_row": row, "esc_k": korig, "esc_bits": true_bits,
            "n_esc": int(esc.sum()), "n": int(esc.numel())}


def prmt(a, b, sel):
    """PTX prmt.b32 default mode for selector nibbles 0..7: byte n = byte sel_n of {b:a}."""
    src = (b << 32) | a
    out = torch.zeros_like(a)
    for n in range(4):
        idx = (sel >> (4 * n)) & 0x7
        out |= ((src >> (8 * idx)) & 0xFF) << (8 * n)
    return out


def p22_decode_step(q, ra, rb, top):
    """The kernel's decode of one step for tensors of words (int64 holding uint32). Returns 4 registers."""
    top4 = (top * 0x01010101) & M32
    ca, cb = q & 0x0F0F0F0F, (q >> 4) & 0x0F0F0F0F
    ea, eb = (top4 - ca) & M32, (top4 - cb) & M32
    hia = (ra & 0x80808080) | ((ea >> 1) & 0x7F7F7F7F)
    loa = ((ea << 7) & 0x80808080) | (ra & 0x7F7F7F7F)
    hib = (rb & 0x80808080) | ((eb >> 1) & 0x7F7F7F7F)
    lob = ((eb << 7) & 0x80808080) | (rb & 0x7F7F7F7F)
    return prmt(loa, hia, 0x5140), prmt(loa, hia, 0x7362), prmt(lob, hib, 0x5140), prmt(lob, hib, 0x7362)


def p22_decode_all(enc):
    """Decode every fragment from the planes exactly as the kernel indexes them -> [RT, KS, 32, 8] bits."""
    q, r, tops = enc["q"], enc["r"], enc["tops"]
    rt, chunks = q.shape[0], q.shape[1]
    out = torch.zeros(rt, chunks * 4, 32, 8, dtype=torch.int64)
    for j in range(4):
        qw = q[:, :, :, j]                                   # Q uint4 word j
        h, w0 = j >> 1, 2 * (j & 1)
        ra, rb = r[:, :, h, :, w0], r[:, :, h, :, w0 + 1]    # R uint4 h, words w0 and w0 + 1
        regs = p22_decode_step(qw, ra, rb, tops.view(rt, 1, 1))
        for reg, v in enumerate(regs):
            out[:, j::4, :, 2 * reg] = v & 0xFFFF
            out[:, j::4, :, 2 * reg + 1] = (v >> 16) & 0xFFFF
    return out


def p22_dec_bits(top):
    """Bits an escape decodes to: exponent top - 15, sign 0, mantissa 0."""
    return (top - 15) << 7


P22_MAX_ESCAPE_RATE = 0.005


def p22_device_planes(w, n_rows):
    """BF16 W [N, K] -> kernel tensors on W's device: q [RT, K/64, 32, 4] int32, r [RT, K/64, 2, 32, 4]
    int32, tops [RT] uint8, CSR erow [N+1] int32 / ecol int32 / ebits int16 (sorted by row), or None when
    escapes exceed P22_MAX_ESCAPE_RATE (that layer then stays on uncompressed kernels)."""
    n, k = w.shape
    enc = p22_encode(p9_pack(w), k_perm(k))
    if enc["n_esc"] > P22_MAX_ESCAPE_RATE * enc["n"]:
        return None
    order = torch.argsort(enc["esc_row"] * k + enc["esc_k"])
    rows, cols, bits = enc["esc_row"][order], enc["esc_k"][order], enc["esc_bits"][order]
    erow = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    erow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    as32 = lambda t: t.to(torch.int64).where(t < 2**31, t - 2**32).to(torch.int32).contiguous()
    return {"q": as32(enc["q"]), "r": as32(enc["r"]), "tops": enc["tops"].to(torch.uint8).contiguous(),
            "erow": erow.to(torch.int32).contiguous(), "ecol": cols.to(torch.int32).contiguous(),
            "ebits": bits.where(bits < 2**15, bits - 2**16).to(torch.int16).contiguous(),
            "rate": enc["n_esc"] / enc["n"]}

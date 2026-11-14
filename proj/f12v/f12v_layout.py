"""F12V: lossless 12-bit storage of BF16 weights for the m = 1 CUDA-core GEMV (proj/f12v/f12v.py),
decoded in registers straight to FP32 with 13 integer ops per 8 weights.

Per row a window of 16 exponents [base, base + 15], base chosen per row to hold the most weights
(1 <= base <= 239, so every scale below is a normal FP32 number). A weight with bits [s:1][e:8][m:7]
and c = e - base in [0, 15] is stored as
  H info  4 bits: s and c >> 1 (3 bits)
  L byte  8 bits: (c & 1) << 7 | m
Escapes (e outside the window: zeros, subnormals, Inf/NaN, rare outliers) are stored as s = c = m = 0
and listed in the row's record with their true bits.

Decode (the kernel's 13 ops per 8 weights, check_f12v.py runs the CUDA text itself on every input):
  H byte = s << 7 | 0x38 | (c >> 1)    built for 2 weights at once by (Q & K1) | K2, with zero bytes left
                                       between them
  FP32   = H << 24 | L << 16           one __byte_perm per weight (L byte from an L word, H byte and two
                                       zero bytes from the H word)
         = (-1)^s 1.m 2^(112 + c - 127) = w * 2^(112 - base)
The dot product is accumulated in that scaled domain and multiplied once per row by 2^(base - 112)
(a power of two: exact). Escapes then get  sum += -x[k] * w_dec + x[k] * w_true  with
w_dec = 2^(base - 127), what the stored zero pattern decodes to.

Layout. P [N][K/1024][3][32][4] int32: lane l of 1024-wide k block kb holds the 32 weights
k = 1024 kb + 32 l + 8 g + i (group g = 0..3, i = 0..7), as three uint4 (slots), each warp's slot a
contiguous 512 bytes:
  slot 0: word g = Q(g), the 8 H infos of group g;  slot 1: L0(0), L1(0), L0(1), L1(1);
  slot 2: L0(2), L1(2), L0(3), L1(3);  L0 = L bytes of weights 0..3, L1 = weights 4..7.
  Q bits: weight i < 4 has s at bit 8 i + 7 and c >> 1 at bits 8 i + 0..2; weight 4 + j has s at bit
  8 j + 3 and c >> 1 at bits (8 j - 4) mod 32 + 0..2 (they land at 8 j + 7 and 8 j + 0..2 after the
  kernel's rotate left by 4).
Record RC [N][8] int32: word 0 = base | count << 8, words 1..7 = escapes (k | true bf16 bits << 16),
in increasing k. A row with more than 7 escapes means the layer stays on uncompressed kernels.

Functions run on the GPU in prepare() and on the CPU in the local checks; all integer ops, exact.
"""

import torch

F12V_SLOTS = 7          # escapes per row in the record
F12V_KBLOCK = 1024      # k per lane-block: 32 lanes x 32 weights
M32 = 0xFFFFFFFF


def f12v_bases(e):
    """e: [N, K] int64 exponents -> [N] int64 base of the 16-exponent window holding the most weights
    (first such window), within [1, 239]."""
    n = e.shape[0]
    hist = torch.zeros(n, 256, dtype=torch.int64, device=e.device)
    hist.scatter_add_(1, e, torch.ones_like(e))
    pad = torch.zeros(n, 257, dtype=torch.int64, device=e.device)
    pad[:, 1:] = hist.cumsum(1)                       # pad[:, j] = count of e < j
    win = pad[:, 16:257] - pad[:, 0:241]              # win[:, b] = count of e in [b, b + 15], b = 0..240
    win[:, 0] = -1
    win[:, 240] = -1
    return win.argmax(1)


def f12v_fields(w):
    """BF16 W [N, K] -> int64 fields (s, c, m, esc, bits, base)."""
    b = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    s, e, m = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F
    base = f12v_bases(e)
    c = e - base[:, None]
    esc = (c < 0) | (c > 15)
    zero = torch.zeros_like(c)
    return (torch.where(esc, zero, s), torch.where(esc, zero, c), torch.where(esc, zero, m), esc, b, base)


def f12v_words(s, c, m):
    """Fields [..., 8] of one group of 8 weights -> (Q, L0, L1) int64 words."""
    h = c >> 1
    q = torch.zeros(s.shape[:-1], dtype=torch.int64, device=s.device)
    for i in range(4):
        q |= (s[..., i] << (8 * i + 7)) | (h[..., i] << (8 * i))
        j = i
        q |= (s[..., 4 + j] << (8 * j + 3)) | (h[..., 4 + j] << ((8 * j - 4) % 32))
    lb = ((c & 1) << 7) | m
    l0 = lb[..., 0] | (lb[..., 1] << 8) | (lb[..., 2] << 16) | (lb[..., 3] << 24)
    l1 = lb[..., 4] | (lb[..., 5] << 8) | (lb[..., 6] << 16) | (lb[..., 7] << 24)
    return q, l0, l1


def f12v_encode(w):
    """BF16 W [N, K] (K % 1024 == 0) -> dict: P int64 words [N, K/1024, 3, 32, 4], base [N], and the
    escapes (row, k, true bits) sorted by (row, k)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % F12V_KBLOCK == 0, (w.shape, w.dtype)
    s, c, m, esc, bits, base = f12v_fields(w)
    kb = k // F12V_KBLOCK
    shape = (n, kb, 32, 4, 8)                         # row, kb, lane, group, i
    q, l0, l1 = f12v_words(s.view(shape), c.view(shape), m.view(shape))   # [n, kb, 32, 4]
    slot0 = q
    slot1 = torch.stack((l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]), dim=-1)
    slot2 = torch.stack((l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]), dim=-1)
    p = torch.stack((slot0, slot1, slot2), dim=2)     # [n, kb, 3, 32, 4]
    rows, cols = esc.nonzero(as_tuple=True)           # row-major order: sorted by (row, k)
    return {"p": p, "base": base, "esc_row": rows, "esc_k": cols, "esc_bits": bits[rows, cols],
            "n_esc": int(esc.sum()), "n": n * k}


def as_int32(t):
    """int64 holding uint32 values -> int32 with the same bits."""
    t = t & M32
    return torch.where(t >= 2**31, t - 2**32, t).to(torch.int32)


def f12v_records(enc, n):
    """[N, 8] int64 record words, or None if a row has more than F12V_SLOTS escapes."""
    rows, cols, bits = enc["esc_row"], enc["esc_k"], enc["esc_bits"]
    dev = enc["base"].device
    count = torch.bincount(rows, minlength=n) if len(rows) else torch.zeros(n, dtype=torch.int64, device=dev)
    if len(rows) and int(count.max()) > F12V_SLOTS:
        return None
    rec = torch.zeros(n, 1 + F12V_SLOTS, dtype=torch.int64, device=dev)
    rec[:, 0] = enc["base"] | (count << 8)
    if len(rows):
        start = torch.zeros(n, dtype=torch.int64, device=dev)
        start[1:] = count.cumsum(0)[:-1]
        rank = torch.arange(len(rows), device=dev) - start[rows]
        rec[rows, 1 + rank] = cols | (bits << 16)
    return rec


def f12v_device_planes(w):
    """BF16 W [N, K] -> {"p": int32 [N, K/1024, 3, 32, 4], "rec": int32 [N, 8], "rate", "max_row"} on W's
    device, or None when a row has more than F12V_SLOTS escapes (that layer keeps its other kernels)."""
    n = w.shape[0]
    enc = f12v_encode(w)
    rec = f12v_records(enc, n)
    if rec is None:
        return None
    rows = enc["esc_row"]
    max_row = int(torch.bincount(rows, minlength=n).max()) if len(rows) else 0
    return {"p": as_int32(enc["p"]).contiguous(), "rec": as_int32(rec).contiguous(),
            "rate": enc["n_esc"] / enc["n"], "max_row": max_row}

"""PyTorch ports of R14's kernels (v4/r14.py) with the kernels' own index formulas, shared by
check_v4_cpu.py (correctness at the real sizes) and dryrun_lab_re4.py (lab control flow)."""

import torch

H = I = 4096
KC = 64


def bf16_from_bits(bits16):
    return bits16.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def dec(wd, r):
    """dec<R>: int64 tensor of 32-bit words -> (lane0 bits, lane1 bits) of value * 2^-126."""
    x = (wd << (4 * r)) & 0xFFFFFFFF
    v = (x & 0x80008000) | ((x >> 6) & 0x01C001C0)
    return v & 0xFFFF, (v >> 16) & 0xFFFF


def rotr(wd, n):
    return ((wd >> n) | (wd << (32 - n))) & 0xFFFFFFFF if n else wd


def dec_v2(wd, r):
    """dec_k16_v2 (R47 layout), register r: the kernel's rotate/mask formulas."""
    if r == 0:
        v = wd & 0x81C081C0
    elif r == 1:
        v = rotr(wd, 3) & 0x81C081C0
    elif r == 2:
        v = (rotr(wd, 1) & 0x80008000) | (rotr(wd, 6) & 0x01C001C0)
    else:
        v = (rotr(wd, 2) & 0x80008000) | (rotr(wd, 13) & 0x01C001C0)
    return v & 0xFFFF, (v >> 16) & 0xFFFF


def hmul2_lane(bits_a, bits_s):
    """mul.rn.bf16x2 on one lane: bf16 * bf16 -> bf16 (exact product in fp32, then one RNE rounding)."""
    return (bf16_from_bits(bits_a).float() * bf16_from_bits(bits_s).float()).to(torch.bfloat16)


def decode_blob(blob_e, layout=0):
    """[NB, KC, 5120] -> fp32 [NB*128 rows in blob order, KC*64] exactly as the consumers decode it."""
    nb_count = blob_e.shape[0]
    words = blob_e[..., :4096].contiguous().view(torch.int32).view(nb_count, KC, 2, 4, 32, 4).to(torch.int64) & 0xFFFFFFFF
    sw = blob_e[..., 4096:].contiguous().view(torch.int32).view(nb_count, KC, 2, 4, 8, 4).to(torch.int64) & 0xFFFFFFFF
    # thread (c, w, lane): soff picks g = lane >> 2 -> broadcast sw over q
    lane = torch.arange(32)
    g, q = lane // 4, lane % 4
    swl = sw[:, :, :, :, g, :]  # [NB, KC, 2, 4, 32, 4]
    s_lo, s_hi = swl & 0xFFFF, (swl >> 16) & 0xFFFF  # __byte_perm 0x1010 / 0x3232 lanes
    out = torch.zeros(nb_count * 128, KC * 64)
    # a[i] for i = 0..7: registers r0 = (a0, a1) = dec<0> * sg, r1 = (a2, a3) = dec<1> * sg8, r2 = dec<2> * sg, r3 = dec<3> * sg8
    vals = {}
    for r, s in ((0, s_lo), (1, s_hi), (2, s_lo), (3, s_hi)):
        l0, l1 = dec(words, r) if layout == 0 else dec_v2(words, r)
        vals[2 * r] = hmul2_lane(l0, s).float()
        vals[2 * r + 1] = hmul2_lane(l1, s).float()
    nb_i = torch.arange(nb_count).view(-1, 1, 1, 1, 1, 1)
    kc_i = torch.arange(KC).view(1, -1, 1, 1, 1, 1)
    c_i = torch.arange(2).view(1, 1, -1, 1, 1, 1)
    w_i = torch.arange(4).view(1, 1, 1, -1, 1, 1)
    g_i = g.view(1, 1, 1, 1, -1, 1)
    q_i = q.view(1, 1, 1, 1, -1, 1)
    ks_i = torch.arange(4).view(1, 1, 1, 1, 1, -1)
    for i in range(8):
        row = nb_i * 128 + c_i * 64 + w_i * 16 + g_i + 8 * ((i >> 1) & 1)
        k = kc_i * 64 + ks_i * 16 + 2 * q_i + (i & 1) + 8 * (i >> 2)
        row, k = torch.broadcast_tensors(row, k)
        out[row.reshape(-1), k.reshape(-1)] = vals[i].reshape(-1)
    return out


def route(ids, max_pieces, mode=0):
    """Port of route_kernel (sequential row order inside segments; the kernel's atomics give any order)."""
    cnt = [0] * 32
    for e in ids.tolist():
        if e >= 0:
            cnt[e] += 1
    start = [0] * 33
    pieces = []
    for e in range(32):
        start[e + 1] = start[e] + ((cnt[e] + 16 - 1) // 16) * 16
        rem, r = cnt[e], start[e]
        while rem > 0:
            cap = 64 if mode == 2 else 128  # route_kernel: cap = mode == 2 ? 64 : 128
            if mode:
                n = cap if rem >= cap else -(-rem // 16) * 16
            else:
                n = 128 if rem >= 128 else 64 if rem >= 64 else 32 if rem >= 32 else 16
            v = rem if rem < n else n
            pieces.append((e, r, v, n))
            r += n
            rem -= v
    assert len(pieces) <= max_pieces
    cur = start[:32]
    pos = []
    for e in ids.tolist():
        if e >= 0:
            pos.append(cur[e])
            cur[e] += 1
        else:
            pos.append(-1)
    return torch.tensor(pos, dtype=torch.int32), pieces, start, cnt


def gather(x, pos, rows):
    """Port of _r14_gather_kernel: out[kc, p, (u ^ (p & 7)) * 8 + e] = x[token, kc*64 + u*8 + e]."""
    out = torch.zeros(KC, rows, 64, dtype=torch.bfloat16)
    offs = torch.arange(H)
    kc, u, e8 = offs // 64, (offs // 8) % 8, offs % 8
    for i, p in enumerate(pos.tolist()):
        if p < 0:
            continue
        out[kc, p, (u ^ (p & 7)) * 8 + e8] = x[i // 8]
    return out


def read_b(src, row0, nsel, kc, rows):
    """What wgmma reads as B[n, k] (n < nsel, k < 64) from the stage the producer copied:
    bytes [(kc*P + row0)*128, +nsel*128) of src, addressed as start + ks*32 + (n//8)*1024 + (n%8)*128
    + (kk//8)*16 + (kk%8)*2 with address bits [4:6] XOR [7:9] (tile base 1024-aligned)."""
    tile = src[kc].reshape(-1)[row0 * 64:(row0 + nsel) * 64]  # contiguous rows row0.. (bulk copy)
    n = torch.arange(nsel).view(-1, 1, 1)
    ks = torch.arange(4).view(1, -1, 1)
    kk = torch.arange(16).view(1, 1, -1)
    addr = ks * 32 + (n // 8) * 1024 + (n % 8) * 128 + (kk // 8) * 16 + (kk % 8) * 2
    phys = addr ^ (((addr >> 7) & 7) << 4)
    return tile[(phys // 2).reshape(-1)].view(nsel, 64)


def gemm(blob_dec, src, e, nb_count, gated, cs, out, rows, pieces):
    """Every CTA (nb, piece): acc = A_block @ B^T over K, then the kernel's epilogue formulas."""
    for (pe, row0, valid, nsel) in pieces:
        if pe != e:
            continue
        B = torch.cat([read_b(src, row0, nsel, kc, rows) for kc in range(KC)], dim=1).float()  # [nsel, K]
        for nb in range(nb_count):
            A = blob_dec[nb * 128:(nb + 1) * 128]  # [128, K] in blob row order
            acc = A @ B.T  # [128 rows (c*64 + w*16 + half*8 + g), nsel tokens]
            if gated:
                stage = torch.zeros(nsel * 128, dtype=torch.uint8)
                for c in range(2):
                    for w in range(4):
                        for g in range(8):
                            col = c * 32 + w * 8 + g
                            gt = acc[c * 64 + w * 16 + g] * cs
                            up = acc[c * 64 + w * 16 + g + 8] * cs
                            v = (gt / (1.0 + torch.exp(-gt)) * up).to(torch.bfloat16)
                            t = torch.arange(nsel)
                            off = t * 128 + (((col >> 3) ^ (t & 7)) << 4) + (col & 7) * 2
                            vb = v.view(torch.int16).to(torch.int32)
                            stage[off] = (vb & 0xFF).to(torch.uint8)
                            stage[off + 1] = ((vb >> 8) & 0xFF).to(torch.uint8)
                nrows = (valid + 7) & ~7
                dst = out[nb].reshape(-1).view(torch.uint8)
                dst[row0 * 128:(row0 + nrows) * 128] = stage[:nrows * 128]
            else:
                for c in range(2):
                    for w in range(4):
                        for g in range(8):
                            r = c * 64 + w * 16 + g
                            out[row0:row0 + valid, nb * 128 + r] = (acc[r, :valid] * cs).to(torch.bfloat16)
                            out[row0:row0 + valid, nb * 128 + r + 8] = (acc[r + 8, :valid] * cs).to(torch.bfloat16)


def gemm_wide(blob_dec, src, e, nb_count, gated, cs, out, rows, pieces):
    """R54 wide_kernel: CTA (nb2, piece) with nsel <= 64; consumer thread tid < 512: warpgroup C = tid >> 7,
    c = (tid >> 7) & 1, h = tid >> 8; weights of block 2 * nb2 + h (wsrc0 = block 2 * nb2, wsrc1 = wsrc0 + KC blobs),
    output block nb = 2 * nb2 + h; R14's consume() math per (nb, c). The two warpgroups of a block then run R14's
    epilogue on their 128 rows (same formulas as gemm())."""
    for (pe, row0, valid, nsel) in pieces:
        if pe != e:
            continue
        assert nsel <= 64, "wide_kernel flags N > 64"
        B = torch.cat([read_b(src, row0, nsel, kc, rows) for kc in range(KC)], dim=1).float()
        for nb2 in range(nb_count // 2):
            accs = {}
            for tid in (0, 128, 256, 384):
                c, h = (tid >> 7) & 1, tid >> 8
                blk = 2 * nb2 + h  # wsrc0 + h * KC * WS_BYTES
                A = blob_dec[blk * 128 + c * 64:blk * 128 + c * 64 + 64]
                accs[(2 * nb2 + h, c)] = A @ B.T
            for nb in (2 * nb2, 2 * nb2 + 1):
                acc = torch.cat((accs[(nb, 0)], accs[(nb, 1)]), 0)
                _epilogue(acc, nb, gated, cs, out, row0, valid, nsel)


def _epilogue(acc, nb, gated, cs, out, row0, valid, nsel):
    if gated:
        stage = torch.zeros(nsel * 128, dtype=torch.uint8)
        for c in range(2):
            for w in range(4):
                for g in range(8):
                    col = c * 32 + w * 8 + g
                    gt = acc[c * 64 + w * 16 + g] * cs
                    up = acc[c * 64 + w * 16 + g + 8] * cs
                    v = (gt / (1.0 + torch.exp(-gt)) * up).to(torch.bfloat16)
                    t = torch.arange(nsel)
                    off = t * 128 + (((col >> 3) ^ (t & 7)) << 4) + (col & 7) * 2
                    vb = v.view(torch.int16).to(torch.int32)
                    stage[off] = (vb & 0xFF).to(torch.uint8)
                    stage[off + 1] = ((vb >> 8) & 0xFF).to(torch.uint8)
        nrows = (valid + 7) & ~7
        dst = out[nb].reshape(-1).view(torch.uint8)
        dst[row0 * 128:(row0 + nrows) * 128] = stage[:nrows * 128]
    else:
        for c in range(2):
            for w in range(4):
                for g in range(8):
                    r = c * 64 + w * 16 + g
                    out[row0:row0 + valid, nb * 128 + r] = (acc[r, :valid] * cs).to(torch.bfloat16)
                    out[row0:row0 + valid, nb * 128 + r + 8] = (acc[r + 8, :valid] * cs).to(torch.bfloat16)



def gemm_swap(blob_dec, src, e, nb_count, gated, cs, out, rows, pieces):
    """R55 dq_mma_swap + store_out_t: MMA warpgroup m (active if m == 0 or nsel > 64) multiplies tokens 64m..64m+63
    (A rows read from the token slot at +m*8192; rows past nsel are stale, here NaN, never stored) by the 128-row
    decoded weight tile (B, N = 128); D[t][n]; thread (m, w, lane): t = 64m + 16w + g + 8r, n = 8j + 2q + h <-
    acc[4j + 2r + h]; gated: gate n = 16i + 2q + h (acc[8i + 2r + h]), up n + 8 (acc[8i + 4 + 2r + h]), column
    8i + 2q + h; then R14's copy loops. Vectorized over (w, g, r, q, h, i|j)."""
    w_, g_, r_, q_, h_ = torch.meshgrid(torch.arange(4), torch.arange(8), torch.arange(2), torch.arange(4),
                                        torch.arange(2), indexing="ij")
    for (pe, row0, valid, nsel) in pieces:
        if pe != e:
            continue
        tok = torch.full((128, KC * 64), float("nan"))
        tok[:nsel] = torch.cat([read_b(src, row0, nsel, kc, rows) for kc in range(KC)], dim=1).float()
        for nb in range(nb_count):
            W = blob_dec[nb * 128:(nb + 1) * 128]
            stage = torch.zeros(128 * 256, dtype=torch.uint8)
            for m in (0, 1):
                if not (m == 0 or nsel > 64):
                    continue
                D = tok[64 * m:64 * m + 64] @ W.T  # [64 tokens, 128 weight rows]
                lrow = (16 * w_ + g_ + 8 * r_).unsqueeze(-1)   # row inside the warpgroup's 64 tokens
                t = 64 * m + lrow
                if gated:
                    i_ = torch.arange(8)
                    n_gate = 16 * i_ + 2 * q_.unsqueeze(-1) + h_.unsqueeze(-1)
                    gt = D[lrow.expand_as(n_gate), n_gate] * cs
                    up = D[lrow.expand_as(n_gate), n_gate + 8] * cs
                    v = (gt / (1.0 + torch.exp(-gt)) * up).to(torch.bfloat16)
                    col = 8 * i_ + 2 * q_.unsqueeze(-1) + h_.unsqueeze(-1)
                    off = t * 128 + (((col >> 3) ^ (t & 7)) << 4) + (col & 7) * 2
                else:
                    j_ = torch.arange(16)
                    n = 8 * j_ + 2 * q_.unsqueeze(-1) + h_.unsqueeze(-1)
                    v = (D[lrow.expand_as(n), n] * cs).to(torch.bfloat16)
                    off = t * 256 + n * 2
                vb = v.view(torch.int16).to(torch.int32).reshape(-1)
                off = off.expand_as(v).reshape(-1)
                stage[off] = (vb & 0xFF).to(torch.uint8)
                stage[off + 1] = ((vb >> 8) & 0xFF).to(torch.uint8)
            if gated:
                nrows = (valid + 7) & ~7
                dst = out[nb].reshape(-1).view(torch.uint8)
                dst[row0 * 128:(row0 + nrows) * 128] = stage[:nrows * 128]
            else:
                st = stage.view(torch.bfloat16).view(128, 128)
                out[row0:row0 + valid, nb * 128:(nb + 1) * 128] = st[:valid]

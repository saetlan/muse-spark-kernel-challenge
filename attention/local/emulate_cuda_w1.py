"""CPU emulation of attention/local/cuda_w1.py (W1: wgmma FP8 verify), with the kernel's index formulas and
the layouts from the PTX ISA (9.7.17) and CUTLASS (cute/atom/mma_traits_sm90_gmma.hpp):
- shared memory is one byte array with the kernel's slot / tile offsets; the copier writes K rows into
  the K-major tile and V rows into staging by its lane mapping (key 8*kg + (lane & 7), chunk
  4*hf + (lane >> 3)), page entries from their ring;
- the transposers re-lay V into Vt by their formulas (key order 16p + 8*(b>>1) + 2t + (b&1), tr4 with
  the kernel's prmt selectors, 16 B stores at (n%8)*16 + (n/8)*SBO_V + p*LBO_V);
- wgmma.m64n64k32 e4m3: A from each thread's registers by ALayout_64x32 (reg r: row 16w + g + 8*(r&1),
  k = 4*t4 + byte + 16*(r>>1)); B decoded from the descriptor the kernel builds (start / LBO / SBO
  fields, swizzle 0) at start + (n%8)*16 + (n/8)*SBO + (k%16) + (k/16)*LBO; D scattered by CLayout_64xN
  (d[4j + 2*half + c] = row 16w + g + 8*half, col 8j + 2*t4 + c); scale-d 0 / 1;
- consumers: V5's softmax on those accumulators, P packed as the kernel does, O update, partials.
Merged by emulate.py's combine; outputs through the judge's check_implementation.
MUT (mutation_test_cuda_w1.py) plants single bugs.
Run: .venv/bin/python attention/local/emulate_cuda_w1.py [--quick]"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF, e4m3  # noqa: E402
from emulate_cuda_ver4 import CHUNK, SKEYS, P_OFF, e4m3x4, tr4  # noqa: E402

L = 8
KC_HEAD, SBO_K, LBO_K = 4096, 512, 128
VS_PITCH = 144
VS_BYTES = 64 * VS_PITCH
SBO_V, LBO_V = 528, 128
VT_HEAD = 8 * SBO_V
OFF_VS, OFF_VT = 2 * KC_HEAD, 2 * KC_HEAD + VS_BYTES
SLOT = OFF_VT + 2 * VT_HEAD
MUT = set()
DUMP = None   # a dict: attend_w1 fills 'dbg' (cuda_w1dbg.py's layout) for split 0, request 0 x pair 0, stage 0

# ALayout_64x32 / CLayout_64xN index maps (thread = (warp, lane))
_W = torch.arange(4)[:, None]
A_M = torch.zeros(4, 32, 4, 4, dtype=torch.long)
A_K = torch.zeros(4, 32, 4, 4, dtype=torch.long)
for r in range(4):
    for byte in range(4):
        A_M[:, :, r, byte] = 16 * _W + G_OF[None, :] + 8 * (r & 1)
        A_K[:, :, r, byte] = 4 * T_OF[None, :] + byte + 16 * (r >> 1)
C_M = torch.zeros(4, 32, 32, dtype=torch.long)
C_N = torch.zeros(4, 32, 32, dtype=torch.long)
for i in range(32):
    j, half, c = i // 4, (i >> 1) & 1, i & 1
    C_M[:, :, i] = 16 * _W + G_OF[None, :] + 8 * half
    C_N[:, :, i] = 8 * j + 2 * T_OF[None, :] + c


def gdesc(saddr, lbo, sbo):
    return ((saddr & 0x3FFFF) >> 4) | (((lbo & 0x3FFFF) >> 4) << 16) | (((sbo & 0x3FFFF) >> 4) << 32)


def wgmma(acc, a_regs, desc, smem, scale_d):
    """acc [4, 32, 32] f32, a_regs [4, 32, 4, 4] uint8, desc int, smem uint8 [bytes]."""
    start, lbo, sbo = (desc & 0x3FFF) << 4, ((desc >> 16) & 0x3FFF) << 4, ((desc >> 32) & 0x3FFF) << 4
    assert desc >> 62 == 0 and (desc >> 49) & 7 == 0
    A = torch.zeros(64, 32)
    A[A_M.reshape(-1), A_K.reshape(-1)] = e4m3(a_regs.reshape(-1, 4)).reshape(-1)
    n = torch.arange(64)[None, :]
    k = torch.arange(32)[:, None]
    addr = start + (n % 8) * 16 + (n // 8) * sbo + (k % 16) + (k // 16) * lbo
    assert int(addr.max()) < smem.numel()
    B = e4m3(smem[addr.reshape(-1)].reshape(-1, 1)).reshape(32, 64)
    D = A @ B
    out = D[C_M, C_N]
    return out + acc if scale_d else out


def tile_off(m, k):
    """W2's Q / P tiles (make_w2.py tile_off): K-major canonical, LBO 128, SBO 512."""
    return (m & 7) * 16 + (m >> 3) * 512 + (k & 15) + (k >> 4) * 128


def wgmma_ss(acc, desc_a, desc_b, smem, scale_d):
    """SS form (W2): A (64 x 32) from shared memory through its descriptor, read as K-major canonical like B."""
    sa, la, ba = (desc_a & 0x3FFF) << 4, ((desc_a >> 16) & 0x3FFF) << 4, ((desc_a >> 32) & 0x3FFF) << 4
    assert desc_a >> 62 == 0 and (desc_a >> 49) & 7 == 0
    m = torch.arange(64)[:, None]
    k = torch.arange(32)[None, :]
    addr = sa + (m % 8) * 16 + (m // 8) * ba + (k % 16) + (k // 16) * la
    assert int(addr.max()) < smem.numel()
    A = e4m3(smem[addr.reshape(-1)].reshape(-1, 1)).reshape(64, 32)
    a_regs = torch.zeros(4, 32, 4, 4, dtype=torch.uint8)
    a_regs[...] = smem[(sa + (A_M % 8) * 16 + (A_M // 8) * ba + (A_K % 16) + (A_K // 16) * la)]
    return wgmma(acc, a_regs, desc_b, smem, scale_d)


def attend_w1(data, q, nsplit, stages=4, sbov=SBO_V, w2=False, kv_view=None, patch=None, npar=0):
    """npar > 0: W3 (cuda_w3.py): W2's tiles per consumer warpgroup cw = h + 2 par, pair par takes global stages
    par, par + npar, ..., two copiers (keys 32c .. 32c + 31), P.V into O in place (O += P.V(i-1), then O *= alpha(i)),
    partial split index split * npar + par."""
    NPAR = max(npar, 1)
    w2 = w2 or npar > 0
    cfg = data["config"]
    assert cfg["q_len"] == L
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq == 8 * hkv and hkv % 2 == 0
    Bt = cfg["batch"]
    kcache = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vcache = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    if kv_view is not None:     # UW2: the copier reads the cache as it stands (the block's rows may be stale)
        kcache, vcache = kv_view
    table = data["page_table"]
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    qf = q.reshape(Bt * L, hq, 64).float()
    T = Bt * L
    o_part = torch.full((T * hq * nsplit * NPAR, 64), float("nan"))
    m_part = torch.full((T * hq * nsplit * NPAR,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    for b in range(Bt):
        n = int(data["cache_seqlens"][b])
        lo = max(n - L - window + 1, 0) if window > 0 else 0
        per = (n - lo + nsplit - 1) // nsplit
        per = (per + CHUNK - 1) // CHUNK * CHUNK
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            poff = kv0 * 64
            for split in range(nsplit):
                start = lo + split * per
                end = min(start + per, n)
                nchunks = (end - start + CHUNK - 1) // CHUNK if end > start else 0
                nst = (nchunks + 3) // 4
                slot = OFF_VT + 2 * 8 * sbov
                smem = torch.full((stages * slot + (4 * 4096 * NPAR if w2 else 0),), 0x7F, dtype=torch.uint8)   # NaN where not written
                qt0, pt0 = stages * slot, stages * slot + 2 * NPAR * 4096     # W2/W3: Q tiles, P tiles (per warpgroup cw: + 4096 cw)

                def copy(i):
                    base = (i % stages) * slot
                    for it in range(16):
                        key = 8 * (it >> 1) + (LANES & 7)
                        ch = 4 * (it & 1) + (LANES >> 3)
                        tok = torch.clamp(start + i * SKEYS + key, max=end - 1)
                        if npar:   # W3: copier c = it >> 3 owns keys 32c .. 32c + 31; row from its own entry (32c + (key & 31))
                            c_ = it >> 3
                            assert bool(((key >> 5) == c_).all())
                            kk = 32 * (c_ if "w3_copier" not in MUT else 1 - c_) + (key & 31)
                            tok = torch.clamp(start + i * SKEYS + kk, max=end - 1)
                        row = table[b, tok // page].long() * page + tok % page
                        for ln in range(32):
                            k_, c_, r_ = int(key[ln]), int(ch[ln]), int(row[ln])
                            if "k_layout" in MUT:
                                d = base + (c_ >> 2) * KC_HEAD + (k_ & 7) * SBO_K // 8 + (c_ & 3) * LBO_K + (k_ >> 3) * 16
                            else:
                                d = base + (c_ >> 2) * KC_HEAD + (k_ >> 3) * SBO_K + (c_ & 3) * LBO_K + (k_ & 7) * 16
                            smem[d:d + 16] = kcache[r_, poff + 16 * c_: poff + 16 * c_ + 16]
                            d = base + OFF_VS + k_ * VS_PITCH + 16 * c_
                            smem[d:d + 16] = vcache[r_, poff + 16 * c_: poff + 16 * c_ + 16]

                def transpose(i):
                    base = (i % stages) * slot
                    hh, qd = LANES >> 4, LANES & 15
                    for p in range(4):
                        w = []
                        for kp in range(16):
                            if "sigma" in MUT:
                                key = 16 * p + kp
                            else:
                                key = 16 * p + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1)
                            a = base + OFF_VS + key * VS_PITCH + 64 * hh + 4 * qd
                            w.append(smem[a[:, None] + torch.arange(4)])            # [32, 4]
                        t = [tr4(*w[4 * gq: 4 * gq + 4]) for gq in range(4)]     # t[gq][jd]: [32, 4]
                        for jd in range(4):
                            dn = 4 * qd + jd
                            a = base + OFF_VT + hh * 8 * sbov + (dn & 7) * 16 + (dn >> 3) * sbov + p * LBO_V
                            vals = torch.cat([t[gq][jd] for gq in range(4)], 1)       # [32, 16]
                            for ln in range(32):
                                smem[int(a[ln]): int(a[ln]) + 16] = vals[ln]

                states = []
                for cw in range(2 * NPAR):    # consumer warpgroup cw (head h = cw & 1): per warp w rows g (token 2w), g+8 (token 2w+1)
                    h = cw & 1
                    head = (kv0 + h) * 8 + G_OF
                    qa = torch.zeros(2, 4, 32, 4, 4, dtype=torch.uint8)          # [kstep, warp, lane, reg, byte]
                    for w in range(4):
                        qA, qB = qf[b * L + 2 * w, head], qf[b * L + 2 * w + 1, head]
                        for s in range(2):
                            o = 32 * s + 4 * T_OF
                            col = lambda qq, kk: qq[LANES, o + kk]  # noqa: E731
                            regs = [e4m3x4(col(qA, 0), col(qA, 1), col(qA, 2), col(qA, 3)),
                                    e4m3x4(col(qB, 0), col(qB, 1), col(qB, 2), col(qB, 3)),
                                    e4m3x4(col(qA, 16), col(qA, 17), col(qA, 18), col(qA, 19)),
                                    e4m3x4(col(qB, 16), col(qB, 17), col(qB, 18), col(qB, 19))]
                            if "q_perm" in MUT:          # the k-step's two 16-dim halves swapped
                                regs = [regs[2], regs[3], regs[0], regs[1]]
                            qa[s, w] = torch.stack(regs, 1)
                    states.append(dict(qa=qa, o=torch.zeros(4, 32, 32), m=[torch.full((4, 32), -1.0e30)] * 2,
                                       l=[torch.zeros(4, 32)] * 2, pend=None))
                    if w2:   # each thread stores its Q bytes (row 16w + g + 8*(r&1), dims 32s + 4*t4 + 16*(r>>1)) into the Q tile
                        sbo_q = 528 if "q_tile" in MUT else 512
                        for s_ in range(2):
                            for w in range(4):
                                for r in range(4):
                                    row = 16 * w + G_OF + 8 * (r & 1)
                                    k0 = 32 * s_ + 4 * T_OF + 16 * (r >> 1)
                                    for ln in range(32):
                                        rr, kk = int(row[ln]), int(k0[ln])
                                        a = qt0 + cw * 4096 + (rr & 7) * 16 + (rr >> 3) * sbo_q + (kk & 15) + (kk >> 4) * 128
                                        smem[a:a + 4] = qa[s_, w, ln, r]
                posA = n - L + 2 * torch.arange(4)[:, None]
                posB = posA + 1
                rec = DUMP is not None and b == 0 and pair == 0 and split == 0
                if rec:
                    dbg = DUMP.setdefault("dbg", torch.full((65536,), float("nan")))
                for i in range(nst):
                    copy(i)            # the slot's previous stage was consumed by both warpgroups (empty_bar)
                    if patch is not None:   # UW2: the transposers' fresh-row patch, before the transpose
                        patch(smem, (i % stages) * slot, b, kv0, start, end, n, i)
                    transpose(i)
                    if rec and i == 0:
                        tk = torch.clamp(start + torch.arange(64), max=end - 1)
                        dbg[58624:58688] = (table[b, tk // page].long() * page + tk % page).float()
                        for hh_ in range(2):
                            dbg[32768 + hh_ * 4096: 32768 + hh_ * 4096 + KC_HEAD] = smem[hh_ * KC_HEAD: (hh_ + 1) * KC_HEAD].float()
                            vt0 = OFF_VT + hh_ * 8 * sbov
                            dbg[40960 + hh_ * 4224: 40960 + hh_ * 4224 + 8 * sbov] = smem[vt0: vt0 + 8 * sbov].float()
                        dbg[49408: 49408 + 64 * VS_PITCH] = smem[OFF_VS: OFF_VS + 64 * VS_PITCH].float()
                    for h in range(2):
                        par_i = (i % NPAR) if "w3_parity" not in MUT else 0
                        cw = h + 2 * par_i
                        stt = states[cw]
                        base = (i % stages) * slot
                        kc_addr = base + h * KC_HEAD
                        vt_addr = base + OFF_VT + h * 8 * sbov
                        s0 = start + i * SKEYS
                        kstep = LBO_K if "kstep" in MUT else 2 * LBO_K
                        if w2:
                            qa_ = qt0 + cw * 4096
                            sc = wgmma_ss(torch.zeros(4, 32, 32), gdesc(qa_, 128, 512), gdesc(kc_addr, LBO_K, SBO_K), smem, 0)
                            sc = wgmma_ss(sc, gdesc(qa_ + 256, 128, 512), gdesc(kc_addr + kstep, LBO_K, SBO_K), smem, 1)
                        else:
                            sc = wgmma(torch.zeros(4, 32, 32), stt["qa"][0], gdesc(kc_addr, LBO_K, SBO_K), smem, 0)
                            sc = wgmma(sc, stt["qa"][1], gdesc(kc_addr + kstep, LBO_K, SBO_K), smem, 1)
                        if rec and i == 0:
                            dbg[h * 4096 + C_M.reshape(-1) * 64 + C_N.reshape(-1)] = sc.reshape(-1)
                            for s_ in range(2):
                                qa_ = stt["qa"][s_]
                                dbg[24576 + h * 4096 + A_M.reshape(-1) * 64 + 32 * s_ + A_K.reshape(-1)] = qa_.reshape(-1).float()
                        if s0 + SKEYS > end or s0 + SKEYS > n - L or (window > 0 and s0 < lo + L):
                            for j in range(8):
                                for c in range(2):
                                    key = s0 + 8 * j + 2 * T_OF[None, :] + c
                                    ok = key < end
                                    okA = ok & (key <= posA)
                                    okB = ok & (key <= posB)
                                    if window > 0:
                                        okA = okA & (key > posA - window)
                                        okB = okB & (key > posB - window)
                                    sc[:, :, 4 * j + c] = torch.where(okA, sc[:, :, 4 * j + c], float("-inf"))
                                    sc[:, :, 4 * j + 2 + c] = torch.where(okB, sc[:, :, 4 * j + 2 + c], float("-inf"))
                        rowA = [4 * j + c for j in range(8) for c in range(2)]
                        rowB = [4 * j + 2 + c for j in range(8) for c in range(2)]
                        mxA = sc[:, :, rowA].amax(-1).view(4, 8, 4).amax(-1).repeat_interleave(4, -1)
                        mxB = sc[:, :, rowB].amax(-1).view(4, 8, 4).amax(-1).repeat_interleave(4, -1)
                        m_run, l_run = stt["m"], stt["l"]
                        mnA, mnB = torch.maximum(m_run[0], mxA * qk_scale), torch.maximum(m_run[1], mxB * qk_scale)
                        alA, alB = torch.exp2(m_run[0] - mnA), torch.exp2(m_run[1] - mnB)
                        p = sc.clone()
                        p[:, :, rowA] = torch.exp2(sc[:, :, rowA] * qk_scale + (P_OFF - mnA)[:, :, None])
                        p[:, :, rowB] = torch.exp2(sc[:, :, rowB] * qk_scale + (P_OFF - mnB)[:, :, None])
                        stt["l"] = [l_run[0] * alA + p[:, :, rowA].sum(-1), l_run[1] * alB + p[:, :, rowB].sum(-1)]
                        if rec and i == 0:
                            dbg[8192 + h * 4096 + C_M.reshape(-1) * 64 + C_N.reshape(-1)] = p.reshape(-1)
                        stt["m"] = [mnA, mnB]
                        pa = torch.zeros(2, 4, 32, 4, 4, dtype=torch.uint8)
                        for c in range(2):
                            P = lambda idx: p[:, :, 16 * c + idx].reshape(-1)  # noqa: E731
                            regs = [e4m3x4(P(0), P(1), P(4), P(5)), e4m3x4(P(2), P(3), P(6), P(7)),
                                    e4m3x4(P(8), P(9), P(12), P(13)), e4m3x4(P(10), P(11), P(14), P(15))]
                            if "pack" in MUT:
                                regs = [regs[0], regs[2], regs[1], regs[3]]
                            pa[c] = torch.stack(regs, 1).view(4, 32, 4, 4)
                        sbo_v = (sbov - 16 if sbov == 528 else sbov + 16) if "sbo_v" in MUT else sbov
                        if w2:   # P bytes into the P tile: row 16w + g + 8*(r&1), k 32c + 4*t4 + 16*(r>>1)
                            pb = pt0 + cw * 4096
                            for c in range(2):
                                for w in range(4):
                                    for r in range(4):
                                        row = 16 * w + G_OF + 8 * (r & 1)
                                        k0 = 32 * c + 4 * T_OF + 16 * ((r >> 1) if "p_tile" not in MUT else 1 - (r >> 1))
                                        for ln in range(32):
                                            rr, kk = int(row[ln]), int(k0[ln])
                                            a = pb + tile_off(rr, kk)
                                            smem[a:a + 4] = pa[c, w, ln, r]
                            opv = wgmma_ss(torch.zeros(4, 32, 32), gdesc(pb, 128, 512), gdesc(vt_addr, LBO_V, sbo_v), smem, 0)
                            opv = wgmma_ss(opv, gdesc(pb + 256, 128, 512), gdesc(vt_addr + 2 * LBO_V, LBO_V, sbo_v), smem, 1)
                        else:
                            opv = wgmma(torch.zeros(4, 32, 32), pa[0], gdesc(vt_addr, LBO_V, sbo_v), smem, 0)
                            opv = wgmma(opv, pa[1], gdesc(vt_addr + 2 * LBO_V, LBO_V, sbo_v), smem, 1)
                        if rec and i == 0:
                            dbg[16384 + h * 4096 + C_M.reshape(-1) * 64 + C_N.reshape(-1)] = opv.reshape(-1)
                        al = torch.stack([alA, alA, alB, alB], -1).repeat(1, 1, 8)    # index 4j + 2*half + c
                        if npar:   # W3 order: P.V(i-1) accumulates into O, then O *= alpha(i); P.V(i) waits for the next stage
                            if "w3_rescale" in MUT:
                                stt["o"] = stt["o"] * al
                            if stt["pend"] is not None:
                                stt["o"] = stt["o"] + stt["pend"]
                            if "w3_rescale" not in MUT:
                                stt["o"] = stt["o"] * al
                            stt["pend"] = opv
                        else:
                            stt["o"] = stt["o"] * al + opv
                for cw in range(2 * NPAR):
                    h, par = cw & 1, cw >> 1
                    stt = states[cw]
                    if stt["pend"] is not None:   # W3: the last P.V
                        stt["o"] = stt["o"] + stt["pend"]
                    for w in range(4):
                        for half in range(2):
                            lsum = stt["l"][half][w].view(8, 4).sum(1).repeat_interleave(4)
                            tok = 2 * w + half
                            part = ((b * L + tok) * hq + (kv0 + h) * 8 + G_OF) * (nsplit * NPAR) + split * NPAR + (par if "w3_part" not in MUT else 0)
                            for j in range(8):
                                for c in range(2):
                                    dim = (2 * T_OF + 8 * j + c) if "out_dims" not in MUT else (8 * T_OF + 2 * j + c)
                                    o_part[part, dim] = stt["o"][w, :, 4 * j + 2 * half + c]
                            m_part[part] = stt["m"][half][w]
                            l_part[part] = lsum
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    assert not torch.isnan(o_part).any(), "an output partial was never written"
    return o_part, m_part, l_part


def run_w1(data, sub, nsplit, stages=4, sbov=SBO_V, w2=False, npar=0):
    q = prologue(data, sub)
    o, m, l_ = attend_w1(data, q, nsplit, stages, sbov, w2, npar=npar)
    ns = nsplit * max(npar, 1)
    return combine(data, o, m, l_, ns, 1 << max(0, (ns - 1).bit_length()), sub)


CASES = [
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1, 5]),
    (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [1, 4]),
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"), [3]),
    (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [3]),
    (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [2]),
    (dict(batch=1, ctx=1500, heads="dp", layer="full", q_len=8, dist="coherent"), [2]),
    (dict(batch=1, ctx=40, heads="tp8", layer="full", q_len=8, page=64), [1, 2]),
    (dict(batch=1, ctx=632, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1]),
]


def run_cases(quick, verbose=True, w2=False, npar=0):
    import reference as ref
    from emulate import window_rows_case
    sub = load_submission()
    cases = [CASES[0], CASES[1], CASES[6], CASES[7]] if quick else CASES
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            for stages, sbov in ((4, 528),) if quick else ((3, 528), (4, 528), (4, 512)):
                data = ref.generate_input(seed=840 + i, device="cpu", **case)
                out = run_w1(data, sub, nsplit, stages, sbov, w2, npar)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                if verbose:
                    print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} stages {stages} -> {msg}")
                bad += not ok
    for page in (1, 64) if not quick else (1,):
        data = window_rows_case(ref, page)
        out = run_w1(data, sub, 3, 4, SBO_V, w2, npar)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        if verbose:
            print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}) -> {msg}")
        bad += not ok
    return bad


if __name__ == "__main__":
    w2 = "--w2" in sys.argv
    npar = 2 if "--w3" in sys.argv else (1 if "--w3n2" in sys.argv else 0)
    for a in sys.argv:
        if a.startswith("--mut="):
            MUT.add(a[6:])
    bad = run_cases("--quick" in sys.argv, w2=w2, npar=npar)
    tag = f" (W3, {2 * max(npar, 1)} consumer warpgroups)" if npar else (" (W2: Q / P tiles, SS wgmma)" if w2 else "")
    print("emulate_cuda_w1" + tag + ":", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

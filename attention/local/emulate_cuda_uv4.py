"""Register- and shared-memory-level CPU emulation of attention/local/cuda_uv4.py (UV4: one launch per
sliding verify layer = V4's FP8 attention + in-kernel prologue + cluster push merge), CTA by CTA
with the kernel's index formulas:
- RoPE table entry e = (token e >> 5, frequency e & 31), sincosf of float(position) * inv_freq;
- query fold per consumer lane (warp w: head w / (4/MT), m-tiles; lane: rows g / g+8 = tokens
  2*mt / 2*mt+1, dims 16*t4 .. +15, RMS over the 4 lanes of the group, RoPE pair p = frequency
  8*t4 + p), E4M3 of the normalized query, V4's fragment registers, qk_scale_factor in qk_scale;
- the block's keys/values: warp w handles tokens w + j*WC; lane: head lane >> 4, dims
  4*(lane & 15) .. +3, RMS over the head's 16 lanes, RoPE frequency d0/2 + p, E4M3 bytes at
  new_k[128*l + 4*lane ..], written to the cache at cache_loc only by the CTA holding the key;
- the producer's ring copies read a cache in which the block's rows hold garbage (the kernel's
  copy may predate the write), so the consumer must take them from new_k / new_v; new_k /
  new_v rows a CTA did not compute hold garbage (incl. E4M3 NaN bytes);
- consumer loop: V4's (emulate_cuda_ver4.py's fragment formulas, PTX m16n8k32 e4m3 layouts) with
  the kernel's row select (key clamped to end-1; fresh && key >= n-8 -> new rows);
- second level (ncl > 1): slices cluster-major (split = cl*cs + rank); each owner's merged rows go
  to gpart[cl][R]; the request's rows are merged over the ncl clusters in cluster order, then finalized;
- push: row R = 16*token + 8*h + g to owner R // RPC, slot [rank][R % RPC] at dims 16*t4 + 8*jj + dt,
  m and l at 64 / 65; every slot written exactly once (checked); finalize per owner warp/lane
  (slot = warp + j*WC: merge, v_scale / l, RMSNorm, sigmoid(gate from gsm's cp.async pieces)).
Outputs and the new cache entries go through the judge's check_implementation.
MUT (mutation_test_cuda_uv4.py) plants single bugs: each must make some case fail.
Run: .venv/bin/python attention/local/emulate_cuda_uv4.py [--quick]
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate_cuda_dec import LANES, G_OF, T_OF  # noqa: E402
from emulate_cuda_ver4 import KP, VP, SKEYS, STAGE_BYTES, P_OFF, CHUNK, SUB, e4m3x4, tr4, mma_e4m3, to_e4m3_bytes  # noqa: E402

L, LBP, ROWS = 8, 68, 128
MUT = set()


def bf(x):
    return x.float()


def garbage(gen, n):
    """Random E4M3 bytes including NaN encodings (0x7f / 0xff)."""
    g = torch.randint(0, 256, (n,), generator=gen, dtype=torch.int64).to(torch.uint8)
    g[::7] = 0x7F
    return g


def uv4(data, cs, stages=3, mt=1, seed=0, ncl=1):
    cfg = data["config"]
    assert cfg["q_len"] == L
    B, hq, hkv, page, window = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq == 8 * hkv and hkv % 2 == 0
    rope = bool(cfg["rope_theta"])
    WC, WPH = 8 // mt, 4 // mt
    RPC = ROWS // cs
    RPW, LPW = RPC // WC, L // WC
    gen = torch.Generator().manual_seed(seed)
    qkv = data["qkv"].reshape(B, L, -1)
    D = 64
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)      # the kernel writes the block's rows here
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    table = data["page_table"]
    positions, cache_loc = data["positions"], data["cache_loc"]
    inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float) / 64)) if rope else torch.ones(32)
    k_scale, v_scale = float(data["k_scale"]), float(data["v_scale"])
    qk_factor = cfg["qk_scale_factor"]
    qk_scale = cfg["sm_scale"] * (1.0 if "qk_factor" in MUT else qk_factor) * k_scale * 1.4426950408889634
    kinv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(k_scale, dtype=torch.float32)
    vinv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(v_scale, dtype=torch.float32)
    # the producer's copies see garbage in the block's rows
    kc_seen, vc_seen = kc.clone(), vc.clone()
    for row in cache_loc.tolist():
        kc_seen[row] = garbage(gen, hkv * 64)
        vc_seen[row] = garbage(gen, hkv * 64)
    out = torch.full((B, L, hq * D), float("nan"))

    def finish(b, kv0, tok, hh, o, l_, gt):   # RMSNorm x sigmoid(gate), as step 9
        o = o * (v_scale / l_)
        rn = torch.rsqrt((o * o).sum() / 64 + 1e-5)
        out[b, tok, (kv0 * 8 + hh) * 64:(kv0 * 8 + hh) * 64 + 64] = o * rn / (1 + torch.exp(-gt))

    for b in range(B):
        n = int(data["cache_seqlens"][b])
        lo = max(n - L - window + 1, 0) if window > 0 else 0
        per = (n - lo + cs * ncl - 1) // (cs * ncl)    # slices cluster-major: split = cl * cs + rank
        per = (per + CHUNK - 1) // CHUNK * CHUNK
        # RoPE table (identical in every CTA): entry e = token e >> 5, frequency e & 31
        rc, rs = torch.zeros(L, 32), torch.zeros(L, 32)
        if rope:
            for e in range(L * 32):
                ang = torch.tensor(float(positions[b * L + (e >> 5)]), dtype=torch.float32) * inv_freq[e & 31]
                rs[e >> 5, e & 31] = math.sin(float(ang))
                rc[e >> 5, e & 31] = math.cos(float(ang))
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            poff = kv0 * 64
            gst, gfin = {}, {}
            for cl in range(ncl):
                rbufs = [torch.full((cs * RPC * LBP,), float("nan")) for _ in range(cs)]
                writes = [torch.zeros(cs * RPC * LBP, dtype=torch.int64) for _ in range(cs)]
                gsms = []
                for rank in range(cs):
                    split = cl * cs + rank
                    start = lo + split * per
                    end = min(start + per, n)
                    nchunks = (end - start + CHUNK - 1) // CHUNK if end > start else 0
                    nsub = (end - start + SUB - 1) // SUB if end > start else 0
                    # gate pieces (cp.async): c -> slot c >> 3, R = rank*RPC + slot, 16 B piece c & 7
                    gsm = torch.full((RPC * 64,), float("nan"))
                    for c in range(RPC * 8):
                        slot, R = c >> 3, rank * RPC + (c >> 3)
                        piece = ((c >> 3) & 7) if "gate_piece" in MUT else (c & 7)
                        base = (hq + 2 * hkv) * 64 + (kv0 * 8 + (R & 15)) * 64 + 8 * piece
                        gsm[slot * 64 + 8 * (c & 7): slot * 64 + 8 * (c & 7) + 8] = bf(qkv[b, R >> 4, base:base + 8])
                    gsms.append(gsm)
                    # new rows this CTA holds (garbage elsewhere)
                    new_k = garbage(gen, L * 128)
                    new_v = garbage(gen, L * 128)
                    for w in range(WC):
                        for j in range(LPW):
                            l_ = w + j * WC
                            key = n - L + l_
                            if not (start <= key < end):
                                continue
                            hk, d0 = LANES >> 4, 4 * (LANES & 15)
                            kidx = hq * 64 + (kv0 + hk)[:, None] * 64 + d0[:, None] + torch.arange(4)
                            k4 = bf(qkv[b, l_][kidx])                     # [32 lanes, 4]
                            v4 = bf(qkv[b, l_][kidx + hkv * 64])
                            ss = (k4 * k4).sum(1)
                            red = (16, 8) if "newrow_norm" not in MUT else (8, 4)
                            ss = ss.view(-1, red[0]).sum(1).repeat_interleave(red[0])   # lanes of one head
                            k4 = k4 * torch.rsqrt(ss / 64 + 1e-5)[:, None]
                            if rope:
                                for p in range(2):
                                    f = (d0 + p) if "newrow_rope" in MUT else (d0 // 2 + p)
                                    f = f.clamp(max=31)
                                    sn, cs_ = rs[l_][f], rc[l_][f]
                                    e_, o_ = k4[:, 2 * p].clone(), k4[:, 2 * p + 1].clone()
                                    k4[:, 2 * p], k4[:, 2 * p + 1] = e_ * cs_ - o_ * sn, e_ * sn + o_ * cs_
                            kb = to_e4m3_bytes(k4 * kinv)                 # [32, 4]: byte 4*lane + i
                            vb = to_e4m3_bytes(v4 * vinv)
                            new_k[128 * l_: 128 * l_ + 128] = kb.reshape(-1)
                            new_v[128 * l_: 128 * l_ + 128] = vb.reshape(-1)
                            crow = int(cache_loc[b * L + l_])
                            kc[crow, poff:poff + 128] = kb.reshape(-1)
                            vc[crow, poff:poff + 128] = vb.reshape(-1)
                    ring = torch.zeros(stages, STAGE_BYTES, dtype=torch.uint8)

                    def produce(i):          # V4's producer: tokens clamped to end - 1, from the cache as seen
                        st = i % stages
                        s0 = torch.clamp(start + i * SKEYS + LANES, max=end - 1)
                        s1 = torch.clamp(start + i * SKEYS + 32 + LANES, max=end - 1)
                        r0 = table[b, s0 // page].long() * page + s0 % page
                        r1 = table[b, s1 // page].long() * page + s1 % page
                        for qq in range(16):
                            for ln in range(32):
                                tq, ch = qq * 4 + (ln >> 3), ln & 7
                                row = int((r0 if qq < 8 else r1)[tq & 31])
                                o = st * STAGE_BYTES + tq * KP + 16 * ch
                                ring.view(-1)[o:o + 16] = kc_seen[row, poff + 16 * ch: poff + 16 * ch + 16]
                                o = st * STAGE_BYTES + SKEYS * KP + tq * VP + 16 * ch
                                ring.view(-1)[o:o + 16] = vc_seen[row, poff + 16 * ch: poff + 16 * ch + 16]

                    warps = []
                    for w in range(WC):
                        h, mt0 = w // WPH, (w % WPH) * mt
                        qa = []
                        for mi in range(mt):
                            xs = []
                            for r in range(2):
                                tok = 2 * (mt0 + mi) + r
                                head = (kv0 + h) * 8 + G_OF
                                x = bf(qkv[b, tok][head[:, None] * 64 + 16 * T_OF[:, None] + torch.arange(16)])   # [32, 16]
                                ss = (x * x).sum(1)
                                grp = 2 if "fold_norm" in MUT else 4
                                ss = ss.view(-1, grp).sum(1).repeat_interleave(grp)
                                x = x * torch.rsqrt(ss / 64 + 1e-5)[:, None]
                                if rope:
                                    for p in range(8):
                                        f = (16 * (T_OF & 1) + p) if "fold_freq" in MUT else (8 * T_OF + p)
                                        sn, cs_ = rs[tok][f], rc[tok][f]
                                        e_, o_ = x[:, 2 * p].clone(), x[:, 2 * p + 1].clone()
                                        x[:, 2 * p], x[:, 2 * p + 1] = e_ * cs_ - o_ * sn, e_ * sn + o_ * cs_
                                xs.append(x)
                            steps = []
                            for s in range(2):
                                o = 8 * s
                                steps.append(torch.stack([e4m3x4(xs[0][:, o], xs[0][:, o + 1], xs[0][:, o + 2], xs[0][:, o + 3]),
                                                          e4m3x4(xs[1][:, o], xs[1][:, o + 1], xs[1][:, o + 2], xs[1][:, o + 3]),
                                                          e4m3x4(xs[0][:, o + 4], xs[0][:, o + 5], xs[0][:, o + 6], xs[0][:, o + 7]),
                                                          e4m3x4(xs[1][:, o + 4], xs[1][:, o + 5], xs[1][:, o + 6], xs[1][:, o + 7])], 1))
                            qa.append(steps)
                        warps.append(dict(h=h, mt0=mt0, qa=qa, m=[[torch.full((32,), -1.0e30)] * 2 for _ in range(mt)],
                                          l=[[torch.zeros(32)] * 2 for _ in range(mt)],
                                          acc=[torch.zeros(8, 32, 4) for _ in range(mt)]))
                    for c in range(nsub):
                        i, scx = c // 2, c % 2
                        if scx == 0:
                            produce(i)
                        st = i % stages
                        s0 = start + c * SUB
                        sk = ring[st][:SKEYS * KP]
                        sv = ring[st][SKEYS * KP:]
                        masked = s0 + SUB > end or s0 + SUB > n - L or window > 0
                        fresh = s0 + SUB > n - L

                        def eff(key):            # rows past the slice end are copies of row end-1
                            return key if "no_clamp" in MUT else key.clamp(max=end - 1)

                        def sel(key):            # per-lane: take the in-kernel row? (key already clamped)
                            if "no_sel" in MUT:
                                return torch.zeros_like(key, dtype=torch.bool)
                            if "no_clamp" in MUT:
                                return fresh & (key >= n - L) & (key < end)
                            return fresh & (key >= n - L)

                        for wd in warps:
                            h, mt0 = wd["h"], wd["mt0"]
                            rows = []
                            for add in (0, 16):
                                for vk in (2 * T_OF, 2 * T_OF + 1, 2 * T_OF + 8, 2 * T_OF + 9):
                                    r = vk + add
                                    key = eff(s0 + r)
                                    ring_v = sv[((32 * scx + r) * VP + 64 * h + 8 * G_OF)[:, None] + torch.arange(8)]
                                    nidx = ((key - (n - L)).clamp(0, L - 1) * 128 + 64 * h + 8 * G_OF)[:, None] + torch.arange(8)
                                    rows.append(torch.where(sel(key)[:, None], new_v[nidx], ring_v))
                            vb0 = tr4(*[r[:, 0:4] for r in rows[0:4]]) + tr4(*[r[:, 4:8] for r in rows[0:4]])
                            vb1 = tr4(*[r[:, 0:4] for r in rows[4:8]]) + tr4(*[r[:, 4:8] for r in rows[4:8]])
                            for mi in range(mt):
                                sc = []
                                for nt in range(4):
                                    kr = 8 * nt + G_OF
                                    key = eff(s0 + kr)
                                    ring_k = sk[((32 * scx + kr) * KP + 64 * h + 16 * T_OF)[:, None] + torch.arange(16)]
                                    nidx = ((key - (n - L)).clamp(0, L - 1) * 128 + 64 * h + 16 * T_OF)[:, None] + torch.arange(16)
                                    kb = torch.where(sel(key)[:, None], new_k[nidx], ring_k)
                                    w4 = [kb[:, 4 * k:4 * k + 4] for k in range(4)]
                                    cfr = mma_e4m3(torch.zeros(32, 4), wd["qa"][mi][0], torch.stack([w4[0], w4[1]], 1))
                                    cfr = mma_e4m3(cfr, wd["qa"][mi][1], torch.stack([w4[2], w4[3]], 1))
                                    sc.append(cfr)
                                posA = n - L + 2 * (mt0 + mi)
                                posB = posA + 1
                                x = torch.zeros(4, 32, 4)
                                for nt in range(4):
                                    for jj in range(2):
                                        key = s0 + nt * 8 + 2 * T_OF + jj
                                        xa, xb = sc[nt][:, jj] * qk_scale, sc[nt][:, 2 + jj] * qk_scale
                                        if masked:
                                            ok = key < end
                                            okA = ok & (key <= posA)
                                            okB = ok & (key <= posB)
                                            if window > 0:
                                                okA = okA & (key > posA - window)
                                                okB = okB & (key > posB - window)
                                            xa = torch.where(okA, xa, float("-inf"))
                                            xb = torch.where(okB, xb, float("-inf"))
                                        x[nt][:, jj], x[nt][:, 2 + jj] = xa, xb
                                mxA = x[:, :, 0:2].amax(dim=(0, 2)).view(8, 4).max(1).values.repeat_interleave(4)
                                mxB = x[:, :, 2:4].amax(dim=(0, 2)).view(8, 4).max(1).values.repeat_interleave(4)
                                mA, mB = wd["m"][mi]
                                mnA, mnB = torch.maximum(mA, mxA), torch.maximum(mB, mxB)
                                alA, alB = torch.exp2(mA - mnA), torch.exp2(mB - mnB)
                                p = torch.zeros(4, 32, 4)
                                for nt in range(4):
                                    p[nt][:, 0] = torch.exp2(x[nt][:, 0] + (P_OFF - mnA))
                                    p[nt][:, 1] = torch.exp2(x[nt][:, 1] + (P_OFF - mnA))
                                    p[nt][:, 2] = torch.exp2(x[nt][:, 2] + (P_OFF - mnB))
                                    p[nt][:, 3] = torch.exp2(x[nt][:, 3] + (P_OFF - mnB))
                                lA, lB = wd["l"][mi]
                                wd["l"][mi] = [lA * alA + p[:, :, 0:2].sum(dim=(0, 2)), lB * alB + p[:, :, 2:4].sum(dim=(0, 2))]
                                wd["m"][mi] = [mnA, mnB]
                                pa = torch.stack([e4m3x4(p[0][:, 0], p[0][:, 1], p[1][:, 0], p[1][:, 1]),
                                                  e4m3x4(p[0][:, 2], p[0][:, 3], p[1][:, 2], p[1][:, 3]),
                                                  e4m3x4(p[2][:, 0], p[2][:, 1], p[3][:, 0], p[3][:, 1]),
                                                  e4m3x4(p[2][:, 2], p[2][:, 3], p[3][:, 2], p[3][:, 3])], 1)
                                acc = wd["acc"][mi]
                                for dt in range(8):
                                    pv = mma_e4m3(torch.zeros(32, 4), pa, torch.stack([vb0[dt], vb1[dt]], 1))
                                    acc[dt][:, 0] = acc[dt][:, 0] * alA + pv[:, 0]
                                    acc[dt][:, 1] = acc[dt][:, 1] * alA + pv[:, 1]
                                    acc[dt][:, 2] = acc[dt][:, 2] * alB + pv[:, 2]
                                    acc[dt][:, 3] = acc[dt][:, 3] * alB + pv[:, 3]
                    # push: row R = 16*token + 8*h + g -> owner R // RPC, slot [rank][R % RPC]
                    for wd in warps:
                        h, mt0 = wd["h"], wd["mt0"]
                        for mi in range(mt):
                            for r in range(2):
                                lsum = wd["l"][mi][r] if "lsum" in MUT else wd["l"][mi][r].view(8, 4).sum(1).repeat_interleave(4)
                                R = (2 * (mt0 + mi) + r) * 16 + h * 8 + G_OF
                                owner = (R % cs) if "push_owner" in MUT else (R // RPC)
                                slot = R % RPC
                                for ln in range(32):
                                    own, base = int(owner[ln]), (rank * RPC + int(slot[ln])) * LBP
                                    rb, wr = rbufs[own], writes[own]
                                    for jj in range(2):
                                        for dt in range(8):
                                            a = base + 16 * int(T_OF[ln]) + 8 * jj + dt
                                            rb[a] = wd["acc"][mi][dt][ln, 2 * r + jj]
                                            wr[a] += 1
                                    if int(T_OF[ln]) == 0:
                                        sm, sl = (65, 64) if "ml_swap" in MUT else (64, 65)
                                        rb[base + sm] = wd["m"][mi][r][ln]
                                        rb[base + sl] = lsum[ln]
                                        wr[base + 64] += 1
                                        wr[base + 65] += 1
                # every slot (src, row, 0..65) written exactly once
                for own in range(cs):
                    w = writes[own].view(cs * RPC, LBP)[:, :66]
                    assert bool((w == 1).all()), f"receive buffer of CTA {own}: slots written {w.min()}..{w.max()} times"
                # finalize: owner CTA, consumer warp w, rows slot = w + j*WC; lane: dims lane, 32 + lane
                for own in range(cs):
                    rb, gsm = rbufs[own].view(cs * RPC, LBP), gsms[own]
                    for w in range(WC):
                        for j in range(RPW):
                            slot = w + j * WC
                            R = own * RPC + slot
                            tok, hh = R >> 4, R & 15
                            src = torch.arange(cs) * RPC + slot
                            m_ = rb[src, 64]
                            mm = max(-1.0e30, float(m_.max()))
                            wt = torch.exp2(m_ - mm)
                            l_ = float((wt * rb[src, 65]).sum())
                            o = (wt[:, None] * rb[src, :64]).sum(0)
                            if ncl > 1:                 # to gpart[cl][R]; the second level below
                                gst[(cl, R)] = (o, mm, l_)
                                gfin[R] = gsm[slot * 64: slot * 64 + 64]
                                continue
                            finish(b, kv0, tok, hh, o, l_, gsm[slot * 64: slot * 64 + 64])
            if ncl > 1:   # the last cluster's owner of rank r merges gpart[0..ncl-1][R] for its rows
                for R in range(ROWS):
                    rr = (R ^ 1) if "l2_row" in MUT else R
                    cls_ = [c for c in range(ncl) if not ("l2_skip_last" in MUT and c == ncl - 1)]
                    m_ = torch.tensor([gst[(c, rr)][1] for c in cls_])
                    mm = max(-1.0e30, float(m_.max()))
                    wt = torch.exp2(m_ - mm)
                    l_ = float(sum(float(wt[k]) * gst[(c, rr)][2] for k, c in enumerate(cls_)))
                    o = sum(wt[k] * gst[(c, rr)][0] for k, c in enumerate(cls_))
                    finish(b, kv0, R >> 4, R & 15, o, l_, gfin[R])
    assert not torch.isnan(out).any(), "an output element was never written"
    return out.to(torch.bfloat16)


CASES = [
    (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [(8, 3, 1), (4, 3, 2)]),
    (dict(batch=1, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1, dist="coherent"), [(16, 3, 1), (16, 3, 2)]),
    (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [(8, 4, 1), (4, 4, 1)]),
    (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [(8, 3, 2)]),
    (dict(batch=1, ctx=1500, heads="dp", layer="sliding", q_len=8, dist="coherent"), [(4, 3, 1)]),
    (dict(batch=2, ctx=300, heads="tp8", layer="full", q_len=8, page=1), [(4, 3, 1), (16, 3, 2)]),
    (dict(batch=1, ctx=40, heads="tp8", layer="sliding", q_len=8, page=64), [(8, 3, 1), (16, 3, 1)]),
    # n = 197, cs 4: slices [128, 192) and [192, 197) split the block's keys 189 .. 196 over two CTAs
    (dict(batch=1, ctx=189, heads="tp8", layer="sliding", q_len=8, page=1), [(4, 3, 1), (4, 3, 2)]),
    (dict(batch=3, ctx=2100, heads="tp8", layer="sliding", q_len=8, page=1, spread="agentic"), [(8, 3, 1)]),
    # second level: (cs, stages, mt, ncl); n = 308 with 32 slices of 16 leaves cluster 3 empty
    (dict(batch=1, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [(8, 3, 1, 4), (4, 3, 2, 3)]),
    (dict(batch=1, ctx=300, heads="tp8", layer="full", q_len=8, page=1), [(8, 3, 1, 4)]),
    (dict(batch=2, ctx=2300, heads="tp8", layer="sliding", q_len=8, page=1, dist="edges"), [(4, 3, 1, 2)]),
]


def run_cases(quick, verbose=True):
    import reference as ref
    from emulate import window_rows_case
    cases = [CASES[0], CASES[1], CASES[5], CASES[6], CASES[7], CASES[9], CASES[10]] if quick else CASES
    bad = 0
    for i, (case, variants) in enumerate(cases):
        for cs, stages, mt, *rest in variants[:1] if quick else variants:
            ncl = rest[0] if rest else 1
            data = ref.generate_input(seed=760 + i, device="cpu", **case)
            try:
                out = uv4(data, cs, stages, mt, seed=i, ncl=ncl)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
            except AssertionError as exc:
                ok, msg = False, f"assertion: {exc}"
            if verbose:
                print(("PASS " if ok else "FAIL ") + f"{case} cs {cs} stages {stages} mt {mt} ncl {ncl} -> {msg}")
            bad += not ok
    for page in (1, 64):
        data = window_rows_case(ref, page)
        for cs in (4, 16) if not quick else (8,):
            try:
                out = uv4(data, cs, 3, 1, seed=page)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
            except AssertionError as exc:
                ok, msg = False, f"assertion: {exc}"
            if verbose:
                print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}, cs {cs}) -> {msg}")
            bad += not ok
    return bad


if __name__ == "__main__":
    bad = run_cases("--quick" in sys.argv)
    print("emulate_cuda_uv4:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

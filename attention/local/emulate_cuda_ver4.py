"""Register- and shared-memory-level CPU emulation of attention/local/cuda_ver4.py (V4: S2c's producer,
FP8 consumers). The ring is byte arrays with the kernel's pitches and the producer's copy mapping
(as emulate_cuda_ver2.py); per consumer warp (head w / (4/MT), m-tiles (w % (4/MT))*MT + mi) the
emulator reads the kernel's bytes (Q rounded to E4M3 in the kernel's register order, K words
16*t4 .. +15 of each key row, V 8-byte reads at 8*g of keys 2t4, 2t4+1, 8+2t4, 9+2t4 (+16) and
the 4x4 byte transposes with the kernel's prmt selectors), packs P exactly as the kernel does
(E4M3 of exp2(x - m + 8), byte order by cvt.rn.satfinite.e4m3x2's packing) and places every
fragment element by the PTX m16n8k32 .e4m3 layout formulas (PTX ISA 9.4 9.7.16.5.10), not by
the kernel's index reasoning. Partials in the kernel's layout, merged by emulate.py's combine;
outputs go through the judge's check_implementation.
MUT (mutation_test_cuda_ver4.py) plants single bugs: each must make some case fail.
Run: .venv/bin/python attention/local/emulate_cuda_ver4.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF, byte_perm, e4m3  # noqa: E402

CHUNK, SUB, L, KP, VP = 16, 32, 8, 192, 144
SKEYS = 64
STAGE_BYTES = SKEYS * (KP + VP)
P_OFF = 8.0
MUT = set()


def to_e4m3_bytes(x):
    """cvt.rn.satfinite.e4m3 of float values -> uint8 bytes (no value here exceeds 448)."""
    assert float(x.abs().max()) <= 448.0
    return x.float().to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3x4(x0, x1, x2, x3):
    """e4m3x4 in the kernel: lo = cvt(a=x1, b=x0) -> bits 15:8 = x1, 7:0 = x0; hi likewise; d = {lo, hi}."""
    return torch.stack([to_e4m3_bytes(x0), to_e4m3_bytes(x1), to_e4m3_bytes(x2), to_e4m3_bytes(x3)], 1)


def tr4(r0, r1, r2, r3):
    """The kernel's prmt sequence on [lanes, 4] byte tensors; returns o[0..3]."""
    x0, x1 = byte_perm(r0, r1, 0x5140), byte_perm(r0, r1, 0x7362)
    y0, y1 = byte_perm(r2, r3, 0x5140), byte_perm(r2, r3, 0x7362)
    if "tr_sel" in MUT:
        return [byte_perm(x0, y0, 0x5410), byte_perm(x1, y1, 0x5410), byte_perm(x0, y0, 0x7632), byte_perm(x1, y1, 0x7632)]
    return [byte_perm(x0, y0, 0x5410), byte_perm(x0, y0, 0x7632), byte_perm(x1, y1, 0x5410), byte_perm(x1, y1, 0x7632)]


def mma_e4m3(c, a_regs, b_regs):
    """mma.m16n8k32.row.col.f32.e4m3.e4m3.f32 for one warp. a_regs [32, 4 regs, 4 bytes], b_regs
    [32, 2 regs, 4 bytes] (uint8), c [32, 4]; elements placed by the PTX formulas: A element
    i (byte i & 3 of register i >> 2): row g for i in [0,4) u [8,12) else g + 8, col 4*t4 + (i & 3)
    (+16 for i >= 8); B element i: row 4*t4 + (i & 3) (+16 for i >= 4), col g; C: c0,c1 row g,
    c2,c3 row g + 8, col 2*t4 + (i & 1)."""
    A = torch.zeros(16, 32)
    B = torch.zeros(32, 8)
    av = e4m3(a_regs.reshape(32, 16).contiguous())
    bv = e4m3(b_regs.reshape(32, 8).contiguous())
    for i in range(16):
        row = G_OF if (i < 4 or 8 <= i < 12) else G_OF + 8
        col = 4 * T_OF + (i & 3) + (16 if i >= 8 else 0)
        A[row, col] = av[:, i]
    for i in range(8):
        row = 4 * T_OF + (i & 3) + (16 if i >= 4 else 0)
        B[row, G_OF] = bv[:, i]
    D = A @ B
    out = c.clone()
    for i in range(4):
        row = G_OF if i < 2 else G_OF + 8
        out[:, i] += D[row, 2 * T_OF + (i & 1)]
    return out


def attend_ver4(data, q, nsplit, stages=3, mt=2):
    cfg = data["config"]
    assert cfg["q_len"] == L
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq // hkv == 8 and hkv % 2 == 0
    B = cfg["batch"]
    WC, WPH = 8 // mt, 4 // mt
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    table = data["page_table"]
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    qf = q.reshape(B * L, hq, 64).float()                      # BF16 values from the prologue
    T = B * L
    o_part = torch.full((T * hq * nsplit, 64), float("nan"))
    m_part = torch.full((T * hq * nsplit,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    for b in range(B):
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
                nsub = (end - start + SUB - 1) // SUB if end > start else 0
                nst = (nchunks + 3) // 4
                ring = torch.zeros(stages, STAGE_BYTES, dtype=torch.uint8)

                def produce(i):          # S2c's producer: lanes' tokens lane / 32 + lane, rows clamped to end - 1
                    st = i % stages
                    lanes = torch.arange(32)
                    s0 = torch.clamp(start + i * SKEYS + lanes, max=end - 1)
                    s1 = torch.clamp(start + i * SKEYS + 32 + lanes, max=end - 1)
                    r0 = table[b, s0 // page].long() * page + s0 % page
                    r1 = table[b, s1 // page].long() * page + s1 % page
                    for qq in range(16):
                        for ln in range(32):
                            tq, ch = qq * 4 + (ln >> 3), ln & 7
                            row = int((r0 if qq < 8 else r1)[tq & 31])
                            o = st * STAGE_BYTES + tq * KP + 16 * ch
                            ring.view(-1)[o:o + 16] = kc[row, poff + 16 * ch: poff + 16 * ch + 16]
                            o = st * STAGE_BYTES + SKEYS * KP + tq * VP + 16 * ch
                            ring.view(-1)[o:o + 16] = vc[row, poff + 16 * ch: poff + 16 * ch + 16]

                warps = []
                for w in range(WC):
                    h, mt0 = w // WPH, (w % WPH) * mt
                    qa = []
                    for mi in range(mt):
                        ta = 2 * (mt0 + mi)
                        head = (kv0 + h) * 8 + G_OF
                        qA = qf[b * L + ta, head]               # [32 lanes, 64]
                        qB = qf[b * L + ta + 1, head]
                        steps = []
                        for s in range(2):
                            o = 16 * T_OF + 8 * s
                            col = lambda qq, k: qq[LANES, o + k]  # noqa: E731
                            regs = [e4m3x4(col(qA, 0), col(qA, 1), col(qA, 2), col(qA, 3)),
                                    e4m3x4(col(qB, 0), col(qB, 1), col(qB, 2), col(qB, 3)),
                                    e4m3x4(col(qA, 4), col(qA, 5), col(qA, 6), col(qA, 7)),
                                    e4m3x4(col(qB, 4), col(qB, 5), col(qB, 6), col(qB, 7))]
                            if "qa_order" in MUT:
                                regs = [regs[0], regs[2], regs[1], regs[3]]
                            steps.append(torch.stack(regs, 1))  # [32, 4 regs, 4 bytes]
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
                    for wd in warps:
                        h, mt0 = wd["h"], wd["mt0"]
                        vkeys = [2 * T_OF, 2 * T_OF + 1, 2 * T_OF + 8, 2 * T_OF + 9]
                        if "v_keys" in MUT:
                            vkeys = [2 * T_OF, 2 * T_OF + 8, 2 * T_OF + 1, 2 * T_OF + 9]
                        rows = [sv[((32 * scx + vk + add) * VP + 64 * h + 8 * G_OF)[:, None] + torch.arange(8)]
                                for add in (0, 16) for vk in vkeys]
                        vb0 = tr4(*[r[:, 0:4] for r in rows[0:4]]) + tr4(*[r[:, 4:8] for r in rows[0:4]])
                        vb1 = tr4(*[r[:, 0:4] for r in rows[4:8]]) + tr4(*[r[:, 4:8] for r in rows[4:8]])
                        for mi in range(mt):
                            sc = []
                            for nt in range(4):
                                kb = sk[((32 * scx + 8 * nt + G_OF) * KP + 64 * h + 16 * T_OF)[:, None] + torch.arange(16)]
                                w4 = [kb[:, 4 * k:4 * k + 4] for k in range(4)]
                                if "k_words" in MUT:
                                    w4 = [w4[0], w4[2], w4[1], w4[3]]
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
                            mxA = x[:, :, 0:2].amax(dim=(0, 2))
                            mxB = x[:, :, 2:4].amax(dim=(0, 2))
                            mxA = mxA.view(8, 4).max(1).values.repeat_interleave(4)
                            mxB = mxB.view(8, 4).max(1).values.repeat_interleave(4)
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
                            pa = [e4m3x4(p[0][:, 0], p[0][:, 1], p[1][:, 0], p[1][:, 1]),
                                  e4m3x4(p[0][:, 2], p[0][:, 3], p[1][:, 2], p[1][:, 3]),
                                  e4m3x4(p[2][:, 0], p[2][:, 1], p[3][:, 0], p[3][:, 1]),
                                  e4m3x4(p[2][:, 2], p[2][:, 3], p[3][:, 2], p[3][:, 3])]
                            if "p_pack" in MUT:
                                pa = [e4m3x4(p[0][:, 0], p[1][:, 0], p[0][:, 1], p[1][:, 1])] + pa[1:]
                            pa = torch.stack(pa, 1)
                            acc = wd["acc"][mi]
                            for dt in range(8):
                                pv = mma_e4m3(torch.zeros(32, 4), pa, torch.stack([vb0[dt], vb1[dt]], 1))
                                acc[dt][:, 0] = acc[dt][:, 0] * alA + pv[:, 0]
                                acc[dt][:, 1] = acc[dt][:, 1] * alA + pv[:, 1]
                                acc[dt][:, 2] = acc[dt][:, 2] * alB + pv[:, 2]
                                acc[dt][:, 3] = acc[dt][:, 3] * (alA if "rescale_row" in MUT else alB) + pv[:, 3]
                for wd in warps:
                    h, mt0 = wd["h"], wd["mt0"]
                    for mi in range(mt):
                        for r in range(2):
                            lsum = wd["l"][mi][r].view(8, 4).sum(1).repeat_interleave(4)
                            tok = 2 * (mt0 + mi) + r
                            part = ((b * L + tok) * hq + (kv0 + h) * 8 + G_OF) * nsplit + split
                            for dt in range(8):
                                for jj in range(2):
                                    dim = 8 * (2 * T_OF + jj) + dt
                                    if "out_dim" in MUT:
                                        dim = 16 * T_OF + 2 * dt + jj
                                    o_part[part, dim] = wd["acc"][mi][dt][:, 2 * r + jj]
                            m_part[part] = wd["m"][mi][r]
                            l_part[part] = lsum
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    assert not torch.isnan(o_part).any(), "an output partial was never written"
    return o_part, m_part, l_part


def run_ver4(data, sub, nsplit, stages=3, mt=2):
    q = prologue(data, sub)
    o, m, l_ = attend_ver4(data, q, nsplit, stages, mt)
    return combine(data, o, m, l_, nsplit, 1 << max(0, (nsplit - 1).bit_length()), sub)


CASES = [
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1, 5]),
    (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [1, 4]),
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"), [3]),
    (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [3]),
    (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [2]),
    (dict(batch=1, ctx=1500, heads="dp", layer="full", q_len=8, dist="coherent"), [2]),
    (dict(batch=1, ctx=40, heads="tp8", layer="full", q_len=8, page=64), [1, 2]),
]


def run_cases(quick, verbose=True):
    import reference as ref
    from emulate import window_rows_case
    sub = load_submission()
    cases = [CASES[0], CASES[1], CASES[3], CASES[5]] if quick else CASES
    bad, worst = 0, 0.0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            for stages, mt in ((3, 2), (4, 1)) if quick else ((3, 2), (4, 2), (3, 1), (4, 1)):
                data = ref.generate_input(seed=740 + i, device="cpu", **case)
                out = run_ver4(data, sub, nsplit, stages, mt)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                if verbose:
                    print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} stages {stages} mt {mt} -> {msg}")
                bad += not ok
    for page in (1, 64):
        data = window_rows_case(ref, page)
        for mt in (2, 1):
            out = run_ver4(data, sub, 3, 3, mt)
            ok, msg = ref.check_implementation(data, ref.collect(data, out))
            if verbose:
                print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}, mt {mt}) -> {msg}")
            bad += not ok
    return bad


if __name__ == "__main__":
    bad = run_cases("--quick" in sys.argv)
    print("emulate_cuda_ver4:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

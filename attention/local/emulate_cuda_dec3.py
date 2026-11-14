"""Register- and shared-memory-level CPU emulation of attention/local/cuda_dec3.py (D2): A5b's compute
(emulate_cuda_dec2.py's, lane by lane) fed from the per-warp cp.async ring: the copy mapping (lane ->
token 4*q + (lane>>3), chunk lane&7), pitch 144, K tile then V tile, stage indices, issue order and
the row prefetch, with fragments read from the emulated ring at the kernel's offsets.
Original A5b notes: lane by
lane (vectorized over the 32 lanes): the kernel's own loads (two 16 B K loads per token at bytes
32*t4 and 32*t4+16 of the pair's 128-byte piece; 16 B V loads at 16*g), the stacked
block-diagonal query fragments, __byte_perm selectors and cvt halves, fragments placed by the PTX
ISA 9.4 m16n8k16 layout formulas, FP16 rounding, the warps' round-robin chunks, per-head online
softmax on c0,c1 / c2,c3, and the shared-memory merge. Partials go through the emulated _combine
and the judge's check_implementation.

Run: .venv/bin/python attention/local/emulate_cuda_dec2.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF, byte_perm, e4m3, f16, mma, tail_key_case  # noqa: E402

CHUNK = 16
PITCH = 144
STAGE_BYTES = 2 * CHUNK * PITCH
HL = T_OF >> 1                  # head whose columns a lane carries


def attend_dec3(data, q, nsplit, warps=4, stages=3):
    cfg = data["config"]
    assert cfg["q_len"] == 1
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq // hkv == 8 and hkv % 2 == 0
    B = cfg["batch"]
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    table = data["page_table"]
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    qf = q.reshape(B, hq, 64).float()
    o_part = torch.full((B * hq * nsplit, 64), float("nan"))
    m_part = torch.full((B * hq * nsplit,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    zero = torch.zeros(32)
    for b in range(B):
        n = int(data["cache_seqlens"][b])
        lo = max(n - window, 0) if window > 0 else 0
        per = (n - lo + nsplit - 1) // nsplit
        per = (per + CHUNK - 1) // CHUNK * CHUNK
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            poff = kv0 * 64
            # qa[ks] = 8 A elements per lane (a0..a7)
            qrow = qf[b, (kv0 + HL) * 8 + G_OF]                      # [32, 64]: head hl, query g
            base = 32 * (T_OF & 1)
            qa = []
            for ks in range(8):
                vals = [f16(qrow[LANES, base + 4 * ks + i]) for i in range(4)]
                h0 = HL == 0
                qa.append(torch.stack([torch.where(h0, vals[0], zero), torch.where(h0, vals[1], zero),
                                       torch.where(~h0, vals[0], zero), torch.where(~h0, vals[1], zero),
                                       torch.where(h0, vals[2], zero), torch.where(h0, vals[3], zero),
                                       torch.where(~h0, vals[2], zero), torch.where(~h0, vals[3], zero)], 1))
            for split in range(nsplit):
                start = lo + split * per
                end = min(start + per, n)
                nchunks = (end - start + CHUNK - 1) // CHUNK if end > start else 0
                sm_m = torch.full((warps, 2, 8), -1.0e30)
                sm_l = torch.zeros((warps, 2, 8))
                sm_o = torch.zeros((warps, 2, 8, 64))
                for w in range(warps):
                    m_run = [torch.full((32,), -1.0e30), torch.full((32,), -1.0e30)]
                    l_run = [torch.zeros(32), torch.zeros(32)]
                    acc = torch.zeros(16, 32, 4)
                    mine = (nchunks - w + warps - 1) // warps if w < nchunks else 0
                    ring = torch.zeros(stages, STAGE_BYTES, dtype=torch.uint8)

                    def row_of(i):
                        s = torch.clamp(start + (w + i * warps) * CHUNK + (LANES & 15), max=end - 1)
                        return table[b, s // page].long() * page + s % page

                    def issue(i, my_row):
                        if i >= mine:
                            return
                        st = i % stages
                        for qq in range(4):
                            for ln in range(32):
                                tok, ch = qq * 4 + (ln >> 3), ln & 7
                                row = int(my_row[tok])
                                ring[st, tok * PITCH + 16 * ch: tok * PITCH + 16 * ch + 16] = kc[row, poff + 16 * ch: poff + 16 * ch + 16]
                                o = CHUNK * PITCH + tok * PITCH + 16 * ch
                                ring[st, o: o + 16] = vc[row, poff + 16 * ch: poff + 16 * ch + 16]

                    for i in range(stages - 1):
                        issue(i, row_of(i) if i < mine else None)
                    row_next = row_of(stages - 1) if stages - 1 < mine else None
                    for i in range(mine):
                        issue(i + stages - 1, row_next)
                        if i + stages < mine:
                            row_next = row_of(i + stages)
                        c = w + i * warps
                        st = ring[i % stages]
                        sk, sv = st[:CHUNK * PITCH], st[CHUNK * PITCH:]
                        k = []
                        for gg in (G_OF, 8 + G_OF):
                            for half in range(2):
                                k.append(sk[(gg * PITCH + 32 * T_OF + 16 * half)[:, None] + torch.arange(16)])
                        vtok = [2 * T_OF, 2 * T_OF + 1, 2 * T_OF + 8, 2 * T_OF + 9]
                        v = [sv[(vt * PITCH + 16 * G_OF)[:, None] + torch.arange(16)] for vt in vtok]
                        s0 = start + c * CHUNK
                        sc = []
                        for nt in range(2):
                            cfr = torch.zeros(32, 4)
                            for ks in range(8):
                                kw = k[nt * 2 + (ks >> 2)][:, 4 * (ks & 3):4 * (ks & 3) + 4]
                                blo = e4m3(kw[:, 0:2].contiguous())
                                bhi = e4m3(kw[:, 2:4].contiguous())
                                bf = torch.stack([blo[:, 0], blo[:, 1], bhi[:, 0], bhi[:, 1]], 1)
                                cfr = mma(cfr, qa[ks], bf)
                            sc.append(cfr)
                        x = torch.zeros(2, 32, 4)
                        for nt in range(2):
                            for jj in range(2):
                                tok = s0 + nt * 8 + 2 * T_OF + jj
                                ok = tok < end
                                x[nt][:, jj] = torch.where(ok, sc[nt][:, jj] * qk_scale, float("-inf"))
                                x[nt][:, 2 + jj] = torch.where(ok, sc[nt][:, 2 + jj] * qk_scale, float("-inf"))
                        mx0 = torch.maximum(x[0][:, 0:2].max(1).values, x[1][:, 0:2].max(1).values)
                        mx1 = torch.maximum(x[0][:, 2:4].max(1).values, x[1][:, 2:4].max(1).values)
                        mx0 = mx0.view(8, 4).max(1).values.repeat_interleave(4)
                        mx1 = mx1.view(8, 4).max(1).values.repeat_interleave(4)
                        mn0 = torch.maximum(m_run[0], mx0)
                        mn1 = torch.maximum(m_run[1], mx1)
                        al0, al1 = torch.exp2(m_run[0] - mn0), torch.exp2(m_run[1] - mn1)
                        p = torch.zeros(2, 32, 4)
                        for nt in range(2):
                            p[nt][:, 0] = torch.exp2(x[nt][:, 0] - mn0)
                            p[nt][:, 1] = torch.exp2(x[nt][:, 1] - mn0)
                            p[nt][:, 2] = torch.exp2(x[nt][:, 2] - mn1)
                            p[nt][:, 3] = torch.exp2(x[nt][:, 3] - mn1)
                        l_run[0] = l_run[0] * al0 + (p[0][:, 0] + p[0][:, 1] + p[1][:, 0] + p[1][:, 1])
                        l_run[1] = l_run[1] * al1 + (p[0][:, 2] + p[0][:, 3] + p[1][:, 2] + p[1][:, 3])
                        m_run = [mn0, mn1]
                        a = torch.stack([f16(p[0][:, 0]), f16(p[0][:, 1]), f16(p[0][:, 2]), f16(p[0][:, 3]),
                                         f16(p[1][:, 0]), f16(p[1][:, 1]), f16(p[1][:, 2]), f16(p[1][:, 3])], 1)
                        for j in range(16):
                            acc[j][:, 0] *= al0
                            acc[j][:, 1] *= al0
                            acc[j][:, 2] *= al1
                            acc[j][:, 3] *= al1
                            wi = j >> 2
                            ws = [v[i][:, 4 * wi:4 * wi + 4] for i in range(4)]
                            sel = (j & 3) | ((4 + (j & 3)) << 4)
                            b0 = e4m3(byte_perm(ws[0], ws[1], sel)[:, 0:2].contiguous())
                            b1 = e4m3(byte_perm(ws[2], ws[3], sel)[:, 0:2].contiguous())
                            bf = torch.stack([b0[:, 0], b0[:, 1], b1[:, 0], b1[:, 1]], 1)
                            acc[j] = mma(acc[j], a, bf)
                    l0 = l_run[0].view(8, 4).sum(1)
                    l1 = l_run[1].view(8, 4).sum(1)
                    sm_m[w, 0] = m_run[0].view(8, 4)[:, 0]
                    sm_m[w, 1] = m_run[1].view(8, 4)[:, 0]
                    sm_l[w, 0] = l0
                    sm_l[w, 1] = l1
                    for j in range(16):
                        for jj in range(2):
                            d = 32 * (T_OF & 1) + 16 * jj + j
                            vals = torch.where(HL == 0, acc[j][:, jj], acc[j][:, 2 + jj])
                            sm_o[w, HL, G_OF, d] = vals
                mm = sm_m.max(0).values
                wt = torch.exp2(sm_m - mm)
                o = (wt[..., None] * sm_o).sum(0)
                lsum = (wt * sm_l).sum(0)
                for h in range(2):
                    for row in range(8):
                        part = (b * hq + (kv0 + h) * 8 + row) * nsplit + split
                        o_part[part] = o[h, row]
                        m_part[part] = mm[h, row]
                        l_part[part] = lsum[h, row]
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    return o_part, m_part, l_part


def run_cuda3(data, sub, nsplit, warps=4, stages=3):
    q = prologue(data, sub)
    o, m, l_ = attend_dec3(data, q, nsplit, warps, stages)
    cb = min(64, 1 << (nsplit - 1).bit_length())
    return combine(data, o, m, l_, nsplit, cb, sub)


def main():
    import reference as ref
    sub = load_submission()
    quick = "--quick" in sys.argv
    cases = [
        (dict(batch=1, ctx=10, heads="tp8", layer="sliding"), [1, 3]),
        (dict(batch=3, ctx=2047, heads="tp8", layer="sliding", page=1, dist="soft"), [1, 5, 16]),
        (dict(batch=2, ctx=3000, heads="tp8", layer="full", page=1, dist="coherent"), [1, 7, 40]),
        (dict(batch=2, ctx=2600, heads="tp8", layer="full", page=64), [3]),
        (dict(batch=2, ctx=4000, heads="tp8", layer="sliding", dist="edges", page=1), [4]),
        (dict(batch=2, ctx=3000, heads="tp8", layer="full", dist="edges"), [2]),
        (dict(batch=2, ctx=2500, heads="tp8", layer="full", kv_scale=0.015625, page=1), [3]),
        (dict(batch=1, ctx=1500, heads="dp", layer="full", dist="coherent"), [2]),
        (dict(batch=2, ctx=2100, heads="dp", layer="sliding", page=1), [2]),
    ]
    if quick:
        cases = [cases[1], cases[2], cases[3], cases[4], cases[7]]
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            for warps in (4,) if quick else (4, 8):
                data = ref.generate_input(seed=410 + i, device="cpu", **case)
                out = run_cuda3(data, sub, nsplit, warps, 3 if warps == 4 else 2)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} warps {warps} -> {msg}")
                bad += not ok
    for nsplit in (1, 2):
        data = tail_key_case(ref)
        out = run_cuda3(data, sub, nsplit, 4, 4)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"newest key weighted, partial last chunk, nsplit {nsplit} -> {msg}")
        bad += not ok
    print("emulate_cuda_dec3:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

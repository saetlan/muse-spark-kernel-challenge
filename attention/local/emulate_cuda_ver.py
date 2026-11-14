"""Register- and shared-memory-level CPU emulation of attention/local/cuda_ver.py's attend_ver_kernel
(S2a, q_len 8): the STAGES-deep ring as byte arrays with the kernel's pitches (K 192 B, V 144 B),
its copy mapping (thread -> token tid>>3, 16 B chunk tid&7), issue order and stage indices; per
warp (head w>>1, m-tiles 2*(w&1)+mi) the fragment reads, cvt/byte_perm, MMAs placed by the PTX
m16n8k16 formulas, causal/window/tail masks, online softmax and the partial writes. The
prologue and merge are emulate.py's (the Triton kernels around it).
Run: .venv/bin/python attention/local/emulate_cuda_ver.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF, byte_perm, e4m3, f16, mma  # noqa: E402

CHUNK, L, KP, VP = 16, 8, 192, 144


def attend_ver(data, q, nsplit, stages=4):
    cfg = data["config"]
    assert cfg["q_len"] == L
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq // hkv == 8 and hkv % 2 == 0
    B = cfg["batch"]
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    table = data["page_table"]
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    qf = q.reshape(B * L, hq, 64).float()
    T = B * L
    o_part = torch.full((T * hq * nsplit, 64), float("nan"))
    m_part = torch.full((T * hq * nsplit,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    tid = torch.arange(128)
    ctok, cch = tid >> 3, tid & 7
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
                sk = torch.zeros(stages, CHUNK * KP, dtype=torch.uint8)
                sv = torch.zeros(stages, CHUNK * VP, dtype=torch.uint8)

                def row_of(c):
                    s = torch.clamp(start + c * CHUNK + ctok, max=end - 1)
                    return table[b, s // page].long() * page + s % page

                def issue(c, row):
                    st = c % stages
                    for t in range(128):        # each thread's two 16 B copies
                        r, k_ = int(row[t]), int(cch[t])
                        tk = int(ctok[t])
                        sk[st, tk * KP + 16 * k_: tk * KP + 16 * k_ + 16] = kc[r, poff + 16 * k_: poff + 16 * k_ + 16]
                        sv[st, tk * VP + 16 * k_: tk * VP + 16 * k_ + 16] = vc[r, poff + 16 * k_: poff + 16 * k_ + 16]

                # per warp state
                warps = []
                for w in range(4):
                    h, mt0 = w >> 1, 2 * (w & 1)
                    qa = []
                    for mi in range(2):
                        ta = 2 * (mt0 + mi)
                        head = (kv0 + h) * 8 + G_OF
                        qA = qf[b * L + ta, head]           # [32, 64]
                        qB = qf[b * L + ta + 1, head]
                        per_ks = []
                        for ks in range(4):
                            base = 16 * T_OF + 4 * ks
                            vals = lambda qq, i: f16(qq[LANES, base + i])  # noqa: E731
                            per_ks.append(torch.stack([vals(qA, 0), vals(qA, 1), vals(qB, 0), vals(qB, 1),
                                                       vals(qA, 2), vals(qA, 3), vals(qB, 2), vals(qB, 3)], 1))
                        qa.append(per_ks)
                    warps.append(dict(h=h, mt0=mt0, qa=qa, m=[[torch.full((32,), -1.0e30)] * 2 for _ in range(2)],
                                      l=[[torch.zeros(32)] * 2 for _ in range(2)],
                                      acc=[torch.zeros(8, 32, 4) for _ in range(2)]))
                for i in range(stages - 1):
                    if i < nchunks:
                        issue(i, row_of(i))
                rnext = row_of(stages - 1) if stages - 1 < nchunks else None
                for c in range(nchunks):
                    if c + stages - 1 < nchunks:
                        issue(c + stages - 1, rnext)
                    if c + stages < nchunks:
                        rnext = row_of(c + stages)
                    st = c % stages
                    s0 = start + c * CHUNK
                    masked = s0 + CHUNK > end or s0 + CHUNK > n - L or window > 0
                    for wd in warps:
                        h, mt0 = wd["h"], wd["mt0"]
                        kb = [sk[st][(g * KP + 64 * h + 16 * T_OF)[:, None] + torch.arange(16)]
                              for g in (G_OF, 8 + G_OF)]
                        vtok = [2 * T_OF, 2 * T_OF + 1, 2 * T_OF + 8, 2 * T_OF + 9]
                        vb = [sv[st][(vt * VP + 64 * h + 8 * G_OF)[:, None] + torch.arange(8)] for vt in vtok]
                        for mi in range(2):
                            sc = []
                            for nt in range(2):
                                cfr = torch.zeros(32, 4)
                                for ks in range(4):
                                    word = kb[nt][:, 4 * ks:4 * ks + 4]
                                    blo, bhi = e4m3(word[:, 0:2].contiguous()), e4m3(word[:, 2:4].contiguous())
                                    bf = torch.stack([blo[:, 0], blo[:, 1], bhi[:, 0], bhi[:, 1]], 1)
                                    cfr = mma(cfr, wd["qa"][mi][ks], bf)
                                sc.append(cfr)
                            posA = n - L + 2 * (mt0 + mi)
                            posB = posA + 1
                            x = torch.zeros(2, 32, 4)
                            for nt in range(2):
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
                            mxA = torch.maximum(x[0][:, 0:2].max(1).values, x[1][:, 0:2].max(1).values)
                            mxB = torch.maximum(x[0][:, 2:4].max(1).values, x[1][:, 2:4].max(1).values)
                            mxA = mxA.view(8, 4).max(1).values.repeat_interleave(4)
                            mxB = mxB.view(8, 4).max(1).values.repeat_interleave(4)
                            mA, mB = wd["m"][mi]
                            mnA, mnB = torch.maximum(mA, mxA), torch.maximum(mB, mxB)
                            alA, alB = torch.exp2(mA - mnA), torch.exp2(mB - mnB)
                            p = torch.zeros(2, 32, 4)
                            for nt in range(2):
                                p[nt][:, 0] = torch.exp2(x[nt][:, 0] - mnA)
                                p[nt][:, 1] = torch.exp2(x[nt][:, 1] - mnA)
                                p[nt][:, 2] = torch.exp2(x[nt][:, 2] - mnB)
                                p[nt][:, 3] = torch.exp2(x[nt][:, 3] - mnB)
                            lA, lB = wd["l"][mi]
                            wd["l"][mi] = [lA * alA + (p[0][:, 0] + p[0][:, 1] + p[1][:, 0] + p[1][:, 1]),
                                           lB * alB + (p[0][:, 2] + p[0][:, 3] + p[1][:, 2] + p[1][:, 3])]
                            wd["m"][mi] = [mnA, mnB]
                            a = torch.stack([f16(p[0][:, 0]), f16(p[0][:, 1]), f16(p[0][:, 2]), f16(p[0][:, 3]),
                                             f16(p[1][:, 0]), f16(p[1][:, 1]), f16(p[1][:, 2]), f16(p[1][:, 3])], 1)
                            acc = wd["acc"][mi]
                            for j in range(8):
                                acc[j][:, 0] *= alA
                                acc[j][:, 1] *= alA
                                acc[j][:, 2] *= alB
                                acc[j][:, 3] *= alB
                                half = 0 if j < 4 else 4
                                ws = [vb[i][:, half:half + 4] for i in range(4)]
                                sel = (j & 3) | ((4 + (j & 3)) << 4)
                                b0 = e4m3(byte_perm(ws[0], ws[1], sel)[:, 0:2].contiguous())
                                b1 = e4m3(byte_perm(ws[2], ws[3], sel)[:, 0:2].contiguous())
                                bf = torch.stack([b0[:, 0], b0[:, 1], b1[:, 0], b1[:, 1]], 1)
                                acc[j] = mma(acc[j], a, bf)
                for wd in warps:
                    h, mt0 = wd["h"], wd["mt0"]
                    for mi in range(2):
                        for r in range(2):
                            lsum = wd["l"][mi][r].view(8, 4).sum(1).repeat_interleave(4)
                            tok = 2 * (mt0 + mi) + r
                            part = ((b * L + tok) * hq + (kv0 + h) * 8 + G_OF) * nsplit + split
                            for j in range(8):
                                for jj in range(2):
                                    o_part[part, 8 * (2 * T_OF + jj) + j] = wd["acc"][mi][j][:, 2 * r + jj]
                            m_part[part] = wd["m"][mi][r]
                            l_part[part] = lsum
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    assert not torch.isnan(o_part).any(), "an output partial was never written"
    return o_part, m_part, l_part


def run_ver(data, sub, nsplit, stages=4):
    q = prologue(data, sub)
    o, m, l_ = attend_ver(data, q, nsplit, stages)
    return combine(data, o, m, l_, nsplit, 1 << max(0, (nsplit - 1).bit_length()), sub)


def main():
    import reference as ref
    sub = load_submission()
    quick = "--quick" in sys.argv
    cases = [
        (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1, 5]),
        (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [1, 4]),
        (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"), [3]),
        (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [3]),
        (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [2]),
        (dict(batch=1, ctx=1500, heads="dp", layer="full", q_len=8, dist="coherent"), [2]),
        (dict(batch=1, ctx=40, heads="tp8", layer="full", q_len=8, page=64), [1, 2]),
    ]
    if quick:
        cases = [cases[0], cases[1], cases[3], cases[5]]
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            for stages in (3,) if quick else (3, 4):
                data = ref.generate_input(seed=700 + i, device="cpu", **case)
                out = run_ver(data, sub, nsplit, stages)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} stages {stages} -> {msg}")
                bad += not ok
    from emulate import window_rows_case
    for page in (1, 64):
        data = window_rows_case(ref, page)
        out = run_ver(data, sub, 3)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}) -> {msg}")
        bad += not ok
    print("emulate_cuda_ver:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

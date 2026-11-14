"""Register-level CPU emulation of attention/local/cuda_dec.py's attend_dec_kernel (A5), lane by
lane (vectorized over the 32 lanes): the same loads (16 B of K, 8 B of V per lane), the same
__byte_perm selectors and cvt halves, fragments placed by the PTX ISA 9.4 m16n8k16 layout
formulas (section 9.7.16.5.8), FP16 rounding of Q, K, V and P, the warps' round-robin chunks,
online softmax and the shared-memory merge. The partials then go through the emulated
_combine and the judge's check_implementation.

Run: .venv/bin/python attention/local/emulate_cuda_dec.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402

CHUNK = 16
LANES = torch.arange(32)
G_OF = LANES >> 2          # groupID
T_OF = LANES & 3           # threadID_in_group


def f16(x):
    return x.to(torch.float16).float()


def e4m3(byte_tensor):
    """uint8 bytes -> float (exact; FP16 holds every E4M3 value)."""
    return byte_tensor.view(torch.float8_e4m3fn).float()


def byte_perm(x, y, sel):
    """__byte_perm(x, y, s) on [lanes, 4] byte tensors: result byte i = ({y,x} bytes)[nibble i of s]."""
    both = torch.cat([x, y], -1)                              # bytes 0-3 = x, 4-7 = y
    idx = [(sel >> (4 * i)) & 7 for i in range(4)]
    return both[:, idx]


def mma(c, a_frags, b_frags):
    """mma.m16n8k16 f32.f16.f16.f32 for one warp. a_frags [32, 8] (a0..a7 per lane), b_frags
    [32, 4] (b0..b3), c [32, 4] (c0..c3), placed by the PTX layout formulas."""
    A = torch.zeros(16, 16)
    B = torch.zeros(16, 8)
    for i in range(8):
        row = G_OF if (i < 2 or 4 <= i < 6) else G_OF + 8
        col = T_OF * 2 + (i & 1) + (8 if i >= 4 else 0)
        A[row, col] = a_frags[:, i]
    for i in range(4):
        row = T_OF * 2 + (i & 1) + (8 if i >= 2 else 0)
        B[row, G_OF] = b_frags[:, i]
    D = A @ B
    out = c.clone()
    for i in range(4):
        row = G_OF if i < 2 else G_OF + 8
        col = T_OF * 2 + (i & 1)
        out[:, i] += D[row, col]
    return out


def attend_dec(data, q, nsplit, warps=4):
    """Partials (o [T*HQ*nsplit, 64], m, l) exactly as the kernel writes them (q_len 1)."""
    cfg = data["config"]
    assert cfg["q_len"] == 1
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq // hkv == 8
    B = cfg["batch"]
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    table = data["page_table"]
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    qf = q.reshape(B, hq, 64).float()                        # BF16 values from the prologue
    o_part = torch.full((B * hq * nsplit, 64), float("nan"))
    m_part = torch.full((B * hq * nsplit,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    for b in range(B):
        n = int(data["cache_seqlens"][b])
        lo = max(n - window, 0) if window > 0 else 0
        per = (n - lo + nsplit - 1) // nsplit
        per = (per + CHUNK - 1) // CHUNK * CHUNK
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            # qa[h][ks] = (a0, a1, a4, a5) values per lane; a2, a3, a6, a7 = 0 (rows g+8)
            qa = {}
            for h in range(2):
                rows = qf[b, (kv0 + h) * 8 + G_OF]                    # [32, 64]
                for ks in range(4):
                    base = 16 * T_OF + 4 * ks
                    qa[h, ks] = [f16(rows[LANES, base + i]) for i in range(4)]
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
                    acc = [torch.zeros(8, 32, 4), torch.zeros(8, 32, 4)]
                    for c in range(w, nchunks, warps):
                        s = torch.clamp(start + c * CHUNK + (LANES & 15), max=end - 1)
                        my_row = table[b, s // page].long() * page + s % page
                        rk = [my_row[G_OF], my_row[8 + G_OF]]
                        rv = [my_row[2 * T_OF], my_row[2 * T_OF + 1], my_row[2 * T_OF + 8], my_row[2 * T_OF + 9]]
                        s0 = start + c * CHUNK
                        for h in range(2):
                            hoff = (kv0 + h) * 64
                            kr = [kc[rk[nt][:, None], hoff + 16 * T_OF[:, None] + torch.arange(16)] for nt in range(2)]
                            vr = [vc[rv[i][:, None], hoff + 8 * G_OF[:, None] + torch.arange(8)] for i in range(4)]
                            sc = []
                            for nt in range(2):
                                cfr = torch.zeros(32, 4)
                                for ks in range(4):
                                    word = kr[nt][:, 4 * ks:4 * ks + 4]       # bytes of uint4 word ks
                                    b_lo = e4m3(word[:, 0:2].contiguous())     # cvt of the low 16 bits
                                    b_hi = e4m3(word[:, 2:4].contiguous())     # cvt of the high 16 bits
                                    a = torch.stack([qa[h, ks][0], qa[h, ks][1], torch.zeros(32), torch.zeros(32),
                                                     qa[h, ks][2], qa[h, ks][3], torch.zeros(32), torch.zeros(32)], 1)
                                    bf = torch.stack([b_lo[:, 0], b_lo[:, 1], b_hi[:, 0], b_hi[:, 1]], 1)
                                    cfr = mma(cfr, a, bf)
                                sc.append(cfr)
                            xs = []
                            for nt in range(2):
                                for jj in range(2):
                                    tok = s0 + nt * 8 + 2 * T_OF + jj
                                    xs.append(torch.where(tok < end, sc[nt][:, jj] * qk_scale, float("-inf")))
                            mx = torch.stack(xs, 1).max(1).values
                            mx = mx.view(8, 4).max(1).values.repeat_interleave(4)   # xor 1, 2 within the group
                            m_new = torch.maximum(m_run[h], mx)
                            alpha = torch.exp2(m_run[h] - m_new)
                            p = [torch.exp2(x - m_new) for x in xs]           # p00, p01, p10, p11
                            l_run[h] = l_run[h] * alpha + (p[0] + p[1] + p[2] + p[3])
                            m_run[h] = m_new
                            a = torch.stack([f16(p[0]), f16(p[1]), torch.zeros(32), torch.zeros(32),
                                             f16(p[2]), f16(p[3]), torch.zeros(32), torch.zeros(32)], 1)
                            for j in range(8):
                                acc[h][j][:, 0] *= alpha
                                acc[h][j][:, 1] *= alpha
                                half = 0 if j < 4 else 4                       # .x or .y word of each uint2
                                ws = [vr[i][:, half:half + 4] for i in range(4)]
                                sel = (j & 3) | ((4 + (j & 3)) << 4)
                                p01 = byte_perm(ws[0], ws[1], sel)
                                p89 = byte_perm(ws[2], ws[3], sel)
                                b0 = e4m3(p01[:, 0:2].contiguous())
                                b1 = e4m3(p89[:, 0:2].contiguous())
                                bf = torch.stack([b0[:, 0], b0[:, 1], b1[:, 0], b1[:, 1]], 1)
                                acc[h][j] = mma(acc[h][j], a, bf)
                    for h in range(2):
                        lsum = l_run[h].view(8, 4).sum(1)                      # xor 1, 2 reduction
                        sm_m[w, h] = m_run[h].view(8, 4)[:, 0]
                        sm_l[w, h] = lsum
                        for j in range(8):
                            for jj in range(2):
                                sm_o[w, h, G_OF, 8 * (2 * T_OF + jj) + j] = acc[h][j][:, jj]
                mm = sm_m.max(0).values                                       # [2, 8]
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


def run_cuda(data, sub, nsplit, warps=4):
    q = prologue(data, sub)
    o, m, l_ = attend_dec(data, q, nsplit, warps)
    cb = min(64, 1 << (nsplit - 1).bit_length())
    return combine(data, o, m, l_, nsplit, cb, sub)


def tail_key_case(ref):
    """Full-layer decode whose newest key (position n-1, written by the prologue from qkv) has a
    moderate share of each group's first head's softmax (~0.3: cosine 0.47 with the query), and
    whose last chunk is partial (3001 keys = 187 chunks + 9): if the tail past `end` (which reads
    the last valid row again) were not masked, that key would count up to 8 times."""
    data = ref.generate_input(batch=1, ctx=3000, heads="tp8", layer="full", page=1, dist="soft",
                              seed=91, device="cpu")
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"].view(-1).float().clone()
    gen = torch.Generator().manual_seed(5)
    for h in range(hkv):
        g0 = h * (hq // hkv)
        qd = qkv[g0 * 64:(g0 + 1) * 64].clone()
        for g in range(1, hq // hkv):     # the whole group shares the direction: every head's row moves
            qkv[(g0 + g) * 64:(g0 + g + 1) * 64] = qd
        qd = qd / qd.norm()
        z = torch.randn(64, generator=gen)
        z = z - (z @ qd) * qd
        k = 0.47 * qd + (1 - 0.47 ** 2) ** 0.5 * z / z.norm()
        qkv[hq * 64 + h * 64:hq * 64 + (h + 1) * 64] = k * 8.0
    data["qkv"] = qkv.view_as(data["qkv"]).to(torch.bfloat16)
    return data


def main():
    import reference as ref
    sub = load_submission()
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
    if "--quick" in sys.argv:
        cases = [cases[1], cases[2], cases[3], cases[4], cases[7]]
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if "--quick" in sys.argv else splits:
            for warps in (4,) if "--quick" in sys.argv else (4, 8):
                data = ref.generate_input(seed=300 + i, device="cpu", **case)
                out = run_cuda(data, sub, nsplit, warps)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} warps {warps} -> {msg}")
                bad += not ok
    for nsplit in (1, 2):
        data = tail_key_case(ref)
        out = run_cuda(data, sub, nsplit)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"newest key weighted, partial last chunk, nsplit {nsplit} -> {msg}")
        bad += not ok
    print("emulate_cuda_dec:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

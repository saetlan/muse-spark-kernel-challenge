"""Register- and shared-memory-level CPU emulation of attention/local/cuda_ver5.py (V5: V4's producer, query
fragments and partial layout; one softmax round per 64-key stage, edge-only masks, raw-score max
times qk_scale, P = exp2(fma(s, qk_scale, 8 - m)), P.V over the stage's two 32-key chunks into one
fresh accumulator per output tile). Fragments are placed by emulate_cuda_ver4's PTX m16n8k32 e4m3
formulas; ring bytes by V4's producer mapping; per consumer warp w (MT = 1): head w // 4, m-tile w % 4.
Partials in V4's layout, merged by emulate.py's combine; outputs through the judge's check_implementation.
MUT (mutation_test_cuda_ver5.py) plants single bugs: each must make some case fail.
Run: .venv/bin/python attention/local/emulate_cuda_ver5.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission, prologue  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF  # noqa: E402
from emulate_cuda_ver4 import CHUNK, KP, VP, SKEYS, STAGE_BYTES, P_OFF, e4m3x4, tr4, mma_e4m3  # noqa: E402

L = 8
MUT = set()
VFRAG = None   # emulate_cuda_ver5t.py: V fragments through the producer's re-laid tile


def v_frags(sv, c, h):
    """V5's consumer V fragments of chunk c, head h: keys {2t4, 2t4+1, 2t4+8, 2t4+9} (+16) of the chunk
    at bytes 64h + 8g .., 4x4 byte transposes. Returns (vb0, vb1): 8 tensors [32 lanes, 4 bytes] each."""
    base = 0 if "chunk_v" in MUT else 32 * c
    rows = [sv[((base + vk + add) * VP + 64 * h + 8 * G_OF)[:, None] + torch.arange(8)]
            for add in (0, 16) for vk in (2 * T_OF, 2 * T_OF + 1, 2 * T_OF + 8, 2 * T_OF + 9)]
    return (tr4(*[r[:, 0:4] for r in rows[0:4]]) + tr4(*[r[:, 4:8] for r in rows[0:4]]),
            tr4(*[r[:, 0:4] for r in rows[4:8]]) + tr4(*[r[:, 4:8] for r in rows[4:8]]))


def attend_ver5(data, q, nsplit, stages=3):
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
                nst = (nchunks + 3) // 4
                ring = torch.zeros(stages, STAGE_BYTES, dtype=torch.uint8)

                def produce(i):
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
                            ring.view(-1)[o:o + 16] = kc[row, poff + 16 * ch: poff + 16 * ch + 16]
                            o = st * STAGE_BYTES + SKEYS * KP + tq * VP + 16 * ch
                            ring.view(-1)[o:o + 16] = vc[row, poff + 16 * ch: poff + 16 * ch + 16]

                warps = []
                for w in range(8):
                    h, mt0 = w // 4, w % 4
                    ta = 2 * mt0
                    head = (kv0 + h) * 8 + G_OF
                    qA, qB = qf[b * L + ta, head], qf[b * L + ta + 1, head]
                    steps = []
                    for s in range(2):
                        o = 16 * T_OF + 8 * s
                        col = lambda qq, k: qq[LANES, o + k]  # noqa: E731
                        steps.append(torch.stack([e4m3x4(col(qA, 0), col(qA, 1), col(qA, 2), col(qA, 3)),
                                                  e4m3x4(col(qB, 0), col(qB, 1), col(qB, 2), col(qB, 3)),
                                                  e4m3x4(col(qA, 4), col(qA, 5), col(qA, 6), col(qA, 7)),
                                                  e4m3x4(col(qB, 4), col(qB, 5), col(qB, 6), col(qB, 7))], 1))
                    warps.append(dict(h=h, mt0=mt0, qa=steps, m=[torch.full((32,), -1.0e30)] * 2,
                                      l=[torch.zeros(32)] * 2, acc=torch.zeros(8, 32, 4)))
                for i in range(nst):
                    produce(i)
                    st = i % stages
                    s0 = start + i * SKEYS
                    sk = ring[st][:SKEYS * KP]
                    sv = ring[st][SKEYS * KP:]
                    masked = s0 + SKEYS > end
                    if "mask_causal" not in MUT:
                        masked = masked or s0 + SKEYS > n - L
                    if "mask_window" not in MUT:
                        masked = masked or (window > 0 and s0 < lo + L)
                    for wd in warps:
                        h, mt0 = wd["h"], wd["mt0"]
                        sc = []
                        for nt in range(8):
                            kb = sk[((8 * nt + G_OF) * KP + 64 * h + 16 * T_OF)[:, None] + torch.arange(16)]
                            w4 = [kb[:, 4 * k:4 * k + 4] for k in range(4)]
                            cfr = mma_e4m3(torch.zeros(32, 4), wd["qa"][0], torch.stack([w4[0], w4[1]], 1))
                            cfr = mma_e4m3(cfr, wd["qa"][1], torch.stack([w4[2], w4[3]], 1))
                            sc.append(cfr)
                        posA = n - L + 2 * mt0
                        posB = posA + 1
                        x = torch.zeros(8, 32, 4)
                        for nt in range(8):
                            for jj in range(2):
                                key = s0 + nt * 8 + 2 * T_OF + jj
                                xa, xb = sc[nt][:, jj], sc[nt][:, 2 + jj]
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
                        mA, mB = wd["m"]
                        sm = 1.0 if "scale_max" in MUT else qk_scale
                        mnA, mnB = torch.maximum(mA, mxA * sm), torch.maximum(mB, mxB * sm)
                        alA, alB = torch.exp2(mA - mnA), torch.exp2(mB - mnB)
                        p = torch.zeros(8, 32, 4)
                        for nt in range(8):
                            p[nt][:, 0] = torch.exp2(x[nt][:, 0] * qk_scale + (P_OFF - mnA))
                            p[nt][:, 1] = torch.exp2(x[nt][:, 1] * qk_scale + (P_OFF - mnA))
                            p[nt][:, 2] = torch.exp2(x[nt][:, 2] * qk_scale + (P_OFF - mnB))
                            p[nt][:, 3] = torch.exp2(x[nt][:, 3] * qk_scale + (P_OFF - mnB))
                        lA, lB = wd["l"]
                        wd["l"] = [lA * alA + p[:, :, 0:2].sum(dim=(0, 2)), lB * alB + p[:, :, 2:4].sum(dim=(0, 2))]
                        wd["m"] = [mnA, mnB]
                        pa, vb0, vb1 = [], [], []
                        for c in range(2):
                            cp = (1 - c) if "pack_chunk" in MUT else c
                            q4 = 4 * cp
                            pa.append(torch.stack([e4m3x4(p[q4][:, 0], p[q4][:, 1], p[q4 + 1][:, 0], p[q4 + 1][:, 1]),
                                                   e4m3x4(p[q4][:, 2], p[q4][:, 3], p[q4 + 1][:, 2], p[q4 + 1][:, 3]),
                                                   e4m3x4(p[q4 + 2][:, 0], p[q4 + 2][:, 1], p[q4 + 3][:, 0], p[q4 + 3][:, 1]),
                                                   e4m3x4(p[q4 + 2][:, 2], p[q4 + 2][:, 3], p[q4 + 3][:, 2], p[q4 + 3][:, 3])], 1))
                            f0, f1 = (VFRAG or v_frags)(sv, c, h)
                            vb0.append(f0)
                            vb1.append(f1)
                        acc = wd["acc"]
                        for dt in range(8):
                            pv = mma_e4m3(torch.zeros(32, 4), pa[0], torch.stack([vb0[0][dt], vb1[0][dt]], 1))
                            pv = mma_e4m3(pv, pa[1], torch.stack([vb0[1][dt], vb1[1][dt]], 1))
                            acc[dt][:, 0] = acc[dt][:, 0] * alA + pv[:, 0]
                            acc[dt][:, 1] = acc[dt][:, 1] * alA + pv[:, 1]
                            acc[dt][:, 2] = acc[dt][:, 2] * (alA if "al_row" in MUT else alB) + pv[:, 2]
                            acc[dt][:, 3] = acc[dt][:, 3] * alB + pv[:, 3]
                for wd in warps:
                    h, mt0 = wd["h"], wd["mt0"]
                    for r in range(2):
                        lsum = wd["l"][r].view(8, 4).sum(1).repeat_interleave(4)
                        tok = 2 * mt0 + r
                        part = ((b * L + tok) * hq + (kv0 + h) * 8 + G_OF) * nsplit + split
                        for dt in range(8):
                            for jj in range(2):
                                o_part[part, 16 * T_OF + 8 * jj + dt] = wd["acc"][dt][:, 2 * r + jj]
                        m_part[part] = wd["m"][r]
                        l_part[part] = lsum
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    assert not torch.isnan(o_part).any(), "an output partial was never written"
    return o_part, m_part, l_part


def run_ver5(data, sub, nsplit, stages=3):
    q = prologue(data, sub)
    o, m, l_ = attend_ver5(data, q, nsplit, stages)
    return combine(data, o, m, l_, nsplit, 1 << max(0, (nsplit - 1).bit_length()), sub)


CASES = [
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1, 5]),
    (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [1, 4]),
    (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"), [3]),
    (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [3]),
    (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [2]),
    (dict(batch=1, ctx=1500, heads="dp", layer="full", q_len=8, dist="coherent"), [2]),
    (dict(batch=1, ctx=40, heads="tp8", layer="full", q_len=8, page=64), [1, 2]),
    # n = 640: one slice whose last stage [576, 640) ends exactly at the slice end (only the causal term masks it)
    (dict(batch=1, ctx=632, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), [1]),
]


def run_cases(quick, verbose=True):
    import reference as ref
    from emulate import window_rows_case
    sub = load_submission()
    cases = [CASES[0], CASES[1], CASES[3], CASES[7]] if quick else CASES
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            for stages in (3,) if quick else (3, 4):
                data = ref.generate_input(seed=740 + i, device="cpu", **case)
                out = run_ver5(data, sub, nsplit, stages)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                if verbose:
                    print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} stages {stages} -> {msg}")
                bad += not ok
    for page in (1, 64):
        data = window_rows_case(ref, page)
        out = run_ver5(data, sub, 3, 3)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        if verbose:
            print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}) -> {msg}")
        bad += not ok
    return bad


if __name__ == "__main__":
    bad = run_cases("--quick" in sys.argv)
    print("emulate_cuda_ver5:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

"""CPU emulation of attention/local/cuda_uw2.py (UW2: UV4's in-kernel prologue and cluster merge around W2c), with the
kernel's index formulas:
- the block's keys / values (UV4 section 4): warp w = token l, lane: head lane >> 4, dims 4*(lane & 15) .. +3, RMS over the
  head's 16 lanes, RoPE frequency d0/2 + p, E4M3 bytes at new_k[128*l + 4*lane ..]; written to the cache at cache_loc;
- the query fold (UV4 section 3): warp w, lane (g, t4): rows token 2w + r, head (kv0 + h)*8 + g, dims 16*t4 .. +15, RMS
  over the 4 lanes of the group, RoPE pair p = frequency 8*t4 + p; its E4M3 bytes form W2's Q tile (one 16 B store at
  tile_off(16w + g + 8r, 16*t4)), so the attention runs on q = the folded queries with qk_scale_factor in the scale;
- the copier reads the cache with garbage in the block's rows (the kernel's copy may predate the write); the transposers'
  patch (cuda_uw2.py, before the transpose): in a stage with min(s0 + 63, end - 1) >= n - L, row r with clamped key
  min(s0 + r, end - 1) >= n - L gets the new K bytes at (c >> 2)*KC_HEAD + (r >> 3)*SBO_K + (c & 3)*LBO_K + (r & 7)*16
  and the new V bytes at OFF_VS + r*VS_PITCH + 16c, c = 0..7;
- attention: emulate_cuda_w1.attend_w1 in W2 mode (Q / P tiles, SS wgmma) with nsplit = cs, the copier's view and the patch;
- push: row R = 16*token + 8h + g -> owner R // RPC, slot rank*RPC + R % RPC (every slot written exactly once, checked);
  finalize per owner warp / slot (UV4 step 7 + 9): merge, v_scale / l, RMSNorm, sigmoid(gate).
Outputs and the new cache entries go through the judge's check_implementation. MUT plants single bugs (each must fail).
Run: .venv/bin/python attention/local/emulate_cuda_uw2.py [--quick] [--mut]"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_w1 as ew  # noqa: E402
from emulate_cuda_dec import LANES, G_OF, T_OF  # noqa: E402
from emulate_cuda_uv4 import bf, garbage, L, LBP, ROWS  # noqa: E402
from emulate_cuda_ver4 import SKEYS, to_e4m3_bytes  # noqa: E402

MUT = set()


def uw2(data, cs, stages=4, seed=0, npar=0):
    """npar 2: UW3 (make_uw3.py): W3's pair interleave inside each CTA (attend_w1 npar=2: partial split * 2 + par), then
    pair 0 merges pair 1's (o, m, l) by the kernel's formula before the push."""
    cfg = data["config"]
    assert cfg["q_len"] == L and cfg["window"] > 0
    B, hq, hkv, page, window = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    assert hq == 8 * hkv and hkv % 2 == 0
    rope = bool(cfg["rope_theta"])
    WC, RPC = 8, ROWS // cs
    RPW = RPC // WC
    gen = torch.Generator().manual_seed(seed)
    qkv = data["qkv"].reshape(B, L, -1)
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)      # the kernel writes the block's rows here
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    positions, cache_loc = data["positions"], data["cache_loc"]
    inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float) / 64)) if rope else torch.ones(32)
    k_scale, v_scale = float(data["k_scale"]), float(data["v_scale"])
    qk_factor = cfg["qk_scale_factor"]
    kinv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(k_scale, dtype=torch.float32)
    vinv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(v_scale, dtype=torch.float32)
    kc_seen, vc_seen = kc.clone(), vc.clone()                          # the copier's view
    for row in cache_loc.tolist():
        kc_seen[row] = garbage(gen, hkv * 64)
        vc_seen[row] = garbage(gen, hkv * 64)
    new_k = torch.zeros(B, hkv // 2, L, 128, dtype=torch.uint8)
    new_v = torch.zeros(B, hkv // 2, L, 128, dtype=torch.uint8)
    q_fold = torch.zeros(B * L, hq, 64)
    for b in range(B):
        rc, rs = torch.zeros(L, 32), torch.zeros(L, 32)
        if rope:
            for e in range(L * 32):
                ang = torch.tensor(float(positions[b * L + (e >> 5)]), dtype=torch.float32) * inv_freq[e & 31]
                rs[e >> 5, e & 31] = math.sin(float(ang))
                rc[e >> 5, e & 31] = math.cos(float(ang))
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            poff = kv0 * 64
            for l_ in range(L):            # UV4 section 4 (every CTA holding key n - L + l computes the same bytes)
                hk, d0 = LANES >> 4, 4 * (LANES & 15)
                kidx = hq * 64 + (kv0 + hk)[:, None] * 64 + d0[:, None] + torch.arange(4)
                k4 = bf(qkv[b, l_][kidx])
                v4 = bf(qkv[b, l_][kidx + hkv * 64])
                ss = (k4 * k4).sum(1).view(-1, 16).sum(1).repeat_interleave(16)
                k4 = k4 * torch.rsqrt(ss / 64 + 1e-5)[:, None]
                if rope:
                    for p in range(2):
                        f = (d0 // 2 + p).clamp(max=31)
                        sn, cs_ = rs[l_][f], rc[l_][f]
                        e_, o_ = k4[:, 2 * p].clone(), k4[:, 2 * p + 1].clone()
                        k4[:, 2 * p], k4[:, 2 * p + 1] = e_ * cs_ - o_ * sn, e_ * sn + o_ * cs_
                kb = to_e4m3_bytes(k4 * kinv).reshape(-1)
                vb = to_e4m3_bytes(v4 * vinv).reshape(-1)
                new_k[b, pair, l_], new_v[b, pair, l_] = kb, vb
                crow = int(cache_loc[b * L + l_])
                kc[crow, poff:poff + 128] = kb
                vc[crow, poff:poff + 128] = vb
            for w in range(4):             # UV4 section 3 with W2's Q-tile store (dims 16*t4 .. +15 of each row)
                for h in range(2):
                    for r in range(2):
                        tok = 2 * w + r
                        head = (kv0 + h) * 8 + G_OF
                        x = bf(qkv[b, tok][head[:, None] * 64 + 16 * T_OF[:, None] + torch.arange(16)])   # [32, 16]
                        ss = (x * x).sum(1).view(-1, 4).sum(1).repeat_interleave(4)
                        x = x * torch.rsqrt(ss / 64 + 1e-5)[:, None]
                        if rope:
                            for p in range(8):
                                f = 8 * T_OF + p
                                sn, cs_ = rs[tok][f], rc[tok][f]
                                e_, o_ = x[:, 2 * p].clone(), x[:, 2 * p + 1].clone()
                                x[:, 2 * p], x[:, 2 * p + 1] = e_ * cs_ - o_ * sn, e_ * sn + o_ * cs_
                        dbase = 16 * T_OF if "fold_dims" not in MUT else 16 * (T_OF ^ 1)
                        for ln in range(32):
                            q_fold[b * L + tok, int(head[ln]), int(dbase[ln]):int(dbase[ln]) + 16] = x[ln]

    def patch(smem, base, b, kv0, start, end, n, i):
        if "no_patch" in MUT:
            return
        s0 = start + i * SKEYS
        if min(s0 + SKEYS - 1, end - 1) < n - L:
            return
        for r in range(SKEYS):
            key = min(s0 + r, end - 1)
            if (key > n - L) if "fresh_cond" in MUT else (key >= n - L):
                nl = key - (n - L)
                rr = (r ^ 1) if "patch_row" in MUT else r
                nk, nv = new_k[b, kv0 // 2, nl], new_v[b, kv0 // 2, nl]
                for c in range(8):
                    a = base + (c >> 2) * ew.KC_HEAD + (rr >> 3) * ew.SBO_K + (c & 3) * ew.LBO_K + (rr & 7) * 16
                    smem[a:a + 16] = nk[16 * c:16 * c + 16]
                    a = base + ew.OFF_VS + rr * ew.VS_PITCH + 16 * c
                    smem[a:a + 16] = nv[16 * c:16 * c + 16]

    data2 = dict(data)
    data2["config"] = dict(cfg, sm_scale=cfg["sm_scale"] * qk_factor)   # UV4: qk_scale_factor joins the score scale
    o_part, m_part, l_part = ew.attend_w1(data2, q_fold, cs, stages, 528, w2=True, kv_view=(kc_seen, vc_seen), patch=patch,
                                          npar=npar)
    if npar == 2:   # UW3: pair 0 merges pair 1 (kernel formula: mn = max, a0 = 2^(m0 - mn), a1 = 2^(m1 - mn))
        o2, m2, l2 = o_part.view(-1, 2, 64), m_part.view(-1, 2), l_part.view(-1, 2)
        if "pair_merge" in MUT:
            o_part, m_part, l_part = o2[:, 0].clone(), m2[:, 0].clone(), l2[:, 0].clone()
        else:
            mn = torch.maximum(m2[:, 0], m2[:, 1])
            a0, a1 = torch.exp2(m2[:, 0] - mn), torch.exp2(m2[:, 1] - mn)
            o_part = o2[:, 0] * a0[:, None] + o2[:, 1] * a1[:, None]
            l_part = l2[:, 0] * a0 + l2[:, 1] * a1
            m_part = mn
    out = torch.full((B, L, hq * 64), float("nan"))
    for b in range(B):
        for pair in range(hkv // 2):
            kv0 = 2 * pair
            rb = torch.full((cs, cs * RPC, LBP), float("nan"))
            wr = torch.zeros(cs, cs * RPC, dtype=torch.int64)
            for rank in range(cs):         # push: consumer thread (warpgroup h, warp w, lane g / t4), half
                for h in range(2):
                    for w in range(4):
                        for half in range(2):
                            tok = 2 * w + half
                            for g in range(8):
                                R = tok * 16 + (h * 8 + g if "push_row" not in MUT else g * 2 + h)
                                own, slot = R // RPC, rank * RPC + R % RPC
                                part = ((b * L + tok) * hq + (kv0 + h) * 8 + g) * cs + rank
                                rb[own, slot, :64] = o_part[part]
                                rb[own, slot, 64], rb[own, slot, 65] = m_part[part], l_part[part]
                                wr[own, slot] += 1
            assert bool((wr == 1).all()), f"receive buffers: slots written {int(wr.min())}..{int(wr.max())} times"
            for own in range(cs):          # finalize: warp w, rows slot = w + j*WC
                for w in range(WC):
                    for j in range(RPW):
                        slot = w + j * WC
                        R = own * RPC + slot
                        tok, hh = R >> 4, R & 15
                        src = torch.arange(cs) * RPC + slot
                        m_ = rb[own, src, 64]
                        mm = max(-1.0e30, float(m_.max()))
                        wt = torch.exp2(m_ - mm)
                        l_ = float((wt * rb[own, src, 65]).sum())
                        o = (wt[:, None] * rb[own, src, :64]).sum(0) * (v_scale / l_)
                        rn = torch.rsqrt((o * o).sum() / 64 + 1e-5)
                        gt = bf(qkv[b, tok, (hq + 2 * hkv) * 64 + (kv0 * 8 + hh) * 64:(hq + 2 * hkv) * 64 + (kv0 * 8 + hh) * 64 + 64])
                        out[b, tok, (kv0 * 8 + hh) * 64:(kv0 * 8 + hh) * 64 + 64] = o * rn / (1 + torch.exp(-gt))
    assert not torch.isnan(out).any(), "an output element was never written"
    return out.to(torch.bfloat16)


CASES = [
    (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), [8, 4]),
    (dict(batch=1, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1, dist="coherent"), [16]),
    (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1), [8]),
    (dict(batch=2, ctx=2000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625), [8]),
    (dict(batch=1, ctx=40, heads="tp8", layer="sliding", q_len=8, page=64), [8, 16]),
    # n = 197, cs 4: slices [128, 192) and [192, 197) split the block's keys 189 .. 196 over two CTAs
    (dict(batch=1, ctx=189, heads="tp8", layer="sliding", q_len=8, page=1), [4]),
    (dict(batch=3, ctx=2100, heads="tp8", layer="sliding", q_len=8, page=1, spread="agentic"), [8]),
]


def run_cases(quick, verbose=True, npar=0):
    import reference as ref
    from emulate import window_rows_case
    cases = [CASES[0], CASES[1], CASES[4], CASES[5]] if quick else CASES
    bad = 0
    for i, (case, css) in enumerate(cases):
        for cs in css[:1] if quick else css:
            data = ref.generate_input(seed=860 + i, device="cpu", **case)
            try:
                out = uw2(data, cs, 4, seed=i, npar=npar)
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
            except AssertionError as exc:
                ok, msg = False, f"assertion: {exc}"
            if verbose:
                print(("PASS " if ok else "FAIL ") + f"{case} cs {cs} -> {msg}")
            bad += not ok
    data = window_rows_case(ref, 1)
    try:
        out = uw2(data, 8, 4, seed=99, npar=npar)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
    except AssertionError as exc:
        ok, msg = False, f"assertion: {exc}"
    if verbose:
        print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page 1, cs 8) -> {msg}")
    bad += not ok
    return bad


if __name__ == "__main__":
    npar = 2 if "--uw3" in sys.argv else 0
    if "--mut" in sys.argv:
        bad = 0
        for m in ("no_patch", "patch_row", "fresh_cond", "fold_dims", "push_row") + (("pair_merge",) if npar else ()):
            MUT.clear()
            MUT.add(m)
            ew.MUT = set()
            caught = run_cases(True, verbose=False, npar=npar) > 0
            print(("caught  " if caught else "MISSED  ") + m)
            bad += not caught
        MUT.clear()
        print("emulate_cuda_uw2 mutants:", "all caught" if not bad else f"{bad} MISSED")
        sys.exit(1 if bad else 0)
    bad = run_cases("--quick" in sys.argv, npar=npar)
    print("emulate_cuda_uw2" + (" (UW3: pair interleave + merge)" if npar else "") + ":", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)

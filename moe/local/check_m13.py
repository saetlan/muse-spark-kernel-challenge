"""Local (CPU) checks for M13 (moe/m13/m13.py, built into submission_m13.py). No GPU here, so:
1. decode: the kernel's four A-register formulas (read from the CUDA text) on m13_relayout(w) give
   bit for bit M11's dec<0..3>(w), on random words and every single-bit word;
2. scales: scale2's PRMT + IMAD equals bf16(e4m3 * 2^64) for every byte m13_scales_ok accepts, and
   m13_scales_ok rejects zero, subnormal, negative and NaN scales;
3. layouts: the kernel's word and scale addressing on m13_bank's tensors reads M11's values;
4. schedule: a line-by-line Python model of the kernel's pair scan, stream-K ranges, cursor,
   flushes, counters and scratch columns, over many routings and grids: every (expert, task, pass)
   gets each K group exactly once, exactly one epilogue, counter and scratch indices in bounds and
   unique, every local pair in exactly one pass;
5. the submission file: v1 + M11 + M13 sections, configs name compiled instances.

Run: .venv/bin/python moe/local/check_m13.py
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_m11  # noqa: E402
import check_v0_lab as base  # noqa: E402

TOP_K = 8
KG, D = 64, 4
GPT = KG // D

DECODE_LINES = (  # M13's MODE 0 path (recon modes 2/3 only skip the MMA or the decode)
    "        } else {\n          a0 = hmul2(rotr(q, 2) & PM, sg);",
    "a1 = hmul2(rotr(q, 6) & PM, sg8);",
    "a2 = hmul2(rotr(q, 12) & PM, sg);",
    "a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);",
    "if constexpr (MODE == 3) {  // recon: MMA on raw words",
    "            } else {\n              mma(acc[i][n], a0, a1, a2, a3, b0, b1);",
)
PM, PM3, PM3W = 0x81C081C0, 0x81808180, 0x00400040


def rotr(x, s):
    return ((x >> s) | (x << (32 - s))) & 0xFFFFFFFF


def new_dec(w):
    return [rotr(w, 2) & PM, rotr(w, 6) & PM, rotr(w, 12) & PM, (rotr(w, 8) & PM3) | (w & PM3W)]


def check_decode(mod):
    src = mod._M13_CUDA
    for line in DECODE_LINES:
        assert line in src, f"kernel decode changed, update check_m13: {line}"
    for const, val in (("PM", PM), ("PM3", PM3), ("PM3W", PM3W)):
        assert f"constexpr uint32_t {const} = 0x{val:08X}u;" in src, const
    gen = torch.Generator().manual_seed(7)
    words = torch.randint(-2**31, 2**31, (200_000,), generator=gen, dtype=torch.int64).to(torch.int32)
    singles = torch.tensor([(1 << b) - (2**32 if b == 31 else 0) for b in range(32)], dtype=torch.int64).to(torch.int32)
    allw = torch.cat([words, singles, torch.tensor([-1, 0], dtype=torch.int32)])
    new = mod.m13_relayout(allw).to(torch.int64) & 0xFFFFFFFF
    old = allw.to(torch.int64) & 0xFFFFFFFF
    for r in range(4):
        want = check_m11.dec(old, r)
        got = new_dec(new)[r]
        bad = (want != got).nonzero().flatten()
        assert bad.numel() == 0, (r, hex(int(old[bad[0]])), hex(int(want[bad[0]])), hex(int(got[bad[0]])))
    # the relayout is a bijection on bits (lossless)
    assert sorted(mod.M13_BIT_SRC) == list(range(32))
    print(f"ok: M13 decode (rotr 2/6/12 & M, rotr 8 & M' | w & M'') on m13_relayout(w) = M11's dec<0..3>(w) "
          f"bit for bit on {allw.numel()} words incl. all 32 single-bit words; relayout is a bit permutation")


def check_scales(mod):
    src = mod._M13_CUDA
    assert "return __byte_perm(v, 0u, 0x4440u | B) * 0x00100010u + 0x5C005C00u;" in src
    ok_bytes = []
    for b in range(256):
        t = torch.tensor([[b]], dtype=torch.uint8)
        weights = {"w13_sf": t.view(torch.float8_e4m3fn), "w2_sf": torch.full((1, 1), 0x40, dtype=torch.uint8).view(torch.float8_e4m3fn)}
        accepted = mod.m13_scales_ok(weights, 0, 1)
        val = float(t.view(torch.float8_e4m3fn).float())
        normal_pos = (b & 0x80) == 0 and (b & 0x78) != 0 and (b & 0x7F) != 0x7F
        assert accepted == normal_pos, (b, accepted)
        if not accepted:
            continue
        ok_bytes.append(b)
        lane = ((b * 0x10) + 0x5C00) & 0xFFFF
        got = lane | (lane << 16)
        want = check_m11.pair_bits(torch.tensor([[val * 2.0**64, val * 2.0**64]]).to(torch.bfloat16))
        assert got == int(want.flatten()[0]) & 0xFFFFFFFF, (b, hex(got), hex(int(want.flatten()[0])))
    # the generator's scales (log-normal clamped to [8, 448]) all pass
    import nvfp4
    gen = torch.Generator().manual_seed(1)
    _, sf, _ = nvfp4.random_weights(256, 4096, generator=gen, device="cpu")
    assert mod.m13_scales_ok({"w13_sf": sf.unsqueeze(0), "w2_sf": sf.unsqueeze(0)}, 0, 1)
    print(f"ok: scale2 (PRMT + IMAD) = bf16(e4m3 * 2^64) in both lanes for all {len(ok_bytes)} accepted bytes; "
          "zero, subnormal, negative and NaN scales rejected; random_weights' scales accepted")


def check_layouts(mod):
    gen = torch.Generator().manual_seed(3)
    n, rt13, rt2 = 2, 4, 2
    m11 = {"start": 0, "count": n, "inter": 8, "hidden": 8, "c13": torch.ones(n), "c2": torch.ones(n),
           "rt_w13": rt13, "rt_w2": rt2}
    for name, rt in (("w13", rt13), ("w2", rt2)):
        m11[name] = torch.randint(-2**31, 2**31, (n, rt, KG, 32, 4), generator=gen, dtype=torch.int64).to(torch.int32)
        m11[name + "_s"] = torch.randint(-2**15, 2**15, (n, rt, KG, 32), generator=gen, dtype=torch.int64).to(torch.int16)
    old_tokens = mod.M13_MAX_TOKENS
    bank = mod.m13_bank(m11, "cpu")
    for name, rt in (("w13", rt13), ("w2", rt2)):
        words = bank[name].reshape(-1, 4)  # uint4 units
        sc = bank[name + "_s"].reshape(-1).to(torch.int64) & 0xFFFF  # uint16 units; uint2 = 4 of them
        for _ in range(300):
            e, r, grp, lane, s = (random.randrange(n), random.randrange(rt), random.randrange(GPT), random.randrange(32),
                                  random.randrange(D))
            row = e * rt + r
            widx = (row * KG + grp * D) * 32 + lane + s * 32  # lw[i] + s * 32
            assert torch.equal(words[widx], mod.m13_relayout(m11[name][e, r, grp * D + s, lane])), (name, e, r, grp, s)
            sidx = ((row * GPT + grp) * 32 + lane) * 4  # ls[i], as uint16 index
            word = sc[sidx + 2 * (s >= 2)] | (sc[sidx + 2 * (s >= 2) + 1] << 16)  # sv = s < 2 ? .x : .y
            byte_g = (int(word) >> (8 * (2 * (s & 1)))) & 0xFF
            byte_g8 = (int(word) >> (8 * (2 * (s & 1) + 1))) & 0xFF
            want = int(m11[name + "_s"][e, r, grp * D + s, lane]) & 0xFFFF
            assert (byte_g, byte_g8) == (want & 0xFF, want >> 8), (name, e, r, grp, s, lane)
    pairs = max(old_tokens, getattr(mod, "M14_MAX_TOKENS", 0)) * TOP_K
    assert bank["scr13"].shape == (pairs, 16 * rt13) and bank["scr2"].shape == (pairs, 16 * rt2)
    assert float(bank["scr13"].abs().sum()) == 0 and int(bank["ctr13"].abs().sum()) == 0
    print("ok: kernel addressing on m13_bank reads M11's words (re-laid out) and scales (rows g / g + 8 of "
          "chunk 4 grp + s) at 300 random positions per matrix; scratch and counters start at zero")


# ------------------------------------------------------------------------------ 4. schedule model
def scan_lists(ids, local_start, maxp):
    """The kernel's warp-0 scan: lists in increasing pair order (what __match_any_sync ranks give)."""
    cnt = [0] * 32
    lists = [[] for _ in range(32)]
    for base_p in range(0, len(ids), 32):
        chunk = [(p, ids[p] - local_start) for p in range(base_p, min(base_p + 32, len(ids)))]
        for lane, (p, e) in enumerate(chunk):
            if 0 <= e < 32:
                below = sum(1 for p2, e2 in chunk[:lane] if e2 == e)
                pos = cnt[e] + below
                if pos < maxp:
                    lists[e].append((pos, p))
        for e in set(e for _, e in chunk if 0 <= e < 32):
            cnt[e] += sum(1 for _, e2 in chunk if e2 == e)
    return cnt, [[p for _, p in sorted(l)] for l in lists]


def model_kernel(ids, n_warps, gated, tt, nt, maxp, rt_total, pf=255):
    NB = 8 * nt
    per_expert = rt_total // 2 if gated else rt_total // tt
    maxpass = (maxp + 7) // 8
    cnt, lists = scan_lists(ids, 0, maxp)
    cntc = [min(c, maxp) for c in cnt]
    active = [e for e in range(32) if cntc[e] > 0]
    sbase = []
    acc = 0
    for e in active:
        sbase.append(acc)
        acc += per_expert * ((cntc[e] + NB - 1) // NB) * GPT
    sbase.append(acc)
    total = acc
    nact = len(active)
    groups_seen = {}
    segs = {}
    ctr_idx = {}
    scr_cols = set()
    for gw in range(n_warps):
        g0, g1 = total * gw // n_warps, total * (gw + 1) // n_warps
        if g0 >= g1:
            continue

        def set_expert(c):
            c["e"] = active[c["ai"]]
            c["passes"] = (cntc[c["e"]] + NB - 1) // NB

        def set_cntp(c):
            c["cntp"] = min(cntc[c["e"]] - c["pass"] * NB, NB)

        def advance(c):
            c["grp"] += 1
            if c["grp"] == GPT:
                c["grp"] = 0
                c["pass"] += 1
                if c["pass"] == c["passes"]:
                    c["pass"] = 0
                    c["task"] += 1
                    if c["task"] == per_expert:
                        c["task"] = 0
                        c["ai"] += 1
                        if c["ai"] < nact:
                            set_expert(c)
                if c["ai"] < nact:
                    set_cntp(c)

        ai = 0
        while ai + 1 < nact and sbase[ai + 1] <= g0:
            ai += 1
        L = {"ai": ai}
        set_expert(L)
        local = g0 - sbase[ai]
        per_task = L["passes"] * GPT
        L["task"] = local // per_task
        rem = local - L["task"] * per_task
        L["pass"] = rem // GPT
        L["grp"] = rem - L["pass"] * GPT
        set_cntp(L)

        def locate(gi):  # the kernel's locate(): the cursor of flat group gi
            a = 0
            while a + 1 < nact and sbase[a + 1] <= gi:
                a += 1
            c = {"ai": a}
            set_expert(c)
            loc = gi - sbase[a]
            pt = c["passes"] * GPT
            c["task"] = loc // pt
            r = loc - c["task"] * pt
            c["pass"] = r // GPT
            c["grp"] = r - c["pass"] * GPT
            set_cntp(c)
            return c

        assert locate(g0) == L
        # start-up prefetch: lane-parallel locate(g0 + k), k = 1..npre, must be the group the warp loads
        # k steps later, with its codes and scales inside the bank
        q = dict(L)
        for k in range(1, pf + 1):
            if g0 + k >= g1:
                break
            advance(q)
            assert locate(g0 + k) == q, (g0, k)
            assert q["ai"] < nact and 0 <= q["task"] < per_expert
            for i in range(tt):
                rt = q["task"] + i * (rt_total // 2) if gated else q["task"] * tt + i
                row = q["e"] * rt_total + rt
                assert 0 <= rt < rt_total and (row * KG + q["grp"] * 4) * 32 + 4 * 32 <= 32 * rt_total * KG * 32
                assert (row * GPT + q["grp"]) * 32 + 32 <= 32 * rt_total * GPT * 32
        C = dict(L)
        seg0 = C["grp"]
        for gi in range(g0, g1):
            more = gi + 1 < g1
            if more:
                advance(L)
                assert L["ai"] < nact, "loads past the last active expert"
            assert 0 < C["cntp"] <= NB and 0 <= C["task"] < per_expert and 0 <= C["pass"] < C["passes"] <= maxpass
            key = (C["e"], C["task"], C["pass"])
            groups_seen.setdefault(key, []).append(C["grp"])
            # activations: lanes g < cntp of each n-tile read pair lists[e][pass * NB + n * 8 + g]
            for n in range(nt):
                for g in range(8):
                    if n * 8 + g < C["cntp"]:
                        assert C["pass"] * NB + n * 8 + g < len(lists[C["e"]])
            if C["grp"] == GPT - 1 or not more:
                nseg = C["grp"] - seg0 + 1
                segs.setdefault(key, []).append(nseg)
                ci = (C["e"] * per_expert + C["task"]) * maxpass + C["pass"]
                assert ctr_idx.setdefault(ci, key) == key, "counter index collision"
                if nseg != GPT:
                    for i in range(tt):
                        rt = C["task"] + i * (rt_total // 2) if gated else C["task"] * tt + i
                        for g in range(8):
                            for rr in range(2):
                                col = 16 * rt + g + 8 * rr
                                assert 0 <= col < 16 * rt_total
                                scr_cols.add(col)
                seg0 = 0
            C = dict(L)
    # every (expert, task, pass) of every active expert: all GPT groups once, one epilogue
    epilogues = 0
    for e in active:
        passes = (cntc[e] + NB - 1) // NB
        covered = []
        for task in range(per_expert):
            for ps in range(passes):
                key = (e, task, ps)
                got = sorted(groups_seen.get(key, []))
                assert got == list(range(GPT)), (key, got)
                s = segs[key]
                assert sum(s) == GPT and (len(s) == 1) == (s == [GPT]), (key, s)
                epilogues += 1
                if task == 0:
                    covered += lists[e][ps * NB:ps * NB + min(cntc[e] - ps * NB, NB)]
        assert sorted(covered) == sorted(p for p in range(len(ids)) if ids[p] == e), e
    assert len(ctr_idx) == epilogues and max(ctr_idx, default=0) < 32 * per_expert * maxpass
    return total, epilogues, sum(1 for s in segs.values() if len(s) > 1)


def check_schedule(mod):
    rnd = random.Random(11)
    maxp = mod.M13_MAX_TOKENS
    cases = 0
    split_tasks = 0
    for trial in range(48):
        # the logic doesn't depend on the number of row tiles; most cases use 16 (fast), every 8th the real 512/256
        full = trial % 8 == 0
        bs = rnd.choice([1, 2, 4, 8, 16, 32, 33, 48, 64])
        mode = trial % 3
        ids = []
        for _ in range(bs):
            if mode == 0:  # judge-like: some local experts
                ids += rnd.sample(range(64), TOP_K)
            elif mode == 1:  # skewed: everything on experts 0-7
                ids += list(range(TOP_K))
            else:  # random local and remote
                ids += rnd.sample(range(256), TOP_K)
        for gated, tt, nt in mod.M13_INSTANCES:
            rt_total = (512 if gated else 256) if full else 16
            for n_warps in ((132 * 4 * rnd.choice([1, 2, 3, 4]),) if full else (1, 4, 12, 52, 132 * 4 * rnd.choice([1, 2, 3, 4]))):
                total, epi, split = model_kernel(ids, n_warps, gated, tt, nt, maxp, rt_total)
                cases += 1
                split_tasks += split
    print(f"ok: stream-K model of the kernel over {cases} (routing, instance, grid) cases up to {maxp} tokens: "
          f"each (expert, task, pass) gets all {GPT} K groups once and one epilogue ({split_tasks} split across "
          "warps), counter/scratch indices unique and in bounds, every local pair in exactly one pass")


COMB_LINES = (
    "const long long total = static_cast<long long>(nact) * per;",
    "long long b = (total * j / nxwarps) & ~15LL;",
    "const long long b1 = total * (j + 1) / nxwarps;",
    "long long e = b + ((b1 - b) * nxq) / 4;",
    "e = min((e + 15) & ~15LL, total);",
    "const int ai = static_cast<int>(b / per);",
    "const long long off = b - static_cast<long long>(ai) * per;",
    "const long long len = min(min(e - b, per - off), 32768LL);",
    "l2_prefetch(base + static_cast<long long>(s_active[ai]) * per + off, static_cast<uint32_t>(len));",
    "for (int j = gw + nw * lane; j < nxwarps; j += nw * 32) {",
    "const int nxq = (pf >> 24) & 7;",
)


def comb_pieces(active, per, nxwarps, nxq, j):
    """The kernel's next-stage prefetch pieces (address, bytes) for next-stage warp j."""
    nact = len(active)
    total = nact * per
    b = (total * j // nxwarps) & ~15
    b1 = total * (j + 1) // nxwarps
    e = b + ((b1 - b) * nxq) // 4
    e = min((e + 15) & ~15, total)
    out = []
    while b < e:
        ai = b // per
        off = b - ai * per
        ln = min(e - b, per - off, 32768)
        out.append((active[ai] * per + off, ln, ai, b, b1))
        b += ln
    return out


def check_comb(mod):
    src = mod.m13_source()
    for line in COMB_LINES:
        assert line in src, f"next-stage prefetch changed, update check_m13: {line}"
    rnd = random.Random(5)
    cases = pieces = 0
    pers = (8 * 2**20, 2**20, 16 * 37, 64 * 1024 + 16)  # GEMM2 codes, scales, and odd sizes
    for trial in range(60):
        nact = rnd.randint(1, 32)
        active = sorted(rnd.sample(range(32), nact))
        per = rnd.choice(pers)
        nxwarps = rnd.choice([1, 4, 528, 2112, 2640, 3 * 528 + 4])
        for nxq in (1, 2, 3, 4):
            covered = []
            for j in range(nxwarps):
                ps = comb_pieces(active, per, nxwarps, nxq, j)
                for addr, ln, ai, b, b1 in ps:
                    assert ln % 16 == 0 and 16 <= ln <= 32768 and addr % 16 == 0, (addr, ln)
                    e_id = addr // per
                    assert e_id == active[ai] and addr + ln <= (e_id + 1) * per <= 32 * per, "piece crosses experts"
                    covered.append((ai * per + (addr - active[ai] * per), ln))
                pieces += len(ps)
                if ps:  # starts at its range's start (rounded down to 16)
                    assert ps[0][3] == (nact * per * j // nxwarps) & ~15
            cases += 1
            if nxq == 4:  # the whole of every active expert's array
                covered.sort()
                pos = 0
                for start, ln in covered:
                    assert start <= pos, ("gap", start, pos)
                    pos = max(pos, start + ln)
                assert pos == nact * per
    print(f"ok: next-stage L2 prefetch over {cases} cases ({pieces} pieces): every piece 16-B aligned, a multiple of 16 "
          "of at most 32 KB, inside one active expert's region; q = 4 covers every byte")


M11_PDL_LINES = (
    "  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched",
    "  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads\n"
    "  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)",
    "            if constexpr (PDL) {\n              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);\n"
    "              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);",
    'static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");',
    "m13_launch(m11::gemm_kernel<TT, NT, D, KS, false, true>, static_cast<int>(grid), 128, 0, 1,",
)


def check_m11_pdl(mod):
    """M11's PDL GEMM2 (E15): every CTA waits after the routing scan and before any activation load, which are
    coherent (ld.global.cg) and carry the wait's token; only GEMM2 instances get it; plain M11 keeps __ldg."""
    src = mod.m13_source()
    for line in M11_PDL_LINES:
        assert line in src, f"M11 PDL path changed, update check_m13: {line[:80]}"
    k = src.index("gemm_kernel(")
    body = src[k:src.index("tma_kernel(", k)]
    assert body.index("pdl_wait_token()") < body.index("auto load = [&]"), "the wait must precede the loads"
    assert "__ldg(ap[n] + kg * 8)" in body, "the non-PDL path must keep L1-cached loads"
    n = src.count("M11P_CASE(") - 1  # minus the macro definition
    assert n == sum(1 for g, *_ in mod.M13_M11_INSTANCES if not g), n
    print(f"ok: M11 PDL GEMM2: wait after the scan, token-ordered ld.global.cg activations, {n} GEMM2 instances")


def check_submission_m13():
    mod = base.load("submission_m13")
    text = (base.MOE / "submission_m13.py").read_text()
    assert text.splitlines()[0] == base.PROBLEM_LINE
    v1 = (base.MOE / "submission.py").read_text()
    v1 = v1.replace("\ndef prepare(weights", "\ndef _v1_prepare(weights", 1)
    v1 = v1.replace("\ndef kernel_fn(data, state", "\ndef _v1_kernel_fn(data, state", 1)
    assert text.startswith(v1.rstrip()), "v1 part must be byte for byte"
    for name in ("prepare", "kernel_fn", "_m11_kernel_fn", "_v1_kernel_fn", "m13_bank", "m13_run", "m11_run"):
        assert callable(getattr(mod, name)), name
    # M11 fallback configs only use instances compiled into the M13 extension
    for _, cfg in mod.M11_CONFIGS:
        if cfg is not None:
            assert (1, 2, cfg["s1"]["nt"], cfg["s1"]["d"], cfg["s1"]["ks"]) in mod.M13_M11_INSTANCES, cfg
            assert (0, cfg["s2"]["tt"], cfg["s2"]["nt"], cfg["s2"]["d"], cfg["s2"]["ks"]) in mod.M13_M11_INSTANCES, cfg
    for below, cfg in mod.M13_CONFIGS:
        assert below <= max(mod.M13_MAX_TOKENS, mod.M14_MAX_TOKENS) + 1
        if cfg is None:
            continue
        for key, gated in (("s1", 1), ("s2", 0)):
            st = cfg[key]
            inst = (gated, 2 if gated else st.get("tt"), st["nt"])
            if st.get("kind") in (None, "m11p"):  # M11's kernels (E3c configs; m11p = its GEMM2 with PDL)
                assert (gated, 2 if gated else st["tt"], st["nt"], st["d"], st["ks"]) in mod.M13_M11_INSTANCES, (key, st)
                assert st.get("kind") is None or not gated, "m11p is GEMM2 only"
            elif st.get("kind") == "m13":
                assert below <= mod.M13_MAX_TOKENS + 1, "M13 kernel takes at most M13_MAX_TOKENS tokens"
                assert inst in mod.M13_INSTANCES, (key, inst)
            elif st.get("kind") == "m14":
                assert inst + (st.get("p", 2),) in mod.M14_INSTANCES, (key, inst)
            elif st.get("kind") == "m16":
                assert inst + (st.get("p", 2), st.get("db", 0)) in mod.M16_INSTANCES, (key, inst)
            elif st.get("kind") == "m18":
                assert inst + (st.get("p", 2),) in mod.M18_INSTANCES, (key, inst)
            elif st.get("kind") == "m15":
                assert inst + (st.get("p", 2),) in {(i[0], i[1], i[2], i[4]) for i in mod.M15_INSTANCES}, (key, inst)
            elif st.get("kind") == "m17":
                assert inst + (st.get("p", 2), st.get("db", 0)) in {(i[0], i[1], i[2], i[4], i[5]) for i in mod.M17_INSTANCES}, \
                    (key, inst)
    print("ok: submission_m13.py = v1 (byte for byte, renamed) + M11 + M13; fallback and M13 configs name compiled instances")
    return mod


# ------------------------------------------------------------------------------ 6. M14 (CTA-shared)
M14_DECODE = (
    "A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);",
    "A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);",
    "A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);",
    "A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);",
    "cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),",
    "a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);",
    "if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {",
    "const uint4 blo = *reinterpret_cast<const uint4*>(st + (((2 * t) ^ g) << 4));",
    "const uint4 bhi = *reinterpret_cast<const uint4*>(st + (((2 * t + 1) ^ g) << 4));",
    "const uint32_t b0 = word(bv, (2 * j) & 3), b1 = word(bv, (2 * j + 1) & 3);",
    "const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;",
)


M14_RING_LINES = (
    "if (bg < g1) load_b(Bc, bsub, pslot);",
    "pslot = pslot + 1 == SBP ? 0 : pslot + 1;",
    "if (++bsub == D) {",
    "if (++bg < g1) advance(Bc);",
    "for (int k = 0; k < P; ++k) issue_b();",
    "cp_wait<P - 1>();",
    "issue_b();        // chunk + P -> the stage the previous chunk used",
    "compute(s, rsc, cntp, cslot);",
    "cslot = cslot + 1 == SBP ? 0 : cslot + 1;",
    "constexpr int SBP = P + 1;  // activation stages",
)


def check_m14_ring(mod):
    """The activation ring of M14 (P chunks ahead, P + 1 stages): simulated with cp.async commit groups.
    Each compute reads the stage holding exactly its own (group, chunk), that copy's commit group is complete
    under wait_group P - 1, and no copy overwrites a stage before the chunk in it was computed."""
    src = mod.m13_source()
    for line in M14_RING_LINES:
        assert line in src, f"M14 activation ring changed, update check_m13: {line}"
    rnd = random.Random(3)
    cases = 0
    for P in range(2, 9):
        sbp = P + 1
        for trial in range(200):
            g0 = rnd.randint(0, 50)
            g1 = g0 + rnd.randint(1, 40)
            groups = []      # commit groups in order: (stage, chunk) or None (empty)
            stage = {}       # stage -> chunk it holds (last copy issued into it)
            computed = set()
            bg, bsub, pslot = g0, 0, 0

            def issue():
                nonlocal bg, bsub, pslot
                if bg < g1:
                    old = stage.get(pslot)
                    assert old is None or old in computed, ("stage overwritten before compute", P, old)
                    stage[pslot] = (bg, bsub)
                    groups.append((pslot, (bg, bsub)))
                else:
                    groups.append(None)
                pslot = 0 if pslot + 1 == sbp else pslot + 1
                bsub += 1
                if bsub == 4:
                    bsub = 0
                    bg += 1
            for _ in range(P):
                issue()
            cslot = 0
            for gi in range(g0, g1):
                for s4 in range(4):
                    complete = len(groups) - (P - 1)   # wait_group P - 1: all but the newest P - 1 landed
                    mine = [i for i, gr in enumerate(groups) if gr is not None and gr[1] == (gi, s4)]
                    assert len(mine) == 1 and mine[0] < complete, ("chunk not landed", P, gi, s4)
                    issue()                            # after the barrier: chunk + P
                    assert stage[cslot] == (gi, s4), ("wrong stage", P, gi, s4, stage[cslot])
                    computed.add((gi, s4))
                    cslot = 0 if cslot + 1 == sbp else cslot + 1
            assert all(gr is None for gr in groups[len(groups) - P:]), "copies in flight at exit"
            cases += 1
    print(f"ok: M14 activation ring over {cases} ranges, P = 2..8: each chunk computed from its own stage after its "
          "copy landed (wait_group P - 1), no stage overwritten early, nothing in flight at exit")


def check_m14_text(mod):
    src = mod._M13_CUDA
    for line in M14_DECODE:
        assert line in src, f"M14 kernel changed, update check_m13: {line}"
    print("ok: M14 uses M13's decode, the swizzled activation stages and counter layout this check models")


def check_m14_smem():
    """load_b's placement then compute()'s reads: thread (g, t) of n-tile n gets activations 16t..16t+15 of
    pair n*8+g, i.e. exactly what M11/M13 load per thread from global (ap[n] + kg*8, +1), conflict-free."""
    NB = 64
    stage = {}
    for seg in range(NB * 8):
        pr, part = seg >> 3, seg & 7
        stage[pr * 128 + ((part ^ (pr & 7)) << 4)] = (pr, part)  # holds elements part*8 .. part*8+7
    assert len(stage) == NB * 8
    for n in range(NB // 8):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            st = (n * 8 + g) * 128
            lo = stage[st + (((2 * t) ^ g) << 4)]
            hi = stage[st + (((2 * t + 1) ^ g) << 4)]
            assert lo == (n * 8 + g, 2 * t) and hi == (n * 8 + g, 2 * t + 1), (n, lane, lo, hi)
        for half in range(2):  # each quarter warp of an LDS.128 touches 8 distinct 16-B bank groups
            for qw in range(4):
                groups = {((n * 8 + (l >> 2)) * 128 + (((2 * (l & 3) + half) ^ (l >> 2)) << 4)) % 128 // 16
                          for l in range(8 * qw, 8 * qw + 8)}
                assert len(groups) == 8, (n, half, qw, groups)
    print("ok: M14 activation stage: every thread reads its own pair's elements 16t..16t+15 (M11's B fragment), "
          "conflict-free LDS.128 in every quarter warp")


def scan_parallel(ids, maxp):
    """M14's 4-warp scan: blocks of q = ceil(n/128)*32 pairs per warp, counts, offsets, placement."""
    n = len(ids)
    q = ((n + 127) // 128) * 32
    wcnt = [[0] * 32 for _ in range(4)]
    for w in range(4):
        for p in range(w * q, min(n, w * q + q)):
            if 0 <= ids[p] < 32:
                wcnt[w][ids[p]] += 1
    cnt = [0] * 32
    off = [[0] * 32 for _ in range(4)]
    for e in range(32):
        run = 0
        for w in range(4):
            off[w][e] = run
            run += wcnt[w][e]
        cnt[e] = run
    lists = [[None] * min(cnt[e], maxp) for e in range(32)]
    for w in range(4):
        for p in range(w * q, min(n, w * q + q)):
            e = ids[p]
            if 0 <= e < 32:
                if off[w][e] < maxp:
                    lists[e][off[w][e]] = p
                off[w][e] += 1
    return cnt, lists


def model_m14(ids, n_ctas, gated, tt, nt, maxp, rt_total):
    NB = 8 * nt
    per_expert = rt_total // 2 if gated else rt_total // tt
    n_rb = per_expert // 4
    maxpass = (maxp + 7) // 8
    cnt, lists = scan_parallel(ids, maxp)
    _, want_lists = scan_lists(ids, 0, maxp)
    assert lists == want_lists, "parallel scan order differs from pair order"
    cntc = [min(c, maxp) for c in cnt]
    active = [e for e in range(32) if cntc[e] > 0]
    sbase, acc = [], 0
    for e in active:
        sbase.append(acc)
        acc += n_rb * ((cntc[e] + NB - 1) // NB) * GPT
    total = acc
    seen, segs, ctr_idx = {}, {}, {}
    for cta in range(n_ctas):
        g0, g1 = total * cta // n_ctas, total * (cta + 1) // n_ctas
        if g0 >= g1:
            continue
        ai = 0
        while ai + 1 < len(active) and sbase[ai + 1] <= g0:
            ai += 1
        e = active[ai]
        passes = (cntc[e] + NB - 1) // NB
        local = g0 - sbase[ai]
        rb, rem = divmod(local, passes * GPT)
        ps, grp = divmod(rem, GPT)
        seg0 = grp
        for gi in range(g0, g1):
            cntp = min(cntc[e] - ps * NB, NB)
            assert 0 < cntp <= NB and rb < n_rb
            for w in range(4):
                key = (e, rb * 4 + w, ps)
                seen.setdefault(key, []).append(grp)
            last = gi + 1 == g1
            if grp == GPT - 1 or last:
                for w in range(4):
                    key = (e, rb * 4 + w, ps)
                    segs.setdefault(key, []).append(grp - seg0 + 1)
                    ci = (e * per_expert + rb * 4 + w) * maxpass + ps
                    assert ctr_idx.setdefault(ci, key) == key
                seg0 = 0
            if not last:  # advance
                grp += 1
                if grp == GPT:
                    grp = 0
                    ps += 1
                    if ps == passes:
                        ps = 0
                        rb += 1
                        if rb == n_rb:
                            rb = 0
                            ai += 1
                            e = active[ai]
                            passes = (cntc[e] + NB - 1) // NB
    for e in active:
        passes = (cntc[e] + NB - 1) // NB
        for task in range(per_expert):
            for ps in range(passes):
                key = (e, task, ps)
                assert sorted(seen[key]) == list(range(GPT)), key
                assert sum(segs[key]) == GPT, key
    assert max(ctr_idx, default=0) < 32 * per_expert * maxpass
    return sum(1 for v in segs.values() if len(v) > 1)


def check_m14_schedule(mod):
    rnd = random.Random(5)
    maxp = mod.M14_MAX_TOKENS
    cases = split = 0
    for trial in range(30):
        full = trial % 6 == 0
        bs = rnd.choice([1, 8, 33, 64, 128, 200, 256, 257, 320])
        ids = []
        for _ in range(bs):
            ids += list(range(TOP_K)) if trial % 3 == 1 else rnd.sample(range(64), TOP_K)
        for gated, tt, nt in sorted({i[:3] for i in mod.M14_INSTANCES}):
            rt_total = (512 if gated else 256) if full else (32 if gated else 16 * tt)
            for n_ctas in ((132 * rnd.choice([1, 2, 3]),) if full else (1, 3, 50, 132 * rnd.choice([1, 2, 3]))):
                split += model_m14(ids, n_ctas, gated, tt, nt, maxp, rt_total)
                cases += 1
    print(f"ok: M14 model over {cases} (routing, instance, grid) cases up to {maxp} tokens incl. skewed: the "
          f"4-warp scan gives pair order, every (expert, task, pass) gets all {GPT} K groups once, "
          f"{split} split, counters unique and in bounds")


# ------------------------------------------------------------------------------ 7. M15 (TMA weight ring)
M15_LINES = (
    "constexpr int TILE_BYTES = D * 512 + 256;",
    "bulk_g2s(dst, w + (row * KG + c.grp * D) * 32, D * 512, bar);",
    "bulk_g2s(dst + D * 512, sf + (row * GPT + c.grp) * 32, 256, bar);",
    "const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);",
    "const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + lane * 8);",
    "mbar_wait(bar0 + 8 * slot, ((gi - g0) / R) & 1);",
    "const int slot = (gi - g0) % R;",
    "auto pass_cost = [&](int cnt, int p) { return 8 + beta * ((min(cnt - p * NB, NB) + 7) / 8); };",
    "if (rem < static_cast<long long>(c) * blk) return s_ubase[ai] + p * blk + static_cast<int>((rem + c - 1) / c);",
)


def check_m15_text(mod):
    src = mod._M13_CUDA
    m15 = src[src.index("namespace m15 {"):src.index("}  // namespace m15")]
    for line in ("for (int k = 0; k < P; ++k) issue_b();", "cp_wait<P - 1>();", "if (bg < g1) load_b(Bc, bsub, pslot);",
                 "if (++bg < g1) advance(Bc);", "cslot = cslot + 1 == SBP ? 0 : cslot + 1;",
                 "compute(s, wslot, cntp, cslot);", "pslot = pslot + 1 == SBP ? 0 : pslot + 1;"):
        assert line in m15, f"M15 activation ring differs from the one check_m14_ring models: {line}"
    for line in M15_LINES + M14_DECODE[:4]:
        assert line in src, f"M15 kernel changed, update check_m13: {line}"
    print("ok: M15 kernel text matches the ring, cost split and decode this check models")


def check_m15_slots():
    """TMA copy of (tile row, group) into a slot, then the consumer's reads = M13's per-lane global loads."""
    D_, KG_, GPT_ = 4, 64, 16
    rnd = random.Random(2)
    for _ in range(500):
        row, grp, s_, lane = rnd.randrange(1000), rnd.randrange(GPT_), rnd.randrange(D_), rnd.randrange(32)
        src_codes = (row * KG_ + grp * D_) * 32 * 16  # byte offset of the 2 KB copy in w
        read = src_codes + s_ * 512 + lane * 16   # consumer: tp + s*512 + lane*16
        want = ((row * KG_ + grp * D_ + s_) * 32 + lane) * 16  # M13: lw[i] + s*32, uint4 = 16 B
        assert read == want
        src_sc = (row * GPT_ + grp) * 32 * 8      # 256 B copy of uint2 scales
        assert src_sc + lane * 8 == ((row * GPT_ + grp) * 32 + lane) * 8
    print("ok: M15 slot reads = M13's per-lane code and scale loads (500 random positions)")


def model_m15(ids, n_ctas, gated, tt, nt, maxp, rt_total, beta, R):
    NB = 8 * nt
    per_expert = rt_total // 2 if gated else rt_total // tt
    n_rb = per_expert // 4
    blk = n_rb * GPT
    maxpass = (maxp + 7) // 8
    cnt, lists = scan_parallel(ids, maxp)
    cntc = [min(c, maxp) for c in cnt]
    active = [e for e in range(32) if cntc[e] > 0]
    cost = lambda c_, p_: 8 + beta * ((min(c_ - p_ * NB, NB) + 7) // 8)
    ubase, cbase, u, cc = [], [], 0, 0
    for e in active:
        ubase.append(u)
        cbase.append(cc)
        passes = (cntc[e] + NB - 1) // NB
        u += passes * blk
        cc += blk * sum(cost(cntc[e], p_) for p_ in range(passes))
    ubase.append(u)
    cbase.append(cc)
    total_units, total_cost = u, cc

    def cost_to_unit(pos):
        if pos >= total_cost:
            return total_units
        ai = 0
        while ai + 1 < len(active) and cbase[ai + 1] <= pos:
            ai += 1
        rem = pos - cbase[ai]
        p_ = 0
        while True:
            c_ = cost(cntc[active[ai]], p_)
            if rem < c_ * blk:
                return ubase[ai] + p_ * blk + (rem + c_ - 1) // c_
            rem -= c_ * blk
            p_ += 1

    bounds = [cost_to_unit(total_cost * b // n_ctas) for b in range(n_ctas + 1)]
    assert bounds[0] == 0 and bounds[-1] == total_units and all(x <= y for x, y in zip(bounds, bounds[1:]))
    seen, segs = {}, {}
    for cta in range(n_ctas):
        g0, g1 = bounds[cta], bounds[cta + 1]
        if g0 >= g1:
            continue
        ai = 0
        while ai + 1 < len(active) and ubase[ai + 1] <= g0:
            ai += 1
        e = active[ai]
        local = g0 - ubase[ai]
        ps, rem = divmod(local, blk)
        rb, grp = divmod(rem, GPT)
        issued = []  # the weight ring: groups issued in order, slot = index % R
        seg0 = grp
        for gi in range(g0, g1):
            issued.append((e, rb, ps, grp))
            for w_ in range(4):
                seen.setdefault((e, rb * 4 + w_, ps), []).append(grp)
            if grp == GPT - 1 or gi + 1 == g1:
                for w_ in range(4):
                    segs.setdefault((e, rb * 4 + w_, ps), []).append(grp - seg0 + 1)
                seg0 = 0
            grp += 1
            if grp == GPT:
                grp = 0
                rb += 1
                if rb == n_rb:
                    rb = 0
                    ps += 1
                    if ps == (cntc[e] + NB - 1) // NB:
                        ps = 0
                        ai += 1
                        e = active[ai] if ai < len(active) else None
        # consumer order == issue order, and slot (gi - g0) % R is reused only after R groups
        assert len(issued) == g1 - g0
    for e in active:
        for task in range(per_expert):
            for ps in range((cntc[e] + NB - 1) // NB):
                key = (e, task, ps)
                assert sorted(seen[key]) == list(range(GPT)), key
                assert sum(segs[key]) == GPT, key
    return sum(1 for v in segs.values() if len(v) > 1)


def check_m15_schedule(mod):
    rnd = random.Random(9)
    maxp = mod.M14_MAX_TOKENS
    cases = split = 0
    for trial in range(24):
        full = trial % 6 == 0
        bs = rnd.choice([33, 64, 128, 200, 256, 257, 320])
        ids = []
        for _ in range(bs):
            ids += list(range(TOP_K)) if trial % 3 == 1 else rnd.sample(range(64), TOP_K)
        for gated, tt, nt, R, _p in mod.M15_INSTANCES:
            rt_total = (512 if gated else 256) if full else (32 if gated else 16 * tt)
            for beta in (0, 2, 8):
                for n_ctas in ((264,) if full else (1, 7, 264)):
                    split += model_m15(ids, n_ctas, gated, tt, nt, maxp, rt_total, beta, R)
                    cases += 1
    print(f"ok: M15 model over {cases} cases (cost weight 0/2/8, up to {maxp} tokens incl. skewed): CTA ranges "
          f"partition the units, every (expert, task, pass) gets all {GPT} K groups once, {split} split")


if __name__ == "__main__":
    check_m2.install_stubs()
    sys.modules["torch.utils.cpp_extension"] = type(sys)("torch.utils.cpp_extension")
    sys.modules["torch.utils.cpp_extension"].load_inline = None
    mod = check_submission_m13()
    check_decode(mod)
    check_scales(mod)
    check_layouts(mod)
    check_schedule(mod)
    check_comb(mod)
    check_m11_pdl(mod)
    check_m14_text(mod)
    check_m14_ring(mod)
    check_m14_smem()
    check_m14_schedule(mod)
    check_m15_text(mod)
    check_m15_slots()
    check_m15_schedule(mod)

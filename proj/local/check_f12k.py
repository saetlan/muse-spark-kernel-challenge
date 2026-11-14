"""Exactness of F12K (P43, stream-K F12M) checked on the CUDA text of proj/f12m/f12k.py: its index helpers and decode are
parsed out of the source and executed (check_f12v's translator), and the kernel's control flow (which the TEXT lines pin)
is replayed in Python. Checks, for every lab shape and grid:
  1. partition: the CTAs' unit ranges [f12k_u0(b), f12k_u0(b + 1)) cover every (tile, chunk) unit exactly once;
     f12k_owner(u) is the CTA holding unit u; pieces per tile from the kernel's formula = the host mirror f12k_pieces =
     the wrapper's C++ loop; piece indices of a tile are 0 .. pieces - 1, one per CTA; partial slots are distinct and in
     bounds;
  2. segments and ring: per CTA the segments (first at u0's chunk, then chunk 0 of each next tile) and per warp the
     chunks f12k_wchunk(c0, ks, i) for i < f12k_nw(nseg, ks) cover the CTA's units exactly once; the register ring hands
     every step its own chunk (prologue, refills, next segment's prologue after the loop);
  3. math: the kernel replayed in float64 (decoded planes through the parsed decode, per-warp chunk sums, KSW parts in
     order, piece slots, the counter protocol in a random CTA arrival order, finalize with scale and escapes) equals
     FP64 x @ W.T (relative error < 1e-12) on normal inputs, exactly on probe inputs (escape columns included), every
     output written once, every counter back at 0.
  --mutate: plants bugs in the kernel text and requires each to be caught.

Run: .venv/bin/python proj/local/check_f12k.py [--mutate]
"""

import math
import random
import sys
from pathlib import Path

import torch

LOCAL = Path(__file__).resolve().parent
PROJ = LOCAL.parent
sys.path.insert(0, str(LOCAL))
for sub in ("f12v", "f12m", "p9"):
    sys.path.insert(0, str(PROJ / sub))
import check_f12m as cm  # noqa: E402
import check_f12v as cv  # noqa: E402
import f12k  # noqa: E402
from f12m_layout import f12m_device_planes  # noqa: E402
from p9_layout import k_perm  # noqa: E402

TEXT = [
    "const long long u0 = f12k_u0(b, units, grid), u1 = f12k_u0(b + 1, units, grid);",
    "long long tile = u0 / chs;", "int c0 = static_cast<int>(u0 - tile * chs);",
    "const long long cend0 = u1 - tile * chs;", "int nseg = static_cast<int>((cend0 < chs ? cend0 : chs) - c0);",
    "int nw = f12k_nw(nseg, ks, KSW);",
    "loadw(s, tile, f12k_wchunk(c0, ks, s, KSW));", "loadx(s, f12k_wchunk(c0, ks, s, KSW));",
    "for (int s0 = 0; s0 < nw; s0 += D) {", "for (int slot = 0; slot < D; ++slot) {", "const int s = s0 + slot;",
    "if (s < nw) {", "if (s + D < nw) {", "loadw(slot, tile, f12k_wchunk(c0, ks, s + D, KSW));",
    "loadx(slot, f12k_wchunk(c0, ks, s + D, KSW));",
    "const long long ntile = tile + 1;", "const bool more = ntile * chs < u1;",
    "const long long cend = u1 - ntile * chs;", "nnseg = static_cast<int>(cend < chs ? cend : chs);",
    "nnw = f12k_nw(nnseg, ks, KSW);", "loadw(s, ntile, f12k_wchunk(0, ks, s, KSW));", "loadx(s, f12k_wchunk(0, ks, s, KSW));",
    "ld_x(rec + f12k_ridx(tile * 16 + tid, 0), rc0);", "ld_x(rec + f12k_ridx(tile * 16 + tid, 1), rc1);",
    "ld_x(rec + f12k_ridx(ntile * 16 + tid, 0), rn0);", "ld_x(rec + f12k_ridx(ntile * 16 + tid, 1), rn1);",
    "for (int j = 0; j < 3; ++j) ld_w(wp + f12k_widx(tl, chs, c, j, lane), wv[slot][j]);",
    "ld_x(x + f12k_xidx(8 * n + g, k8, c, t, 0), xv[slot][n][0]);", "ld_x(x + f12k_xidx(8 * n + g, k8, c, t, 1), xv[slot][n][1]);",
    "const uint4& lv = wv[slot][f12k_lslot(j)];",
    "const uint4 a = f12k_decode(word(wv[slot][0], j), word(lv, f12k_lword(j, 0)), word(lv, f12k_lword(j, 1)));",
    "const uint32_t b0 = word(xv[slot][n][j >> 1], (2 * j) & 3);", "const uint32_t b1 = word(xv[slot][n][j >> 1], (2 * j + 1) & 3);",
    "for (int r = 0; r < 4; ++r) part[ks][f12k_crow(g, r)][f12k_ctok(n, t, r)] = acc[n][r];",
    "const long long first = f12k_owner(tile * chs, units, grid);",
    "const int pieces = static_cast<int>(f12k_owner(tile * chs + chs - 1, units, grid) - first + 1);",
    "bool fin = pieces == 1;", "const int piece = static_cast<int>(b - first);",
    "for (int idx = tid; idx < 16 * mo; idx += NTH) {", "const int tok = idx / 16, row = idx % 16;",
    "for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];",
    "__stcg(slots + f12k_sidx(tile, piece, maxp, MT, tok, row), sum);",
    "const int old = atomicAdd(cnt + tile, 1);", "last_s = old == pieces - 1;", "fin = last_s != 0;",
    "for (int p = 0; p < pieces; ++p) sum += __ldcg(slots + f12k_sidx(tile, p, maxp, MT, tok, row));",
    "const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;", "sum *= f12k_scale(base);",
    "const float wdec = f12k_wdec(base);",
    "const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * k8) + (esc[j] & 0xFFFFu)]) << 16);",
    "sum = fmaf(-xk, wdec, sum);", "sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);",
    "out[static_cast<long long>(tok) * n_rows + tile * 16 + row] = __float2bfloat16_rn(sum);",
    "if (pieces > 1 && tid == 0) cnt[tile] = 0;",
    "tile = ntile;", "c0 = 0;", "nseg = nnseg;", "nw = nnw;", "rc0 = rn0;", "rc1 = rn1;",
    "const int k8 = 8 * chs;", "const long long n_rows = static_cast<long long>(n_tiles) * 16;",
    "const int mo = m < MT ? m : MT;", "constexpr int MT = 8 * NT;            // token slots",
    "const int tid = threadIdx.x, lane = tid & 31, ks = tid >> 5;", "const int g = lane >> 2, t = lane & 3;",
    # HL (router hi + lo) path
    "for (int h = 0; h < 2; ++h) ld_w(wp + f12k_hidx(tl, 4 * chs, c, j, h, lane), wv[slot][2 * j + h]);",
    "mma(acc[n], wv[slot][2 * j], b0, b1);       // hi", "mma(acc[n], wv[slot][2 * j + 1], b0, b1);   // lo",
    "outf[static_cast<long long>(tok) * n_rows + tile * 16 + row] = sum;", "if (!HL && tid < 16) {",
]
WRAPPER = "const int64_t a = ((tl * chs + 1) * grid - 1) / units, z = ((tl * chs + chs) * grid - 1) / units;"
NAMES = ("u0", "owner", "wchunk", "nw", "widx", "hidx", "xidx", "lslot", "lword", "crow", "ctok", "ridx", "sidx", "decode",
         "scale", "wdec")
# (layer, n_rows, K, m values, grids, (NT, KSW, D) instances): the lab's shapes and grids
CASES = [
    ("qkv_gate", 2304, 8192, (1, 5, 8), (132, 256, 264, 396, 528), ((1, 8, 4), (1, 8, 2), (1, 4, 4))),
    ("qkv_gate", 2304, 8192, (13, 16), (256, 264, 396), ((2, 8, 2), (2, 4, 2), (2, 8, 4))),
    ("unpooler", 1024, 4096, (1, 8), (128, 256, 264, 512, 528), ((1, 8, 2), (1, 4, 2), (1, 8, 4))),
    ("router", 256, 8192, (1, 8, 16, 32), (128, 256, 264, 512), ((1, 8, 2, True), (2, 8, 2, True), (4, 8, 1, True), (2, 4, 2, True))),
]


def helpers(src):
    cv.SRC = src
    for stmt in TEXT:
        assert src.count(stmt) >= 1, f"kernel text changed, update the emulation: {stmt}"
    return {n: cv.parse("f12k_" + n) for n in NAMES}


def check_same_as_f12m(h):
    """F12K's per-(tile, chunk) math is F12M's (RTW = RG = TG = 1, tb = 0): its helpers equal F12M's parsed ones, so
    check_f12m's lane-level emulation covers the fragment and x indexing; chunk c reads natural k [64 c, 64 c + 64)."""
    import f12m
    cv.SRC = f12m.F12M_SRC
    fm = {n: cv.parse("f12m_" + n) for n in ("widx", "xidx", "lslot", "lword", "crow", "ctok", "ridx", "decode", "scale", "wdec")}
    cv.SRC = f12k.F12K_SRC
    g = torch.Generator().manual_seed(3)
    tl, c, sl, ln = (torch.randint(0, 2000, (4096,), generator=g), torch.randint(0, 128, (4096,), generator=g),
                     torch.randint(0, 3, (4096,), generator=g), torch.randint(0, 32, (4096,), generator=g))
    assert torch.equal(h["widx"](tl, 128, c, sl, ln), fm["widx"](tl, 128, c, sl, ln))
    tk, tt, hh = torch.randint(0, 32, (4096,), generator=g), torch.randint(0, 4, (4096,), generator=g), torch.randint(0, 2, (4096,), generator=g)
    assert torch.equal(h["xidx"](tk, 1024, c, tt, hh), fm["xidx"](tk, 1024, c, tt, hh))
    for j in range(4):
        assert h["lslot"](j) == fm["lslot"](j) and all(h["lword"](j, x) == fm["lword"](j, x) for x in (0, 1))
    for gg in range(8):
        for r in range(4):
            assert h["crow"](gg, r) == fm["crow"](0, 1, 0, gg, r)
            for n in range(4):
                for t in range(4):
                    assert h["ctok"](n, t, r) == fm["ctok"](n, t, r)
    assert torch.equal(h["ridx"](tl, hh), fm["ridx"](tl, hh))
    q, l0, l1 = (torch.randint(0, 2**32, (8192,), generator=g) for _ in range(3))
    assert all(torch.equal(a, b) for a, b in zip(h["decode"](q, l0, l1), fm["decode"](q, l0, l1)))
    for base in range(0, 256):
        assert h["scale"](base) == fm["scale"](base) and h["wdec"](base) == fm["wdec"](base)
    for k in (1024, 4096, 8192):
        kp = k_perm(k)
        assert torch.equal(kp.view(-1, 64).sort(dim=1).values, torch.arange(k).view(-1, 64)), "k_perm crosses chunks"
    print("ok: F12K helpers = F12M's (indexing, decode, scale); k_perm stays inside each 64-k chunk")


def check_partition(h, n_tiles, chs, grid):
    units = n_tiles * chs
    u0 = [int(h["u0"](b, units, grid)) for b in range(grid + 1)]
    assert u0[0] == 0 and u0[grid] == units, (n_tiles, chs, grid, "range ends")
    assert all(u0[b] < u0[b + 1] for b in range(grid)), (n_tiles, chs, grid, "empty CTA range")
    holder = torch.empty(units, dtype=torch.int64)
    for b in range(grid):
        holder[u0[b]:u0[b + 1]] = b
    us = torch.arange(units)
    own = h["owner"](us, units, grid)
    assert torch.equal(own, holder), (n_tiles, chs, grid, "owner != holder")
    pieces = [int(h["owner"](t * chs + chs - 1, units, grid) - h["owner"](t * chs, units, grid) + 1) for t in range(n_tiles)]
    assert pieces == f12k.f12k_pieces(n_tiles, chs, grid), (n_tiles, chs, grid, "host mirror")
    cpp = [((t * chs + chs) * grid - 1) // units - ((t * chs + 1) * grid - 1) // units + 1 for t in range(n_tiles)]
    assert cpp == pieces, (n_tiles, chs, grid, "wrapper loop")
    for t in range(n_tiles):
        ctas = sorted(set(holder[t * chs:(t + 1) * chs].tolist()))
        first = int(h["owner"](t * chs, units, grid))
        assert [b - first for b in ctas] == list(range(pieces[t])), (n_tiles, chs, grid, t, "piece indices")
    return u0, pieces


def segments(h, u0, u1, chs, ksw):
    """The kernel's segment walk: [(tile, c0, nseg)], and per warp the chunk list of each segment."""
    tile = u0 // chs
    c0 = u0 - tile * chs
    cend0 = u1 - tile * chs
    nseg = min(cend0, chs) - c0
    segs = [(tile, c0, nseg)]
    while True:
        ntile = tile + 1
        if not ntile * chs < u1:
            break
        cend = u1 - ntile * chs
        nseg = min(cend, chs)
        segs.append((ntile, 0, nseg))
        tile = ntile
    out = []
    for tl, c0, nseg in segs:
        per = []
        for ks in range(ksw):
            nw = int(h["nw"](nseg, ks, ksw))
            per.append([int(h["wchunk"](c0, ks, i, ksw)) for i in range(nw)])
        out.append((tl, c0, nseg, per))
    return out


def check_ring(nw, nnw, d):
    """One segment's ring: prologue loads steps 0..D-1 (< nw), the loop refills; then the next segment's prologue."""
    ring = {s: s for s in range(d) if s < nw}
    for s0 in range(0, nw, d):
        for slot in range(d):
            s = s0 + slot
            if s < nw:
                assert ring.get(slot) == s, ("ring hands a step another chunk", s, nw, d)
                if s + d < nw:
                    ring[slot] = s + d
    nxt = {s: s for s in range(d) if s < nnw}
    return nxt


def raw_weights(planes, k, dec):
    """FP64 decoded planes [N, K] before scale and escapes (escape slots decode to 2^-15), natural k order."""
    p = cm.u32(planes["p"].to(torch.int64))
    rt = p.shape[0]
    n = 16 * rt
    regs = cm.regs_of(p, dec)
    el = torch.stack([(regs[..., e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    v = el.view(rt, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
    wb = torch.empty_like(v)
    wb[:, k_perm(k)] = v
    return cm.bf16_bits_to_f64(wb)


def emulate(h, inst, planes, raw, x, n_rows, grid, seed):
    nt, ksw, d = inst[:3]
    hl = len(inst) > 3 and inst[3]
    mt = 8 * nt
    m, k = x.shape
    chs, n_tiles = k // 64, n_rows // 16
    units = n_tiles * chs
    assert m <= mt
    mo = min(m, mt)
    maxp = max(f12k.f12k_pieces(n_tiles, chs, grid))
    # weight / x reads of every warp and step through the parsed helpers: every slot of the covered planes once
    # per-chunk products (row, tok) = sum over the chunk's 64 k of decoded weight x x (FP64)
    xd = x.double()
    prod = torch.einsum("trck,mck->tcrm", raw[:n_rows].view(n_tiles, 16, chs, 64), xd.view(m, chs, 64))   # [T, CH, 16, m]
    slots = {}
    written = {}
    cnt = [0] * n_tiles
    out = torch.full((n_rows, m), float("nan"), dtype=torch.float64)
    rec4 = cm.u32(planes["rec"].view(-1, 4)) if not hl else torch.zeros(n_rows * 2, 4, dtype=torch.int64)
    xbits = cm.u32(x.contiguous().view(torch.int16).to(torch.int64)).view(m, k)
    covered = torch.zeros(units, dtype=torch.int64)
    recl = rec4.view(-1, 8).tolist()
    xbl = xbits.tolist()
    order = list(range(grid))
    random.Random(seed).shuffle(order)      # CTA arrival order: any interleaving the hardware may produce
    for b in order:
        u0, u1 = int(h["u0"](b, units, grid)), int(h["u0"](b + 1, units, grid))
        segs = segments(h, u0, u1, chs, ksw)
        for si, (tl, c0, nseg, per) in enumerate(segs):
            nnw = [len(segs[si + 1][3][ks]) for ks in range(ksw)] if si + 1 < len(segs) else [0] * ksw
            part = torch.zeros(ksw, 16, mt, dtype=torch.float64)
            for ks in range(ksw):
                check_ring(len(per[ks]), nnw[ks], d)
                for c in per[ks]:
                    assert c0 <= c < c0 + nseg, ("chunk outside its segment", b, tl, c)
                    covered[tl * chs + c] += 1
                    part[ks, :, :m] += prod[tl, c]
            first = int(h["owner"](tl * chs, units, grid))
            pieces = int(h["owner"](tl * chs + chs - 1, units, grid) - first + 1)
            psum = torch.zeros(16, mt, dtype=torch.float64)
            for p in range(ksw):                     # KSW parts in order
                psum += part[p]
            psl = psum.tolist()
            fin = pieces == 1
            if not fin:
                piece = b - first
                for idx in range(16 * mo):
                    tok, row = idx // 16, idx % 16
                    si_ = int(h["sidx"](tl, piece, maxp, mt, tok, row))
                    assert 0 <= si_ < n_tiles * maxp * mt * 16 and si_ not in slots, ("slot out of range or reused", si_)
                    slots[si_] = psl[row][tok]
                old = cnt[tl]
                cnt[tl] += 1
                fin = old == pieces - 1
            if fin:
                for idx in range(16 * mo):
                    tok, row = idx // 16, idx % 16
                    if pieces == 1:
                        v = psl[row][tok]
                    else:
                        v = 0.0
                        for p in range(pieces):
                            si_ = int(h["sidx"](tl, p, maxp, mt, tok, row))
                            assert si_ in slots, ("finalize before every piece arrived", tl, p)
                            v += slots[si_]
                    grow = tl * 16 + row
                    if hl:                           # outf[...] = sum; continue
                        assert (grow, tok) not in written, ("output written twice", grow, tok)
                        written[(grow, tok)] = v
                        continue
                    r8 = recl[grow]
                    assert int(h["ridx"](grow, 1)) == 2 * grow + 1
                    q0, q1 = r8[:4], r8[4:]
                    base, count = int(q0[0]) & 0xFF, (int(q0[0]) >> 8) & 0xFF
                    v *= cv.struct_f32(h["scale"](base))
                    wdec = cv.struct_f32(h["wdec"](base))
                    esc = [q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]]
                    for jj in range(7):
                        if jj < count:
                            kk = int(esc[jj]) & 0xFFFF
                            xk = float(cm.bf16_bits_to_f64(torch.tensor(xbl[tok][kk])))
                            v = v - xk * wdec + xk * cv.struct_f32(int(esc[jj]) & 0xFFFF0000)
                    assert (grow, tok) not in written, ("output written twice", grow, tok)
                    written[(grow, tok)] = v
                if pieces > 1:
                    cnt[tl] = 0
    assert torch.equal(covered, torch.ones(units, dtype=torch.int64)), ("units not covered exactly once", inst, grid)
    assert all(c == 0 for c in cnt), ("counters not back at 0", inst, grid)
    assert len(written) == n_rows * m, ("outputs never written", inst, grid, len(written))
    for (grow, tok), v in written.items():
        out[grow, tok] = v
    return out


PLANES = {}


def check_hidx(h):
    """f12k_hidx addresses p9_pack_hilo's [N/16, K/16, 2, 32, 8] BF16 tensor (8 BF16 = one uint4) exactly."""
    import p9_layout
    n, k = 256, 8192
    w = torch.randn(n, k, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    hilo = p9_layout.p9_pack_hilo(w)
    assert hilo.shape == (n // 16, k // 16, 2, 32, 8)
    flat = torch.arange(hilo.numel() // 8).view(n // 16, k // 16, 2, 32)
    tl = torch.arange(n // 16).view(-1, 1, 1, 1, 1)
    c = torch.arange(k // 64).view(1, -1, 1, 1, 1)
    j = torch.arange(4).view(1, 1, -1, 1, 1)
    hh = torch.arange(2).view(1, 1, 1, -1, 1)
    ln = torch.arange(32).view(1, 1, 1, 1, -1)
    got = h["hidx"](tl, 4 * (k // 64), c, j, hh, ln)
    want = flat.view(n // 16, k // 64, 4, 2, 32)
    assert torch.equal(got, want), "f12k_hidx != p9_pack_hilo layout"
    hi, lo = p9_layout.p9_split_router(w)
    assert torch.equal(hilo[:, :, 0], p9_layout.p9_pack(hi)) and torch.equal(hilo[:, :, 1], p9_layout.p9_pack(lo))
    print("ok: f12k_hidx = p9_pack_hilo's layout (hi then lo per k16 step); fragments = p9_pack(hi) / p9_pack(lo)")
    return w, hi, lo


def check_math(h, quick=False):
    torch.manual_seed(4)
    planes_cache = PLANES
    for layer, n_rows, k, ms, grids, insts in CASES:
        if layer == "router":
            if (layer, k) not in planes_cache:
                w, hi, lo = check_hidx(h)
                planes_cache[(layer, k)] = (None, None, hi.double() + lo.double())
            _, planes, raw = planes_cache[(layer, k)]
            wd = raw
        elif (layer, k) not in planes_cache:
            n_all = {"qkv_gate": 2304, "unpooler": 8192}[layer]
            w = (torch.randn(n_all, k) / math.sqrt(k)).to(torch.bfloat16)
            w[0, :5] = 0
            w[1, 3] = 1e4
            w[2, 77] = 1e-30
            w[17, 77] = 0
            w[1000, k - 1] = -3e3
            planes = f12m_device_planes(w)
            assert planes is not None, layer
            planes_cache[(layer, k)] = (w, planes, raw_weights(planes, k, h["decode"]))
        if layer != "router":
            w, planes, raw = planes_cache[(layer, k)]
            wd = w[:n_rows].double()
        for grid in grids:
            check_partition(h, n_rows // 16, k // 64, grid)
        # every grid once (instances and token counts rotate), then every instance at two grids
        runs = [(ms[i % len(ms)], g, insts[i % len(insts)]) for i, g in enumerate(grids)]
        runs += [(ms[-1 - i % len(ms)], grids[(i + 1) % len(grids)], inst) for i, inst in enumerate(insts)]
        if quick:
            runs = runs[:2] + runs[-1:]
        for m, grid, inst in runs:
            if m > 8 * inst[0]:
                continue
            gen = torch.Generator().manual_seed(m * 7 + grid)
            x_norm = torch.randn(m, k, generator=gen).to(torch.bfloat16)
            x_probe = torch.zeros(m, k, dtype=torch.bfloat16)
            x_probe[0, 77] = -32.0
            x_probe[m - 1, 3] = 0.5
            x_probe[m - 1, k - 1] = 2.0
            for x, exact in ((x_norm, False), (x_probe, True)):
                got = emulate(h, inst, planes, raw, x, n_rows, grid, seed=grid + m)
                ref = (x.double() @ wd.T).T
                if exact:
                    assert torch.equal(got, ref), (layer, m, grid, inst, "probe not exact")
                else:
                    scale = wd.abs() @ x.double().abs().T
                    err = float(((got - ref).abs() / scale.clamp(min=1e-30)).max())
                    assert err < 1e-12, (layer, m, grid, inst, err)
        print(f"ok: {layer:8s} n{n_rows} K{k}: partition, segments, ring, fix-up and math exact for "
              f"{len(runs)} (m, grid, NT/KSW/D) runs; grids {', '.join(map(str, grids))}")


MUTATIONS = [
    ("return ((unit + 1) * grid - 1) / units;", "return (unit * grid) / units;", "owner formula"),
    ("return (nseg + ksw - 1 - ks) / ksw;", "return (nseg - ks) / ksw;", "warp step count"),
    ("return c0 + ks + ksw * i;", "return ks + ksw * i;", "segment start ignored"),
    ("return ((tile * maxp + piece) * mt + tok) * 16 + row;", "return (tile * maxp + piece) * mt * 16 + row;", "slot index"),
    ("return b * units / grid;", "return b * (units / grid);", "range split"),
    ("const int pieces = static_cast<int>(f12k_owner(tile * chs + chs - 1, units, grid) - first + 1);",
     "const int pieces = static_cast<int>(f12k_owner(tile * chs + chs - 1, units, grid) - first);", "piece count"),
]


def mutate():
    base = f12k.F12K_SRC
    caught = 0
    for old, new, what in MUTATIONS:
        assert base.count(old) == 1, what
        src = base.replace(old, new)
        try:
            h = helpers(src)
            # the pieces line is replayed from TEXT, so for that mutation the TEXT pin itself must fail
            check_math(h, quick=True)
        except (AssertionError, ZeroDivisionError, KeyError, IndexError, RuntimeError) as exc:
            caught += 1
            print(f"  caught: {what} ({type(exc).__name__})")
            continue
        print(f"  MISSED: {what}")
    helpers(base)
    assert caught == len(MUTATIONS), f"{len(MUTATIONS) - caught} mutations missed"
    print(f"ok: {caught}/{len(MUTATIONS)} planted bugs caught")


def main():
    h = helpers(f12k.F12K_SRC)
    assert WRAPPER in f12k.f12k_source(((1, 8, 4),)), "wrapper's pieces loop changed: update check_partition"
    check_same_as_f12m(h)
    check_math(h)
    if "--mutate" in sys.argv:
        mutate()


if __name__ == "__main__":
    main()

"""F12M configs: candidates per (BF16 layer, m in 8/16/32) and the instance lists compiled into a module.

A config is dict(RTW, RG, KSW, D, F12); its instance is (NT, RTW, RG, KSW, NS, D, F12) with NT = m / 8
and NS = K / 64 / KSW. Constraints (static_asserts and launcher checks): rows % (16 RG RTW) == 0,
D | NS, 64-512 threads, static shared memory <= 48 KB. Ranking (a guess; the lab times them): bytes in
flight per SM from estimated registers, wave balance, and how much of each warp's traffic is weights
rather than x re-reads.
"""

import math

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
MS = (8, 16, 32)


def n_rows(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else SHAPES[layer][0]


def smem(nt, rtw, rg, ksw, f12):
    rb = 16 * rg * rtw
    return ksw * rb * 8 * nt * 4 + (2 * rb if f12 else 1) * 16


def regs_est(nt, rtw, d, f12):
    sl = 3 if f12 else 4
    return d * (4 * sl * rtw + 8 * nt) + 4 * rtw * nt + 4 * rtw + 34


def instance(cfg, layer, m):
    """(NT, RTW, RG, KSW, NS, D, F12) for CS = 1 and XD = D (the original 7-tuple); CS appended when CS > 1 or
    XD != D, and XD appended when XD != D."""
    k = SHAPES[layer][1]
    cs, xd, tg = cfg.get("CS", 1), cfg.get("XD", cfg["D"]), cfg.get("TG", 1)
    inst = (cfg.get("NT", max(1, m // 8)), cfg["RTW"], cfg["RG"], cfg["KSW"], k // 64 // cfg["KSW"] // cs, cfg["D"], cfg["F12"])
    if tg > 1:
        return inst + (cs, xd, tg)
    if xd != cfg["D"]:
        return inst + (cs, xd)
    return inst + ((cs,) if cs > 1 else ())


def ranked(layer, m, f12=True):
    n, k = n_rows(layer, m), SHAPES[layer][1]
    nt, chs = m // 8, k // 64
    out = []
    for rtw in (1, 2, 4):
        for rg in (1, 2, 4, 8):
            for ksw in (1, 2, 4, 8):
                threads = 32 * rg * ksw
                if chs % ksw or not 64 <= threads <= 512:
                    continue
                ns, rb = chs // ksw, 16 * rg * rtw
                if n % rb or n // rb < 64 or smem(nt, rtw, rg, ksw, f12) > 48 * 1024:
                    continue
                for d in (1, 2, 4):
                    regs = regs_est(nt, rtw, d, f12)
                    if ns % d or regs > (128 if threads > 256 else 168):
                        continue
                    regs_r = (regs + 7) // 8 * 8
                    per_sm = min(65536 // (regs_r * threads), 2048 // threads, 32,
                                 (227 * 1024) // smem(nt, rtw, rg, ksw, f12))
                    if per_sm < 1:
                        continue
                    ctas = n // rb
                    wbytes = (48 if f12 else 64) * rtw
                    inflight = per_sm * threads * d * wbytes
                    waves = ctas / (per_sm * 132)
                    balance = waves / math.ceil(waves)
                    share = wbytes / (wbytes + 32 * nt)          # weights vs x bytes per lane per chunk
                    score = min(inflight, 64 * 1024) * balance * math.sqrt(share)
                    pref = {256: 0, 128: 1, 512: 2}.get(threads, 3)
                    out.append(((-score, pref, ctas), dict(RTW=rtw, RG=rg, KSW=ksw, D=d, F12=f12)))
    out.sort(key=lambda e: e[0])
    return [e[1] for e in out]


def candidates(layer, m, max_n=5):
    """F12 candidates (variety in ring depth and tiles per warp), plus the best one's BF16 twin."""
    r = ranked(layer, m)
    picked = r[:max_n - 2]
    for key, vals in (("D", (1, 2, 4)), ("RTW", (1, 2, 4))):
        for v in vals:
            if len(picked) < max_n - 1 and not any(c[key] == v for c in picked):
                extra = next((c for c in r if c[key] == v), None)
                if extra:
                    picked.append(extra)
    picked += [c for c in r if c not in picked][:max(0, max_n - 1 - len(picked))]
    if picked:
        twin = dict(picked[0], F12=False)
        if twin in ranked(layer, m, f12=False):
            picked.append(twin)
    return picked


def lab_instances():
    insts = []
    for layer in SHAPES:
        for m in MS:
            for cfg in candidates(layer, m):
                inst = instance(cfg, layer, m)
                if inst not in insts:
                    insts.append(inst)
    return tuple(insts)


LAB = lab_instances()


# P36 / PE17: split-K across a cluster of CS CTAs for the shapes with too few 16-row tiles per SM (qkv 144
# tiles, the unpooler's 1024-row slice at m <= 8: 64 tiles) or short K (o: 16 chunks), plus pooler and the
# unpooler at m = 16 for comparison. Ring depth D bounds the registers (D 4 ~110, D 2 ~70 at NT 1), so D 2
# keeps 3 CTAs of 256 threads per SM (qkv CS 2 = 288 CTAs).
def _cs(nt, rtw, ksw, d, cs):
    return dict(NT=nt, RTW=rtw, RG=1, KSW=ksw, D=d, F12=True, CS=cs)


CS_CANDIDATES = {
    ("qkv_gate", 1): [_cs(1, 1, 8, 2, 2), _cs(1, 1, 8, 4, 2)],
    ("qkv_gate", 8): [_cs(1, 1, 8, 2, 2), _cs(1, 1, 8, 4, 2), _cs(1, 1, 4, 4, 2), _cs(1, 1, 8, 4, 4)],
    ("qkv_gate", 16): [_cs(2, 1, 8, 2, 2), _cs(2, 1, 8, 4, 2), _cs(2, 1, 4, 2, 2), _cs(2, 1, 8, 2, 4)],
    ("o", 1): [_cs(1, 1, 8, 1, 2), _cs(1, 2, 8, 1, 2), _cs(1, 1, 4, 2, 2)],
    ("o", 8): [_cs(1, 2, 8, 1, 2), _cs(1, 1, 8, 1, 2), _cs(1, 1, 4, 2, 2), _cs(1, 2, 4, 2, 2)],
    ("o", 16): [_cs(2, 2, 8, 1, 2), _cs(2, 1, 8, 1, 2), _cs(2, 1, 4, 2, 2), _cs(2, 2, 4, 2, 2)],
    ("unpooler", 1): [_cs(1, 1, 8, 4, 2), _cs(1, 1, 8, 2, 4), _cs(1, 1, 4, 2, 8)],
    ("unpooler", 8): [_cs(1, 1, 8, 4, 2), _cs(1, 1, 8, 2, 4), _cs(1, 1, 4, 2, 8), _cs(1, 1, 8, 1, 8)],
    ("unpooler", 16): [_cs(2, 1, 8, 2, 2), _cs(2, 2, 8, 1, 2)],
    ("pooler", 1): [_cs(1, 1, 8, 2, 2)],
    ("pooler", 8): [_cs(1, 1, 8, 2, 2), _cs(1, 1, 8, 4, 2)],
    ("pooler", 16): [_cs(2, 1, 8, 2, 2)],
}
CS_LAB = tuple(dict.fromkeys(instance(c, layer, m) for (layer, m), cands in CS_CANDIDATES.items() for c in cands))


# PE19: a shallower x ring (XD < D; x is L2-resident) so the weight ring can go deeper at the same register
# count: more weight bytes in flight per SM where PE17 showed the total in flight, not the split, sets the rate.
def _xd(nt, rtw, d, xd, ksw=8):
    return dict(NT=nt, RTW=rtw, RG=1, KSW=ksw, D=d, F12=True, XD=xd)


XD_CANDIDATES = {
    ("qkv_gate", 16): [_xd(2, 1, 4, 1), _xd(2, 1, 4, 2), _xd(2, 1, 8, 1), _xd(2, 1, 8, 2)],
    ("qkv_gate", 8): [_xd(1, 1, 4, 1), _xd(1, 1, 8, 1), _xd(1, 1, 8, 2)],
    ("pooler", 16): [_xd(2, 1, 4, 1), _xd(2, 1, 4, 2), _xd(2, 1, 8, 1)],
    ("pooler", 8): [_xd(1, 1, 4, 1), _xd(1, 1, 8, 1)],
    ("unpooler", 16): [_xd(2, 2, 2, 1), _xd(2, 2, 4, 1), _xd(2, 1, 4, 1), _xd(2, 1, 8, 1)],
    ("o", 8): [_xd(1, 2, 2, 1), _xd(1, 1, 2, 1)],
    ("o", 16): [_xd(2, 2, 2, 1), _xd(2, 1, 2, 1)],
}
XD_LAB = tuple(dict.fromkeys(instance(c, layer, m) for (layer, m), cands in XD_CANDIDATES.items() for c in cands))


# PE21: token groups (TG warps per (row group, k part) share each weight chunk through L1/L2, each with 8 NT tokens):
# m = 32 at m16's registers (PE9: F12M m16 beat cuBLAS on pooler/qkv, its NT 4 version lost at m32), and m = 16 at
# m8's. KSW is halved (or quartered) so CTAs keep 256 threads.
def _tg(nt, rtw, ksw, d, tg, xd=None):
    c = dict(NT=nt, RTW=rtw, RG=1, KSW=ksw, D=d, F12=True, TG=tg)
    if xd is not None:
        c["XD"] = xd
    return c


TG_CANDIDATES = {
    ("pooler", 32): [_tg(2, 1, 4, 2, 2), _tg(2, 1, 4, 2, 2, xd=1), _tg(1, 1, 2, 2, 4), _tg(1, 1, 2, 4, 4), _tg(2, 1, 8, 2, 2)],
    ("qkv_gate", 32): [_tg(2, 1, 4, 2, 2), _tg(2, 1, 4, 2, 2, xd=1), _tg(1, 1, 2, 2, 4), _tg(1, 1, 2, 4, 4), _tg(2, 1, 8, 2, 2)],
    ("unpooler", 32): [_tg(2, 1, 4, 2, 2), _tg(2, 2, 4, 1, 2), _tg(1, 1, 2, 2, 4), _tg(2, 1, 8, 2, 2)],
    ("o", 32): [_tg(2, 2, 4, 2, 2), _tg(2, 1, 4, 2, 2), _tg(1, 2, 2, 4, 4), _tg(1, 1, 2, 4, 4)],
    ("qkv_gate", 16): [_tg(1, 1, 4, 4, 2), _tg(1, 1, 4, 2, 2)],
    ("pooler", 16): [_tg(1, 1, 4, 2, 2), _tg(1, 1, 4, 4, 2)],
    ("unpooler", 16): [_tg(1, 1, 4, 2, 2), _tg(1, 2, 4, 1, 2)],
    ("o", 16): [_tg(1, 2, 4, 2, 2), _tg(1, 1, 4, 2, 2)],
}
TG_LAB = tuple(dict.fromkeys(instance(c, layer, m) for (layer, m), cands in TG_CANDIDATES.items() for c in cands))

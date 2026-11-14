"""P9 configs: candidates per (layer, m) and the instance lists compiled into a module.

A config is dict(NT, RTW, RG, KSW, D, CS); its template instance adds (HILO, OUT_F32) = (router,
router). Candidates must satisfy the launcher's constraints (proj/local/check_p9.py check_config)
and aim at a whole number of waves: CTAs within [112, 140] (one per SM) or [240, 270] (two per SM,
which also needs shared memory <= 113 KB per CTA).
"""

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}
SMALL_M = (1, 8, 16, 32)


def n_rows(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else SHAPES[layer][0]


def nt_for(m):
    return (m + 7) // 8


def smem_bytes(nt, rtw, rg, ksw, cs, k):
    return ksw * rg * rtw * 16 * 8 * nt * 4


def candidates_f12(layer, m, max_n=10):
    """P22 (F12) variants for the BF16 layers: 3 uint4 per tile per chunk in the ring instead of 4."""
    if layer == "router":
        return []
    return [dict(c, F12=True) for c in candidates(layer, m, max_n, f12=True)]


def candidates(layer, m, max_n=12, f12=False):
    """Ranked candidate configs for one shape (the lab times them)."""
    n, k = n_rows(layer, m), SHAPES[layer][1]
    nt = nt_for(m)
    hilo = layer == "router"
    out = []
    for rtw in (1, 2):
        for rg in (1, 2, 3, 4, 6, 8, 9, 12, 16):
            for ksw in (1, 2, 4, 8):
                warps = rg * ksw
                if warps < 4 or warps > 16:
                    continue
                for cs in (1, 2, 4, 8):
                    rt = rg * rtw
                    kc = k // cs
                    if n % (rt * 16) or k % (cs * ksw * 64) or (rt * 16) % cs:
                        continue
                    # x is re-read from L2 by every row block: keep that under half the weight bytes.
                    if (n // (rt * 16)) * 8 * nt * k * 2 > 0.5 * n * k * (4 if hilo else 2):
                        continue
                    ctas = n // (rt * 16) * cs
                    sm = smem_bytes(nt, rtw, rg, ksw, cs, k)
                    # One wave: up to 132 CTAs, or up to 144 when two fit per SM (qkv has 144 row
                    # tiles); two waves: 192-280 CTAs with two per SM. At least 64 CTAs either way.
                    one_wave = 64 <= ctas <= 132 or (132 < ctas <= 144 and sm <= 113 * 1024)
                    two_wave = 192 <= ctas <= 280 and sm <= 113 * 1024
                    if not (one_wave or two_wave):
                        continue
                    chunks = kc // ksw // 64
                    for d in (1, 2, 4):
                        slots = 3 if f12 else 4 * (2 if hilo else 1)
                        ring = d * (rtw * slots * 4 + nt * 8)   # registers per thread
                        if d > chunks or ring > 96:
                            continue
                        # Prefer more bytes in flight per SM, fewer cluster peers, then fewer warps.
                        inflight = warps * rtw * d * (2 if hilo else 1) * 2048 * (1 if one_wave else 2)
                        # SMs kept busy (a 64-CTA grid leaves half idle) times bytes in flight.
                        busy = min(ctas, 132) / 132
                        out.append((-inflight * busy, cs, warps, dict(NT=nt, RTW=rtw, RG=rg, KSW=ksw, D=d, CS=cs)))
    out.sort(key=lambda e: e[:3])
    seen, ranked = set(), []
    for *_, cfg in out:
        key = tuple(cfg.values())
        if key not in seen:
            seen.add(key)
            ranked.append(cfg)
    # Keep variety: the top ones plus one per cluster size.
    picked = ranked[:max_n - 4]
    for cs in (1, 2, 4, 8):
        extra = next((c for c in ranked if c["CS"] == cs and c not in picked), None)
        if extra:
            picked.append(extra)
    return picked[:max_n]


def instance(cfg, layer):
    hilo = layer == "router"
    base = (cfg["NT"], cfg["RTW"], cfg["RG"], cfg["KSW"], cfg["D"], cfg["CS"], hilo, hilo)
    return base + ((True,) if cfg.get("F12") else ())


def f12_instances(max_per_shape=5):
    insts = []
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        for m in SMALL_M:
            for cfg in candidates_f12(layer, m)[:max_per_shape]:
                inst = instance(cfg, layer)
                if inst not in insts:
                    insts.append(inst)
    return tuple(insts)


F12 = f12_instances()


def lab_instances(max_per_shape=4):
    insts = []
    for layer in SHAPES:
        for m in SMALL_M:
            for cfg in candidates(layer, m)[:max_per_shape]:
                inst = instance(cfg, layer)
                if inst not in insts:
                    insts.append(inst)
    return tuple(insts)


LAB = lab_instances()


def router_instances(max_per_shape=6):
    insts = []
    for m in SMALL_M:
        for cfg in candidates("router", m)[:max_per_shape]:
            inst = instance(cfg, "router")
            if inst not in insts:
                insts.append(inst)
    return tuple(insts)


ROUTER = router_instances()

# v4's router instances (PE6), kept in labs built on v4 so their timed result includes them.
ROUTER_V4 = ((1, 1, 2, 8, 2, 8, True, True), (2, 1, 1, 8, 2, 8, True, True))
F12_V4 = F12 + ROUTER_V4

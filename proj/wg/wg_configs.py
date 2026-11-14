"""WG configs: per (BF16 layer, m in 128/256) the (KS, ST) choices whose grid (row tiles x token tiles x
KS) lands near a whole number of waves; instance (KS, ST, TT) with TT = m / 128."""

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
MS = (128, 256)


def tiles(m):
    """Token tiles of 128 (m < 128 runs one zero-padded tile: TMA fills the missing rows with zeros)."""
    return max(1, -(-m // 128))


def candidates(layer, m):
    n, k = SHAPES[layer]
    tt, chunks = tiles(m), k // 64
    out = []
    for ks in (1, 2, 4, 8):
        ctas = n // 128 * tt * ks
        if chunks < 2 * ks or not 64 <= ctas <= 300:
            continue
        for st in (4, 3):
            # one CTA per SM at ST 4 (129 KB of shared memory), two at ST 3 (97 KB)
            per_sm = 1 if st == 4 else 2
            waves = ctas / (132 * per_sm)
            balance = waves / -(-ctas // (132 * per_sm))
            out.append((-balance, ks, st, dict(KS=ks, ST=st)))
    out.sort(key=lambda e: e[:3])
    return [e[-1] for e in out][:5]


def candidates_deep(layer, m):
    """PE13: deeper pipelines (5-6 stages, up to 197 KB of shared memory; cuBLAS's nvjet uses 8) and cluster
    sizes of at most 4 (PE12: 4-CTA clusters ran slower than 2-CTA ones at equal work per CTA)."""
    n, k = SHAPES[layer]
    tt, chunks = tiles(m), k // 64
    out = []
    for ks in (1, 2, 4):
        ctas = n // 128 * tt * ks
        if chunks < 2 * ks or not 64 <= ctas <= 300:
            continue
        for st in (6, 5):
            balance = (ctas / 132) / -(-ctas // 132)
            out.append((-balance, ks, -st, dict(KS=ks, ST=st)))
    out.sort(key=lambda e: e[:3])
    return [e[-1] for e in out][:4]


def instance(cfg, m):
    return (cfg["KS"], cfg["ST"], tiles(m))


def lab_instances():
    insts = []
    for layer in SHAPES:
        for m in MS:
            for c in candidates(layer, m):
                i = instance(c, m)
                if i not in insts:
                    insts.append(i)
    return tuple(insts)


LAB = lab_instances()


WGF_MS = (16, 32, 128, 256)


def wgf_candidates(layer, m):
    """WGF (12-bit RS): (KS, ST, D); x stages are 16 KB, so ST 4-6 fit with room for two CTAs per SM's
    worth of nothing else: one CTA per SM, 384 threads."""
    n, k = SHAPES[layer]
    tt, chunks = tiles(m), k // 64
    out = []
    for ks in (1, 2, 4, 8):
        ctas = n // 128 * tt * ks
        if chunks < 2 * ks or not 64 <= ctas <= 300:
            continue
        for st in (4, 6):
            for d in (2, 1):
                waves = ctas / 132
                balance = waves / -(-ctas // 132)
                out.append((-balance, -d, -st, ks, dict(KS=ks, ST=st, D=d)))
    out.sort(key=lambda e: e[:4])
    return [e[-1] for e in out][:5]


def wgf_instance(cfg, m):
    return (cfg["KS"], cfg["ST"], tiles(m), cfg["D"])


def wgf_lab_instances():
    insts = []
    for layer in SHAPES:
        for m in WGF_MS:
            for c in wgf_candidates(layer, m):
                i = wgf_instance(c, m)
                if i not in insts:
                    insts.append(i)
    return tuple(insts)


WGF_LAB = wgf_lab_instances()


NONE = ()


def deep_lab_instances():
    insts = []
    for layer in SHAPES:
        for m in MS:
            for c in candidates_deep(layer, m):
                i = instance(c, m)
                if i not in insts:
                    insts.append(i)
    return tuple(insts)


LAB_DEEP = deep_lab_instances()


PROFILE = ((2, 4, 1),)   # unpooler m128 KS2 ST4 (Modal profiling)

"""F12V configs: candidates per BF16 layer at m = 1 and the instance lists compiled into a module.

A config is dict(R, RG, KSW, D); its instance is (R, RG, KSW, NS, D) with NS = K / 1024 / KSW.
Constraints (the kernel's static_assert and the launcher's TORCH_CHECKs): rows % (R * RG) == 0,
1 <= D <= NS, 64-512 threads. Ranking is only a guess at bytes in flight per SM (registers estimated,
the lab prints the real counts) times wave balance; the lab times them all.
"""

import math

ROWS_M1 = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (1024, 4096)}


def regs_est(r, d):
    return 12 * d * r + 16 * d + 4 * r + 40


def instance(cfg, layer):
    k = ROWS_M1[layer][1]
    return (cfg["R"], cfg["RG"], cfg["KSW"], k // 1024 // cfg["KSW"], cfg["D"])


def candidates(layer, max_n=12):
    n, k = ROWS_M1[layer]
    kbs = k // 1024
    out = []
    for r in (1, 2, 3, 4):
        for rg in range(1, 17):
            for ksw in (1, 2, 4, 8):
                threads = 32 * rg * ksw
                if kbs % ksw or not 64 <= threads <= 512:
                    continue
                ns, rb = kbs // ksw, r * rg
                if n % rb or n // rb < 128:
                    continue
                for d in (1, 2, 4):
                    regs = regs_est(r, d)
                    if d > ns or regs > (128 if threads > 256 else 168):
                        continue
                    smem = kbs * 2048 + ksw * rb * 4
                    regs_r = (regs + 7) // 8 * 8
                    per_sm = min(65536 // (regs_r * threads), 2048 // threads, 32, (227 * 1024) // smem)
                    if per_sm < 1:
                        continue
                    ctas = n // rb
                    inflight = per_sm * threads * d * r * 48
                    waves = ctas / (per_sm * 132)
                    balance = waves / math.ceil(waves)
                    score = min(inflight, 64 * 1024) * balance
                    pref = {256: 0, 128: 1, 512: 2}.get(threads, 3)
                    out.append(((-score, pref, ctas), d == ns, threads, dict(R=r, RG=rg, KSW=ksw, D=d)))
    out.sort(key=lambda e: e[0])
    one_shot = [e[-1] for e in out if e[1]]
    ring = [e[-1] for e in out if not e[1]]
    picked = one_shot[:4] + ring[:4]
    # Variety: the best config of each thread count not yet picked, then the rest by rank.
    for t in (256, 128, 512, 64):
        extra = next((e[-1] for e in out if e[2] == t and e[-1] not in picked), None)
        if extra:
            picked.append(extra)
    picked += [e[-1] for e in out if e[-1] not in picked]
    return picked[:max_n]


def lab_instances(max_per_shape=10):
    insts = []
    for layer in ROWS_M1:
        for cfg in candidates(layer, max_per_shape):
            inst = instance(cfg, layer)
            if inst not in insts:
                insts.append(inst)
    return tuple(insts)


LAB = lab_instances()

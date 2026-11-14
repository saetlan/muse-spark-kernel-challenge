"""DeepGEMM's sm90 BF16 GEMM config heuristic, ported from sgl-project/DeepGEMM cbe1727 (the judge image's
deep_gemm 0.2.1.post1+muse.cbe1727): csrc/jit_kernels/heuristics/sm90.hpp (get_layout_candidates,
get_storage_config, get_pipeline_config, get_launch_config, get_layout_info) and heuristics/common.hpp
(get_best_config: first strict minimum of num_cycles), for desc = Normal GEMM, KernelNoSF, A/B BF16 K-major,
D BF16, no accumulation, 132 SMs, block_n_multiple_of 1. Out = x [m, k] @ W[:n].T: DeepGEMM's A = x, B = W.

dg_best(m, n, k) -> dict(BM, BN, BK, MC, MN, ST, SWA, SWB, SWD, NMATH, SMEM).
"""

import math

SMEM_CAPACITY = 232448
NUM_SMS = 132


def swizzle_mode(block_size, elem_size):
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise ValueError("unreachable")


def storage(bm, bn, bk):
    return dict(SWA=swizzle_mode(bk, 2), SWB=swizzle_mode(bk, 2), SWD=swizzle_mode(bn, 2))


def pipeline(bm, bn, bk):
    smem_cd = -(-(bm * bn * 2) // 1024) * 1024
    smem_barriers = 16 * 8 * 2
    per_stage = bm * bk * 2 + bn * bk * 2
    extra = smem_cd + smem_barriers
    stages = min((SMEM_CAPACITY - extra) // per_stage, 16)
    return dict(ST=stages, SMEM=extra + stages * per_stage)


def candidates(m, n, k, num_sms=NUM_SMS):
    bk = 128 * 8 // 16
    block_ms = [64, 128]
    if m <= 16:
        block_ms.append(16)
    if m <= 32:
        block_ms.append(32)
    block_ms.append(256)                         # BF16 output
    block_ns = list(range(16, 257, 16))          # step lcm(16, 1)
    out = []
    for cm in (1, 2):
        for cn in (1, 2):
            if cm * cn > 2 or num_sms % (cm * cn):
                continue
            for bm in block_ms:
                for bn in block_ns:
                    if bm > 128 and bn > 128:
                        continue
                    st = storage(bm, bn, bk)
                    if st["SWA"] % 64 or st["SWB"] % 64:
                        continue
                    stages = pipeline(bm, bn, bk)["ST"]
                    if stages < 3 or (bm * bn < 128 * 192 and stages < 4):
                        continue
                    out.append((bm, bn, bk, cm, cn))
    return out


def layout_cycles(m, n, k, bm, bn, bk, cm, cn, num_sms=NUM_SMS):
    num_blocks = -(-m // bm) * -(-n // bn)
    num_waves = -(-num_blocks // num_sms)
    l2_bw = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bw = 128 * num_sms
    e = 2
    l2_ab = k * (bm // cn + bn // cm) * e
    l1_ab = k * (bm + bn) * e
    l1_tc = k * (max(64, bm) + bn) * e + bm * bn * e
    cd = bm * bn * e
    l2_cycles = (l2_ab + cd) * num_blocks // l2_bw
    l1_cycles = (l1_ab + l1_tc + cd) * num_blocks // l1_bw
    wave_eff = num_blocks / (num_waves * num_sms)
    cycles = int(max(l1_cycles, l2_cycles) / wave_eff)
    if cn * cm > 1 and num_waves <= 1:
        cycles = 2 ** 63 - 1
    return cycles


def dg_best(m, n, k, num_sms=NUM_SMS):
    cands = candidates(m, n, k, num_sms)
    best, best_c = cands[0], layout_cycles(m, n, k, *cands[0], num_sms=num_sms)
    for c in cands[1:]:
        cyc = layout_cycles(m, n, k, *c, num_sms=num_sms)
        if cyc < best_c:
            best, best_c = c, cyc
    bm, bn, bk, cm, cn = best
    cfg = dict(BM=bm, BN=bn, BK=bk, MC=cm, MN=cn, NMATH=128 if bm <= 64 else 256)
    cfg.update(storage(bm, bn, bk))
    cfg.update(pipeline(bm, bn, bk))
    return cfg


SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}

if __name__ == "__main__":
    for m in (32, 128, 256):
        for layer, (n, k) in SHAPES.items():
            c = dg_best(m, n, k)
            print(f"{layer:9s} m{m:<4d} BM{c['BM']} BN{c['BN']} cluster {c['MC']}x{c['MN']} stages {c['ST']} "
                  f"SWD {c['SWD']} math {c['NMATH']} smem {c['SMEM']} blocks {-(-m // c['BM']) * -(-n // c['BN'])}")

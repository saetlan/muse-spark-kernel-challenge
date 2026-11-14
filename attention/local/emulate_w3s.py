"""CPU check of W3s's new logic (make_w3s.py) with the kernel's own formulas: the segment table (thread 0's loops), each
pair's stages per segment (first = s_off + ((par - s_off % NP) % NP + NP) % NP, step NP), per-stage keys and masks
(key < s_end, key <= n - L + token), partial slots (x - P_b / Qs) and _combine_s's slot count ((last - first + 1) * NP).
Per stage it computes the exact partial (m in log2 units with the kernel's qk_scale, l, o) in float64 instead of the
tile math (unchanged from W3b, emulated by emulate_cuda_w1.py), merges with _combine_s's formula and compares with direct
causal attention over each request's keys. Also asserts: every (request, stage) consumed exactly once over all CTAs and
pairs, every partial slot a merge reads written exactly once, no CTA writes beyond the merge's slots. Lengths: the
judge's agentic spread and the tests' log-normal "mixed" spread (many short requests), batch 1..32, G 132 and small G
(several segments per CTA). Mutants must fail.
Run: .venv/bin/python attention/local/emulate_w3s.py"""
import math
import random
import sys

import torch

SKEYS, L, NP = 64, 8, 2
MAXSEG = 32
MUT = set()


def seg_table(seqlens, G, x):
    stot = sum((n + SKEYS - 1) // SKEYS for n in seqlens)
    Qs = max((stot + G - 1) // G, 1)
    c0, c1 = x * Qs, min(x * Qs + Qs, stot)
    pfx = ns = off = 0
    segs = []
    for bb, nb in enumerate(seqlens):
        if not pfx < c1:
            break
        sb = (nb + SKEYS - 1) // SKEYS
        a, e = max(pfx, c0), min(pfx + sb, c1)
        if a < e and ns < MAXSEG:
            slot = x - pfx // Qs if "slot" not in MUT else x - (pfx + sb - 1) // Qs
            segs.append(dict(b=bb, n=nb, k0=(a - pfx) * SKEYS, off=off, cnt=e - a, slot=slot))
            off += e - a
            ns += 1
        pfx += sb
    return segs, off


def merge_count(seqlens, G, b):
    sb = [(n + 63) // 64 for n in seqlens]
    stot = sum(sb)
    qs = max((stot + G - 1) // G, 1)
    pb = sum(sb[:b])
    first, last = pb // qs, (pb + sb[b] - 1) // qs
    return (last - first + 1) * NP


def run(seqlens, G, seed=0):
    B = len(seqlens)
    rnd = torch.Generator().manual_seed(seed)
    nmax = max(seqlens)
    K = torch.randn(B, nmax, 64, generator=rnd, dtype=torch.float64)
    V = torch.randn(B, nmax, 64, generator=rnd, dtype=torch.float64)
    Qm = torch.randn(B, L, 64, generator=rnd, dtype=torch.float64)
    qk = 0.125 * 1.4426950408889634
    # partial store: [b][tok][slot*NP + par] -> (m, l, o)
    parts = {}
    consumed = {}
    for x in range(G):
        segs, nst = seg_table(seqlens, G, x)
        assert sum(s["cnt"] for s in segs) == nst
        for par in range(NP):
            for s in segs:
                s_end = min(s["n"], s["k0"] + s["cnt"] * SKEYS) if "end" not in MUT else min(s["n"], s["k0"] + (s["cnt"] - 1) * SKEYS)
                first = s["off"] + ((par - s["off"] % NP) % NP + NP) % NP if "first" not in MUT else s["off"]
                segend = s["off"] + s["cnt"]
                nloc = (segend - first + NP - 1) // NP if first < segend else 0
                b = s["b"]
                for tok in range(L):
                    m, l_, o = -1.0e30, 0.0, torch.zeros(64, dtype=torch.float64)
                    pos = s["n"] - L + tok
                    for i in range(nloc):
                        j = first + NP * i
                        s0 = s["k0"] + (j - s["off"]) * SKEYS
                        if tok == 0:
                            key_stage = (b, s0 // SKEYS)
                            consumed[key_stage] = consumed.get(key_stage, 0) + 1
                        keys = [k for k in range(s0, s0 + SKEYS) if k < s_end and k <= pos]
                        if not keys:
                            continue
                        sc = (K[b, keys] @ Qm[b, tok]) * qk
                        mn = max(m, float(sc.max()))
                        a = 2.0 ** (m - mn)
                        p = torch.pow(2.0, sc - mn)
                        l_ = l_ * a + float(p.sum())
                        o = o * a + p @ V[b, keys]
                        m = mn
                    key = (b, tok, s["slot"] * NP + par)
                    assert key not in parts, f"partial written twice: {key}"
                    parts[key] = (m, l_, o)
    for b, n in enumerate(seqlens):
        for k in range((n + SKEYS - 1) // SKEYS):
            c = consumed.get((b, k), 0)
            if c != 1:
                return f"request {b} stage {k} consumed {c} times"
    worst = 0.0
    for b, n in enumerate(seqlens):
        cnt = merge_count(seqlens, G, b)
        for tok in range(L):
            have = [k for k in parts if k[0] == b and k[1] == tok]
            if sorted(s for _, _, s in have) != list(range(cnt)):
                return f"request {b} token {tok}: slots written {sorted(s for _, _, s in have)}, merge reads {cnt}"
            ms = [parts[(b, tok, s)] for s in range(cnt)]
            M = max(p[0] for p in ms)
            den = sum(p[1] * 2.0 ** (p[0] - M) for p in ms)
            num = sum(p[2] * 2.0 ** (p[0] - M) for p in ms)
            got = num / den
            pos = n - L + tok
            sc = (K[b, :pos + 1] @ Qm[b, tok]) * 0.125
            want = torch.softmax(sc, 0) @ V[b, :pos + 1]
            worst = max(worst, float((got - want).norm() / want.norm()))
    return None if worst < 1e-9 else f"merged output off by {worst:.2e}"


def cases():
    rnd = random.Random(5)
    out = []
    for B in (1, 2, 5, 16, 32):
        for G in (132, 7, 33):
            agentic = [int(3000 * (1 + 0.075 * (2 * rnd.random() - 1))) + L for _ in range(B)]
            mixed = [max(16, min(12000, int(3000 * math.exp(rnd.gauss(0, 1) - 0.5)))) + L for _ in range(B)]
            out.append((agentic, G))
            out.append((mixed, G))
    out.append(([9, 70, 64 + 8, 130, 3000], 132))     # tiny requests, stage boundaries
    return out


if __name__ == "__main__":
    bad = 0
    cs = cases()
    for i, (lens, G) in enumerate(cs):
        err = run(lens, G, seed=i)
        if err:
            print(f"FAIL B {len(lens)} G {G}: {err}")
            bad += 1
    print(f"{'ok  ' if not bad else 'FAIL'} W3s partition / slots / merge on {len(cs)} cases")
    # (benign by construction, not mutants: per-segment instead of global pair parity still consumes each stage once;
    # s_end without min(n) is masked by key <= n - L + token)
    for m in ("slot", "first", "end"):
        MUT.clear()
        MUT.add(m)
        caught = any(run(lens, G, seed=i) for i, (lens, G) in enumerate(cs[:14]))
        print(("caught  " if caught else "MISSED  ") + f"mutant {m}")
        bad += not caught
    MUT.clear()
    print("emulate_w3s:", "all ok" if not bad else f"{bad} problems")
    sys.exit(1 if bad else 0)

"""Local checks of X5/X10 (parts/x5.py) on the CPU, at real sizes, with the kernel's own formulas.

1. 12-bit decode: emulate dec_pair() and the kernel's plane addressing (uint4 bytes word, uint2 index
   words, shift 16q + 4m, byte selectors 0x4140 / 0x4342, base pairs, raw-tile fallback) on every
   tile, register and lane, and require bit-exact equality with the plain fragment words.
2. MMA + flush: rebuild each A tile from the lane registers using the PTX m16n8k16 fragment layout
   (independently of x5_frag_maps), B from x with the kernel's word index (k0/2 + t, + 4), the C
   fragment per PTX, then scatter with the kernel's flush formula; compare with W @ x^T in FP32.
3. Mutations: each planted bug must break check 1 or 2.

Run: .venv/bin/python decode_layer/local/check_x5.py
"""

import math
import sys
import types
from pathlib import Path

import torch

DECODE = Path(__file__).resolve().parents[1]


def load_x5():
    mod = types.ModuleType("x5_under_check")
    mod.load_inline = None
    mod.torch = torch
    src = (DECODE / "parts" / "x5.py").read_text()
    exec(compile(src, "parts/x5.py", "exec"), mod.__dict__)
    return mod


def u32(t):
    return t.to(torch.int64) & 0xFFFFFFFF


def byte_perm(x, sel):
    """__byte_perm(x, 0, sel) for selectors whose nibbles are 0-3 (bytes of x) or 4 (zero)."""
    out = torch.zeros_like(x)
    for pos in range(4):
        s = (sel >> (4 * pos)) & 0xF
        if s < 4:
            out |= ((x >> (8 * s)) & 0xFF) << (8 * pos)
    return out


def dec_pair(bw, sel, iL, iH, bb):
    X = byte_perm(bw, sel)
    ms = (X & 0x007F007F) | ((X & 0x00800080) << 8)
    e = ((bb - (iL | (iH << 16))) << 7) & 0xFFFFFFFF
    return (ms | e) & 0xFFFFFFFF


def kernel_regs_enc(pack, sels=(0x4140, 0x4342, 0x4140, 0x4342)):
    """a0..a3 per (rt, ks, lane) exactly as the ENC branch computes them."""
    rt, ks = pack["bases"].shape
    plain_w = u32(pack["plain"].view(torch.int32)).view(rt, ks, 32, 4)
    bv = u32(pack["enc_b"].contiguous().view(torch.int32)).view(rt, ks // 32, 16, 32, 4)
    iv = u32(pack["enc_i"]).view(rt, ks // 32, 16, 32, 2)
    regs = torch.zeros((rt, ks, 32, 4), dtype=torch.int64)
    for j in range(32):
        sp, q = j >> 1, j & 1
        b = pack["bases"].view(rt, ks // 32, 32)[:, :, j].to(torch.int64)[:, :, None]  # [rt, c, 1]
        bb = b | (b << 16)
        w0 = bv[:, :, sp, :, 2 if q else 0]
        w1 = bv[:, :, sp, :, 3 if q else 1]
        L, H = iv[:, :, sp, :, 0], iv[:, :, sp, :, 1]
        sh = 16 * q
        a = [dec_pair(w0, sels[0], (L >> (sh + 0)) & 15, (H >> (sh + 0)) & 15, bb),
             dec_pair(w0, sels[1], (L >> (sh + 4)) & 15, (H >> (sh + 4)) & 15, bb),
             dec_pair(w1, sels[2], (L >> (sh + 8)) & 15, (H >> (sh + 8)) & 15, bb),
             dec_pair(w1, sels[3], (L >> (sh + 12)) & 15, (H >> (sh + 12)) & 15, bb)]
        a = torch.stack(a, -1)  # [rt, c, lane, 4]
        raw = (b == 0)[..., None].expand_as(a)
        pw = plain_w.view(rt, ks // 32, 32, 32, 4)[:, :, j]
        regs.view(rt, ks // 32, 32, 32, 4)[:, :, j] = torch.where(raw, pw, a)
    return regs, plain_w


def ptx_a_tile(regs):
    """A [.., 16, 16] float from lane registers per the PTX m16n8k16 .bf16 A fragment layout."""
    lead = regs.shape[:-2]
    A = torch.zeros((*lead, 16, 16), dtype=torch.float32)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        spots = [((g, 2 * t), (g, 2 * t + 1)), ((g + 8, 2 * t), (g + 8, 2 * t + 1)),
                 ((g, 2 * t + 8), (g, 2 * t + 9)), ((g + 8, 2 * t + 8), (g + 8, 2 * t + 9))]
        for r, ((r0, c0), (r1, c1)) in enumerate(spots):
            w = regs[..., lane, r]
            lo = (w & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.bfloat16).float()
            hi = ((w >> 16) & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.bfloat16).float()
            A[..., r0, c0] = lo
            A[..., r1, c1] = hi
    return A


def emulate_gemv(regs, x, n, k, nt_tiles):
    """out [T, N] FP32 from the kernel's B indexing, PTX C layout and flush formula."""
    rt_n, ks_n = regs.shape[:2]
    T = x.shape[0]
    A = ptx_a_tile(regs)  # [rt, ks, 16, 16]
    xw = u32(x.contiguous().view(torch.int32)).view(T, k // 2)
    out = torch.zeros((T, n), dtype=torch.float64)
    for nt in range(nt_tiles):
        # B[kk][col] for col = token nt*8 + g, from lane (g, t)'s b0 (word k0/2 + t) and b1 (+ 4)
        B = torch.zeros((ks_n, 16, 8), dtype=torch.float32)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            tok = nt * 8 + g
            if tok >= T:
                continue
            for ks in range(ks_n):
                c, j = ks // 32, ks % 32
                # the producer's copy: x[tok, c*512 : c*512+512] at word tok*260 of the stage's x area
                # producer: dst + XOFF + tok * XROW for tok < T; consumer: xs + (nt*8 + g) * (XROW/4) + j*8 + t (+4)
                xs = torch.zeros(32 * 260, dtype=torch.int64)
                xs[tok * 260:tok * 260 + 256] = xw[tok, c * 256:c * 256 + 256]
                for wi, kk in ((0, 2 * t), (4, 2 * t + 8)):
                    w = xs[(nt * 8 + g) * 260 + j * 8 + t + wi]
                    lo = (w & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.bfloat16).float()
                    hi = ((w >> 16) & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.bfloat16).float()
                    B[ks, kk, g] = lo
                    B[ks, kk + 1, g] = hi
        C = torch.einsum("rkij,kjc->ric", A.double(), B.double())  # [rt, 16, 8], summed over ks
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            c = [C[:, g, 2 * t], C[:, g, 2 * t + 1], C[:, g + 8, 2 * t], C[:, g + 8, 2 * t + 1]]  # PTX C layout
            rows = torch.arange(rt_n) * 16 + g  # kernel: r0 = rt * 16 + g
            t0 = nt * 8 + 2 * t
            if t0 < T:
                out[t0, rows] += c[0]
                out[t0, rows + 8] += c[2]
            if t0 + 1 < T:
                out[t0 + 1, rows] += c[1]
                out[t0 + 1, rows + 8] += c[3]
    return out


def simulate_protocol(n, k, grid, orders=20, seed=0):
    """The kernel's item partition (begin/end per CTA), its flush sequence (one flush per row-tile run,
    nitems = items in that run) and the counter protocol under random CTA completion orders: every row
    tile must see exactly one last arriver, after all its chunks, and the counters must return to 0."""
    import random
    chunks = k // 512
    items = (n // 16) * chunks
    flushes = []  # (cta, rt, nitems)
    for b in range(grid):
        begin, end = items * b // grid, items * (b + 1) // grid
        cur, nit = -1, 0
        for it in range(begin, end):
            rt = it // chunks
            if rt != cur:
                if cur >= 0:
                    flushes.append((b, cur, nit))
                cur, nit = rt, 0
            nit += 1
        if cur >= 0:
            flushes.append((b, cur, nit))
    rng = random.Random(seed)
    for _ in range(orders):
        order = flushes[:]
        rng.shuffle(order)
        cnt = [0] * (n // 16)
        lasts = [0] * (n // 16)
        for _, rt, nit in order:
            old = cnt[rt]
            cnt[rt] += nit
            if old + nit == chunks:
                lasts[rt] += 1
                cnt[rt] = 0
        if any(c != 0 for c in cnt) or any(v != 1 for v in lasts):
            return False
    return True


def make_w(n, k, gen):
    w = (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(torch.bfloat16)
    w[3, 100] = 0.0  # a zero -> raw tile
    w[40, 2000 % k] = -0.0
    w[17, 5] = 1e-38  # subnormal -> raw
    w[50, 33] = 40.0  # wide span -> raw
    return w


def check_decode(x5, w, sels=None):
    pack = x5.x5_pack(w)
    regs, plain_w = kernel_regs_enc(pack) if sels is None else kernel_regs_enc(pack, sels)
    exact = bool((regs == plain_w).all())
    return pack, regs, exact


def main():
    x5 = load_x5()
    gen = torch.Generator().manual_seed(3)
    ok = True
    for name, (n, k) in {"qkv": (2304, 8192), "o": (8192, 1024), "unp_slice": (1024, 4096)}.items():
        w = make_w(n, k, gen)
        pack, regs, exact = check_decode(x5, w)
        size = x5.x5_enc_bytes(pack)
        print(f"{name} {n}x{k}: 12-bit decode bit-exact {exact}; raw tiles {pack['raw_tiles']}/{pack['tiles']} "
              f"({100 * pack['raw_tiles'] / pack['tiles']:.2f}%); bytes {size / (n * k * 2):.3f} of BF16")
        ok &= exact
    # MMA + flush emulation on a smaller matrix (python loops), T up to 32 with 4 n-tiles
    w = make_w(64, 1024, gen)
    pack = x5.x5_pack(w)
    plain_w = u32(pack["plain"].view(torch.int32)).view(64 // 16, 1024 // 16, 32, 4)
    for T, ntt in ((1, 1), (5, 1), (8, 1), (13, 2), (32, 4)):
        x = torch.randn((T, 1024), generator=gen).to(torch.bfloat16)
        got = emulate_gemv(plain_w, x, 64, 1024, ntt)
        ref = x.double() @ w.double().T
        err = float((got - ref).norm() / ref.norm())
        print(f"gemv emulation T={T} nt={ntt}: rel err vs W @ x^T {err:.2e}")
        ok &= err < 1e-6
    # mutations
    w = make_w(1024, 4096, gen)
    caught = 0
    muts = {"swapped selectors": (0x4342, 0x4140, 0x4140, 0x4342), "selector to byte 3": (0x4143, 0x4342, 0x4140, 0x4342)}
    for mname, sels in muts.items():
        _, _, exact = check_decode(x5, w, sels)
        caught += not exact
        print(("caught   " if not exact else "MISSED   ") + mname)
    # wrong fragment map: rows of a2/a3 swapped with a0/a1
    real = x5.x5_frag_maps

    def bad_maps(device):
        r, c = real(device)
        return r[:, [2, 3, 0, 1, 4, 5, 6, 7]], c

    x5.x5_frag_maps = bad_maps
    pk = x5.x5_pack(make_w(64, 1024, gen))
    x5.x5_frag_maps = real
    wbad = make_w(64, 1024, torch.Generator().manual_seed(9))
    pk = x5.x5_pack.__globals__["x5_pack"](wbad)  # rebuild with the real map for a reference
    x5.x5_frag_maps = bad_maps
    pk_bad = x5.x5_pack(wbad)
    x5.x5_frag_maps = real
    x = torch.randn((8, 1024), generator=gen).to(torch.bfloat16)
    pw = u32(pk_bad["plain"].view(torch.int32)).view(4, 64, 32, 4)
    err = float((emulate_gemv(pw, x, 64, 1024, 1) - x.double() @ wbad.double().T).norm() / (x.double() @ wbad.double().T).norm())
    caught += err > 1e-3
    print(("caught   " if err > 1e-3 else "MISSED   ") + f"wrong fragment row map (err {err:.2e})")
    proto = all(simulate_protocol(n, k, g) for n, k in ((2304, 8192), (8192, 1024), (4096, 8192), (1024, 4096), (8192, 4096))
                for g in (132, 264))
    print(f"last-arriver protocol (5 weights x 2 grids x 20 random orders): {'ok' if proto else 'BROKEN'}")
    ok &= proto
    print(f"mutations caught {caught}/3")
    ok &= caught == 3
    if not ok:
        raise SystemExit("check_x5: FAILED")
    print("check_x5: all checks passed")


if __name__ == "__main__":
    main()

"""CPU check of P9's index math at the real shapes: emulate p9_kernel with the kernel's own formulas
(fragment stream addresses wp + (tile * ksteps_total + ks0) * 32 * AS + lane + step * 32 * AS + h * 32,
x staging with the padded stride, B-fragment reads at kk = kloc0 + 16 step + 2 t (+8), C-fragment
scatter part[(kpart * ROWS + row) * MT + tok], the fixed-order reduction over peers and K parts, and
the output index tok * n_rows + rowblock * ROWS + row), on packed weights from p9_layout, and compare
with an FP64 reference within half the judge's bounds. MMAs are emulated in FP32 from the
reconstructed fragments (layout per PTX ISA 9.4 Figures 82, 84, 86).

Also checks the config constraints the launcher assumes (n_rows % (RT*16) == 0, k % (CS*KSW*16) == 0,
K slice % 64 == 0 for conflict-free x reads, shared memory <= 227 KB).

Run: .venv/bin/python proj/local/check_p9.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p9"))
from p9 import p9_smem_bytes  # noqa: E402
from p9_layout import p9_pack, p9_pack_hilo  # noqa: E402
from p22_layout import p22_decode_step, p22_device_planes  # noqa: E402

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "unpooler_s": (1024, 4096), "router": (256, 8192)}


def check_config(n_rows, k, cfg):
    nt, rtw, rg, ksw, d, cs = cfg["NT"], cfg["RTW"], cfg["RG"], cfg["KSW"], cfg["D"], cfg["CS"]
    rt = rg * rtw
    kc = k // cs
    problems = []
    if n_rows % (rt * 16):
        problems.append(f"n_rows {n_rows} % {rt * 16}")
    if k % (cs * ksw * 16):
        problems.append(f"k {k} % {cs * ksw * 16}")
    if (kc // ksw) % 64:
        problems.append(f"warp K part {kc // ksw} % 64 (whole permuted chunks)")
    if (rt * 16) % cs:
        problems.append(f"rows {rt * 16} % CS {cs}")
    if p9_smem_bytes(nt, rtw, rg, ksw, cs, k) > 227 * 1024:
        problems.append(f"smem {p9_smem_bytes(nt, rtw, rg, ksw, cs, k)}")
    if rg * ksw * 32 > 1024:
        problems.append("threads > 1024")
    return problems


def a_from_lane_words(frag_tile):
    """frag_tile [32, 8] bf16 -> [16, 16] using the PTX Figure 82 formulas (vectorized)."""
    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    a = torch.empty(16, 16, dtype=frag_tile.dtype)
    for j in range(4):
        for p in range(2):
            a[g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + p] = frag_tile[:, 2 * j + p]
    return a


def emulate(wpk, x, n_rows, k, cfg, hilo):
    """What p9_kernel writes to out (FP32 before the final cast), for every CTA: A fragments from the
    packed stream at wp + (tile * ksteps_total + 4 (c0 + cc) + j) * 32 AS + h * 32 + lane, B words of
    lane (g, t) from x[8 n + g][64 (c0 + cc) + 16 t + 4 j + {0,1}] (b0) and + {2,3} (b1) placed per
    PTX Figure 84 (b0: k = 2 t + p, b1: k = 2 t + 8 + p, column g), C scattered per Figure 86."""
    nt, rtw, rg, ksw, cs = cfg["NT"], cfg["RTW"], cfg["RG"], cfg["KSW"], cfg["CS"]
    rt, mt, as_ = rg * rtw, 8 * nt, 2 if hilo else 1
    rows = rt * 16
    kc, kw = k // cs, k // cs // ksw
    chunks, ks_total = kw // 64, k // 16
    m = x.shape[0]
    flat = wpk.reshape(-1, 8)
    xpad = torch.zeros(mt, k, dtype=torch.bfloat16)
    xpad[:m] = x
    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    out = torch.zeros(m, n_rows, dtype=torch.float32)
    for rowblock in range(n_rows // rows):
        parts = []
        for rank in range(cs):
            part = torch.zeros(ksw, rows, mt, dtype=torch.float32)
            for warp in range(rg * ksw):
                grp, kpart = warp // ksw, warp % ksw
                c0 = (rank * kc + kpart * kw) // 64
                for i in range(rtw):
                    tile = rowblock * rt + grp * rtw + i
                    base = (tile * ks_total + 4 * c0) * (32 * as_)
                    acc = torch.zeros(16, mt, dtype=torch.float64)
                    for cc in range(chunks):
                        for j in range(4):
                            b = torch.zeros(16, mt, dtype=torch.float64)
                            for nn in range(nt):
                                xk = 64 * (c0 + cc) + 16 * t + 4 * j          # per lane
                                xr = xpad[8 * nn + g]                          # [32, k]
                                for p in range(2):
                                    b[2 * t + p, 8 * nn + g] = xr[lane, xk + p].double()
                                    b[2 * t + 8 + p, 8 * nn + g] = xr[lane, xk + 2 + p].double()
                            for h in range(as_):
                                s0 = base + (4 * cc + j) * (32 * as_) + h * 32
                                acc += a_from_lane_words(flat[s0:s0 + 32]).double() @ b
                    part[kpart, (grp * rtw + i) * 16:(grp * rtw + i) * 16 + 16, :] = acc.float()
            parts.append(part)
        share = rows // cs
        for rank in range(cs):
            rsel = slice(rank * share, (rank + 1) * share)
            total = torch.zeros(share, mt, dtype=torch.float32)
            for peer in range(cs):          # fixed order: peer, then K part
                for kp in range(ksw):
                    total += parts[peer][kp, rsel, :]
            out[:, rowblock * rows + rank * share:rowblock * rows + (rank + 1) * share] = total[:, :m].T
    return out


def emulate_f12(pl, x, n_rows, k, cfg):
    """p9_kernel<..., F12=true>: per chunk the lane loads Q = q[tile][c][lane] (uint4) and R =
    r[tile][c][0..1][lane]; step j decodes word j of Q with words 2 (j & 1), 2 (j & 1) + 1 of R uint4
    j >> 1 (p22_decode_step = the kernel's ops); the reducing CTA adds x * w_true - x * w_dec per escape."""
    nt, rtw, rg, ksw, cs = cfg["NT"], cfg["RTW"], cfg["RG"], cfg["KSW"], cfg["CS"]
    rt, mt = rg * rtw, 8 * nt
    rows = rt * 16
    kc, kw = k // cs, k // cs // ksw
    chunks = kw // 64
    m = x.shape[0]
    q, r, tops = pl["q"].to(torch.int64) & 0xFFFFFFFF, pl["r"].to(torch.int64) & 0xFFFFFFFF, pl["tops"].to(torch.int64)
    xpad = torch.zeros(mt, k, dtype=torch.bfloat16)
    xpad[:m] = x
    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    out = torch.zeros(m, n_rows, dtype=torch.float32)
    for rowblock in range(n_rows // rows):
        parts = []
        for rank in range(cs):
            part = torch.zeros(ksw, rows, mt, dtype=torch.float32)
            for warp in range(rg * ksw):
                grp, kpart = warp // ksw, warp % ksw
                c0 = (rank * kc + kpart * kw) // 64
                for i in range(rtw):
                    tile = rowblock * rt + grp * rtw + i
                    acc = torch.zeros(16, mt, dtype=torch.float64)
                    for cc in range(chunks):
                        c = c0 + cc
                        for j in range(4):
                            rv = r[tile, c, j >> 1]                       # [32 lanes, 4 words]
                            regs = p22_decode_step(q[tile, c, :, j], rv[:, 2 * (j & 1)], rv[:, 2 * (j & 1) + 1], tops[tile])
                            frag = torch.zeros(32, 8, dtype=torch.int64)
                            for reg, v in enumerate(regs):
                                frag[:, 2 * reg], frag[:, 2 * reg + 1] = v & 0xFFFF, (v >> 16) & 0xFFFF
                            a = a_from_lane_words(frag.to(torch.int16).view(torch.bfloat16)).double()
                            b = torch.zeros(16, mt, dtype=torch.float64)
                            for nn in range(nt):
                                xk = 64 * c + 16 * t + 4 * j
                                xr = xpad[8 * nn + g]
                                for p in range(2):
                                    b[2 * t + p, 8 * nn + g] = xr[lane, xk + p].double()
                                    b[2 * t + 8 + p, 8 * nn + g] = xr[lane, xk + 2 + p].double()
                            acc += a @ b
                    part[kpart, (grp * rtw + i) * 16:(grp * rtw + i) * 16 + 16, :] = acc.float()
            parts.append(part)
        share = rows // cs
        for rank in range(cs):
            for rr in range(share):
                row = rank * share + rr
                orow = rowblock * rows + row
                wdec = torch.tensor([int(tops[orow // 16] - 15) << 23], dtype=torch.int32).view(torch.float32)
                for tok in range(m):
                    s_ = torch.zeros((), dtype=torch.float32)
                    for peer in range(cs):
                        for kp in range(ksw):
                            s_ += parts[peer][kp, row, tok]
                    for e in range(int(pl["erow"][orow]), int(pl["erow"][orow + 1])):
                        xv = x[tok, int(pl["ecol"][e])].float()
                        wt = torch.tensor([int(pl["ebits"][e]) & 0xFFFF], dtype=torch.int32).bitwise_left_shift(16).view(torch.float32)
                        s_ = s_ + xv * wt - xv * wdec
                    out[tok, orow] = s_
    return out


def main_f12():
    torch.manual_seed(2)
    cases = [("pooler", 1, dict(NT=1, RTW=1, RG=2, KSW=8, D=4, CS=2)), ("o", 16, dict(NT=2, RTW=2, RG=2, KSW=1, D=2, CS=1)),
             ("qkv_gate", 32, dict(NT=4, RTW=1, RG=4, KSW=2, D=2, CS=4))]
    for name, m, cfg in cases:
        n_full, k = SHAPES[name]
        rows = cfg["RG"] * cfg["RTW"] * 16
        w = (torch.randn(rows, k) / math.sqrt(k)).to(torch.bfloat16)
        w[3, 100] = 3.0                      # planted escapes (outside any 15-exponent window)
        w[17 % rows, 4000 % k] = -1e-30
        pl = p22_device_planes(w, rows)
        assert pl is not None and int(pl["erow"][-1]) >= 2
        x = torch.randn(m, k).to(torch.bfloat16)
        x[0].zero_()
        x[0, 100] = 16.0                     # probe on an escape column
        got = emulate_f12(pl, x, rows, k, cfg)
        ref = x.double() @ w.double().T
        rel = float((got.double() - ref).norm() / ref.norm())
        probe = float((got[0].double() - ref[0]).norm() / ref[0].norm())
        assert rel <= 0.5e-2 and probe <= 0.5e-3, (name, m, cfg, rel, probe)
        print(f"ok: F12 {name:8s} m{m:<2d} {cfg}: rel {rel:.2e}, probe row (escape column) {probe:.1e}, "
              f"{int(pl['erow'][-1])} escapes corrected")


def main():
    torch.manual_seed(0)
    # Representative configs per shape (the lab sweeps more); small m and m = 32.
    cases = [
        ("pooler", 1, dict(NT=1, RTW=1, RG=8, KSW=1, D=2, CS=4)),
        ("pooler", 32, dict(NT=4, RTW=1, RG=8, KSW=1, D=2, CS=4)),
        ("qkv_gate", 8, dict(NT=1, RTW=1, RG=9, KSW=1, D=2, CS=8)),
        ("o", 16, dict(NT=2, RTW=2, RG=2, KSW=1, D=2, CS=1)),
        ("unpooler", 32, dict(NT=4, RTW=1, RG=8, KSW=1, D=2, CS=2)),
        ("unpooler_s", 8, dict(NT=1, RTW=1, RG=4, KSW=2, D=2, CS=8)),
        ("router", 1, dict(NT=1, RTW=1, RG=1, KSW=8, D=2, CS=8)),
        ("router", 32, dict(NT=4, RTW=1, RG=1, KSW=4, D=2, CS=8)),
    ]
    # Emulating every CTA at full size is slow on the CPU, so the bigger layers emulate one or two
    # row blocks (the index formulas don't depend on which block) on the full K.
    for name, m, cfg in cases:
        n_full, k = SHAPES[name]
        problems = check_config(n_full, k, cfg)
        assert not problems, (name, cfg, problems)
        hilo = name == "router"
        rows = cfg["RG"] * cfg["RTW"] * 16
        n_rows = min(n_full, rows if k >= 8192 else 2 * rows)
        w = (torch.randn(n_full, k) / math.sqrt(k))
        w = w if hilo else w.to(torch.bfloat16)
        wpk = (p9_pack_hilo(w) if hilo else p9_pack(w))[: n_rows // 16]
        x = torch.randn(m, k).to(torch.bfloat16)
        x[0].zero_()
        x[0, 77] = 32.0                                   # a probe row
        got = emulate(wpk, x, n_rows, k, cfg, hilo)
        ref = x.double() @ w[:n_rows].double().T
        rel = float((got.double() - ref).norm() / ref.norm())
        probe = float((got[0].double() - ref[0]).norm() / ref[0].norm())
        tol = 1e-4 if hilo else 1e-2
        assert rel <= 0.5 * tol and probe <= 0.5e-3, (name, m, cfg, rel, probe)
        print(f"ok: {name:10s} m{m:<2d} {cfg}: rel {rel:.2e}, probe row {probe:.1e}, "
              f"smem {p9_smem_bytes(cfg['NT'], cfg['RTW'], cfg['RG'], cfg['KSW'], cfg['CS'], k) // 1024} KB")


if __name__ == "__main__":
    main()
    main_f12()

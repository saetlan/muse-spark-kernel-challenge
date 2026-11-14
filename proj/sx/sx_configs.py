"""SX configs per (layer, m): instance tuples (NT8, TG, RTW, RGN, KSW, NS, D, XS, F12, CS).

Sizing (H200: 132 SMs, 64K registers and 227 KB shared memory per SM):
  - MT = 8 NT8 TG covers the m tokens; K = 64 NS KSW CS is the layer's K; ROWS = 16 RTW RGN divides N.
  - CTAs = N / ROWS * CS, at most what one wave holds (132 x CTAs per SM, from registers and shared memory).
  - x traffic from L2 per CTA is MT / (0.75 ROWS) of its 12-bit weight traffic: ROWS >= 64 at m = 32.
  - m = 32 is bandwidth-bound (16 warps per SM, ~96 KB of weights in flight per SM); m = 128/256 are
    mma.sync-bound (~495 TF), so their CTAs reuse each x fragment across 2 row tiles (RTW 2).
  - XS >= NS: x resident (staged once, no barrier per chunk); else an XS-stage cp.async pipeline.
"""

MS = (32, 128, 256)
FULL = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}

T = True
CANDIDATES = {
    # m = 32: NT8 4, one token group
    ("pooler", 32): [(4, 1, 2, 4, 4, 8, 2, 8, T, 4), (4, 1, 2, 4, 4, 8, 2, 3, T, 4), (4, 1, 2, 2, 4, 8, 2, 3, T, 4)],
    ("qkv_gate", 32): [(4, 1, 2, 2, 4, 8, 2, 3, T, 4), (4, 1, 3, 3, 4, 4, 2, 4, T, 8)],
    ("unpooler", 32): [(4, 1, 2, 4, 4, 8, 2, 8, T, 2), (4, 1, 2, 2, 4, 16, 2, 3, T, 1), (4, 1, 2, 2, 4, 8, 2, 3, T, 2)],
    ("o", 32): [(4, 1, 2, 4, 2, 4, 2, 4, T, 2), (4, 1, 2, 2, 4, 4, 2, 4, T, 1), (4, 1, 1, 4, 4, 4, 2, 4, T, 1)],
    # m = 128: one token group of 16 n8 tiles, RTW 2
    ("pooler", 128): [(16, 1, 2, 4, 1, 16, 2, 4, T, 8), (16, 1, 2, 4, 2, 16, 2, 3, T, 4)],
    ("qkv_gate", 128): [(16, 1, 2, 4, 1, 16, 2, 4, T, 8)],
    ("unpooler", 128): [(16, 1, 2, 4, 1, 16, 2, 4, T, 4), (16, 1, 2, 4, 2, 16, 2, 3, T, 2)],
    ("o", 128): [(16, 1, 2, 4, 2, 4, 2, 4, T, 2), (16, 1, 1, 4, 2, 8, 2, 3, T, 1)],
    # m = 256: two token groups of 16 n8 tiles (pooler/unpooler stay on cuBLAS: mma.sync-bound above it)
    ("qkv_gate", 256): [(16, 2, 1, 4, 1, 32, 2, 3, T, 4), (16, 2, 2, 2, 1, 32, 2, 3, T, 4)],
    ("o", 256): [(16, 2, 1, 4, 1, 8, 2, 3, T, 2)],
}


def dims(inst):
    nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = inst
    return {"MT": 8 * nt8 * tg, "ROWS": 16 * rtw * rgn, "K": 64 * ns * ksw * cs, "NTH": 32 * rgn * tg * ksw}


def n_rows(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else FULL[layer][0]


def check():
    import sx
    for (layer, m), insts in CANDIDATES.items():
        n, k = n_rows(layer, m), FULL[layer][1]
        for inst in insts:
            dm = dims(inst)
            assert dm["MT"] >= m and dm["K"] == k and n % dm["ROWS"] == 0 and dm["NTH"] <= 512, (layer, m, inst, dm)
            assert dm["MT"] % inst[9] == 0 and inst[9] <= 8 and sx.sx_smem(inst) <= 227 * 1024, (layer, m, inst)
            assert inst[0] == 1 or inst[0] % 2 == 0


def config(inst):
    keys = ("NT8", "TG", "RTW", "RGN", "KSW", "NS", "D", "XS", "F12", "CS")
    return dict(kind="sx", **dict(zip(keys, inst)))


def instance(c):
    return (c["NT8"], c["TG"], c["RTW"], c["RGN"], c["KSW"], c["NS"], c["D"], c["XS"], c["F12"], c["CS"])


LAB = tuple(dict.fromkeys(inst for insts in CANDIDATES.values() for inst in insts))
NONE = ()


# SXW (wgmma RS) instances (NB, WGS, KSW, NS, P, CS): N = 128 NB tokens, ROWS = 64 WGS, NTH = 128 WGS KSW,
# K = 64 NS KSW CS; x slots P + 2, weight ring P + 1 per warp. CTAs <= 132 x CTAs per SM (shared memory).
W_CANDIDATES = {
    ("pooler", 128): [(1, 2, 1, 32, 4, 4), (1, 1, 1, 32, 2, 4)],
    ("qkv_gate", 128): [(1, 1, 1, 32, 2, 4)],
    ("unpooler", 128): [(1, 2, 1, 32, 4, 2)],
    ("o", 128): [(1, 2, 1, 8, 4, 2), (1, 1, 1, 16, 4, 1)],
    ("pooler", 256): [(2, 2, 1, 32, 3, 4)],
    ("unpooler", 256): [(2, 2, 1, 32, 3, 2)],
    ("qkv_gate", 256): [(2, 1, 1, 32, 1, 4)],
    ("o", 256): [(2, 2, 1, 8, 3, 2)],
}


def wdims(inst):
    nb, wgs, ksw, ns, pp, cs = inst
    return {"N": 128 * nb, "ROWS": 64 * wgs, "K": 64 * ns * ksw * cs, "NTH": 128 * wgs * ksw}


def w_check():
    import sx
    for (layer, m), insts in W_CANDIDATES.items():
        n, k = n_rows(layer, m), FULL[layer][1]
        for inst in insts:
            dm = wdims(inst)
            assert dm["N"] >= m and dm["K"] == k and n % dm["ROWS"] == 0 and dm["NTH"] <= 512, (layer, m, inst, dm)
            assert dm["N"] % inst[5] == 0 and inst[5] <= 8 and sx.sxw_smem(inst) <= 227 * 1024, (layer, m, inst, sx.sxw_smem(inst))


def wconfig(inst):
    keys = ("NB", "WGS", "KSW", "NS", "P", "CS")
    return dict(kind="sx", W=True, F12=True, **dict(zip(keys, inst)))


def winstance(c):
    return (c["NB"], c["WGS"], c["KSW"], c["NS"], c["P"], c["CS"])


W_LAB = tuple(dict.fromkeys(inst for insts in W_CANDIDATES.values() for inst in insts))

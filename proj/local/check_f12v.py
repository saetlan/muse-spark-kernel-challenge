"""Exactness of F12V (12-bit weights, m = 1 CUDA-core GEMV), checked on the CUDA text itself:
the decode, x conversion, scale and index helpers are parsed out of proj/f12v/f12v.py's F12V_SRC and
executed in Python (C operators on uint32, __byte_perm and __funnelshift_l emulated). Checks:
  1. decode: every (sign, code, mantissa) at every position of a group of 8 gives exactly
     s << 31 | (112 + c) << 23 | m << 16, with no cross-talk between positions;
  2. scale and w_dec helpers are exactly 2^(base - 112) and 2^(base - 127) for every base;
  3. encode -> decode -> unscale -> escapes rebuilds every weight bit for bit: all 65,536 BF16 patterns
     (NaN, Inf, zeros, subnormals) and Gaussian weights at the real shapes (escape rate, max per row);
     the record path accepts 7 escapes per row and falls back at 8;
  4. kernel emulation for every lab config: each weight slot and each x group read exactly once, the
     ring hands step s its own data, every output row written once, and the result equals FP64
     x @ W.T (to fp64 rounding) on normal inputs, exactly on probe inputs (escape columns included).

Run: .venv/bin/python proj/local/check_f12v.py
"""

import math
import re
import sys
from pathlib import Path

import torch

F12V = Path(__file__).resolve().parents[1] / "f12v"
sys.path.insert(0, str(F12V))
from f12v_layout import (F12V_SLOTS, M32, f12v_device_planes, f12v_encode, f12v_records,  # noqa: E402
                         f12v_words)

SRC = None


def prmt(a, b, sel):
    """__byte_perm(a, b, sel): byte n = byte (sel nibble n & 7) of {b:a}."""
    src = (b << 32) | a
    out = a * 0 if isinstance(a, torch.Tensor) else 0
    for n in range(4):
        idx = (sel >> (4 * n)) & 0x7
        out = out | (((src >> (8 * idx)) & 0xFF) << (8 * n))
    return out


def funnel_l(lo, hi, shift):
    """__funnelshift_l(lo, hi, shift): the high 32 bits of (hi:lo) << (shift & 31)."""
    sh = shift & 31
    return ((((hi & M32) << 32) | (lo & M32)) << sh >> 32) & M32


def c_expr(expr):
    expr = re.sub(r"0x([0-9A-Fa-f]+)u", lambda mo: str(int(mo.group(1), 16)), expr)
    expr = re.sub(r"\b(\d+)u\b", r"\1", expr)
    expr = re.sub(r"(?<!/)/(?!/)", "//", expr)     # integer helpers only: C division of non-negative ints
    return (expr.replace("__byte_perm", "prmt").replace("__funnelshift_l", "funnel_l")
            .replace("__uint_as_float", ""))


def body_of(name):
    src = SRC
    i = src.index(f" {name}(")
    open_ = src.index("{", i)
    depth, j = 0, open_
    while True:
        depth += src[j] == "{"
        depth -= src[j] == "}"
        if depth == 0:
            break
        j += 1
    sig = src[i + 1:open_]
    params = [p.strip().split()[-1].replace("(&", "").split(")")[0].lstrip("&") for p in sig[sig.index("(") + 1:sig.rindex(")")].split(",")]
    return params, src[open_ + 1:j]


def top_level_split(text):
    """Split at commas outside parentheses."""
    depth, parts, cur = 0, [], ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def parse(name):
    """A CUDA helper as a Python function. Values are uint32 bit patterns (floats stay as bits)."""
    params, body = body_of(name)
    stmts = [st.strip() for st in body.split(";") if st.strip()]
    py, outs = [], []
    for st in stmts:
        if st.startswith("return "):
            py.append(f"return ({c_expr(st[len('return '):])})")
            continue
        if st.startswith("const uint32_t "):
            for decl in top_level_split(st[len("const uint32_t "):]):
                lhs, rhs = decl.split("=", 1)
                py.append(f"{lhs.strip()} = ({c_expr(rhs.strip())}) & M32")
            continue
        lhs, rhs = st.split("=", 1)
        lhs = lhs.strip()
        py.append(f"{lhs} = ({c_expr(rhs.strip())}) & M32")
        outs.append(lhs.split("[")[0])
    ins = [p for p in params if p not in outs]
    outs = list(dict.fromkeys(outs))
    pre = [f"{o} = [None] * 8" for o in outs if any(o + "[" in line for line in py)]
    if outs:
        py.append("return " + ", ".join(outs))
    code = f"def f({', '.join(ins)}):\n" + "\n".join("    " + line for line in pre + py)
    env = {"prmt": prmt, "funnel_l": funnel_l, "M32": M32, "make_uint4": lambda *v: tuple(e & M32 for e in v)}
    exec(code, env)
    return env["f"]


def load_src():
    global SRC
    import f12v
    SRC = f12v.F12V_SRC
    return f12v


def u32(t):
    return t.to(torch.int64) & M32


def bits_to_f64(b):
    return u32(b).to(torch.int64).where(u32(b) < 2**31, u32(b) - 2**32).to(torch.int32).view(torch.float32).double()


def check_decode(dec8):
    g = torch.Generator().manual_seed(7)
    n = 2 * 16 * 128
    combo = torch.arange(n)
    for pos in range(8):
        s = torch.randint(0, 2, (n, 8), generator=g)
        c = torch.randint(0, 16, (n, 8), generator=g)
        m = torch.randint(0, 128, (n, 8), generator=g)
        s[:, pos], c[:, pos], m[:, pos] = combo >> 11, (combo >> 7) & 15, combo & 127
        q, l0, l1 = f12v_words(s, c, m)
        f = dec8(q, l0, l1)
        for i in range(8):
            want = (s[:, i] << 31) | ((112 + c[:, i]) << 23) | (m[:, i] << 16)
            assert torch.equal(u32(f[i]), want), f"decode wrong at position {i} (varying {pos})"
    print("ok: CUDA f12v_decode8 (parsed) exact for every sign/code/mantissa at every position, no cross-talk")


def check_scale(scale, wdec):
    for base in range(1, 240):
        assert struct_f32(scale(base)) == 2.0 ** (base - 112), base
        assert struct_f32(wdec(base)) == 2.0 ** (base - 127), base
    print("ok: CUDA f12v_scale / f12v_wdec (parsed) are 2^(base-112) / 2^(base-127) for base 1..239")


def struct_f32(bits):
    import struct
    return struct.unpack("<f", struct.pack("<I", bits & M32))[0]


def reconstruct(enc, dec8, n, k):
    """W bits [n, k] from the planes through the parsed decode, the unscale and the escape list."""
    p = enc["p"]                                         # [n, kb, 3, 32, 4] int64 words
    q = p[:, :, 0]                                       # [n, kb, 32, 4]
    lw = torch.stack((p[:, :, 1, :, 0], p[:, :, 1, :, 1], p[:, :, 1, :, 2], p[:, :, 1, :, 3],
                      p[:, :, 2, :, 0], p[:, :, 2, :, 1], p[:, :, 2, :, 2], p[:, :, 2, :, 3]), dim=-1)
    l0, l1 = lw[..., 0::2], lw[..., 1::2]                # [n, kb, 32, 4]: group g's L0, L1
    f = torch.stack(dec8(q, l0, l1), dim=-1)             # [n, kb, 32, 4, 8] FP32 bits
    f = u32(f)
    assert int((f & 0xFFFF).abs().sum()) == 0, "low half not zero"
    base = enc["base"].view(n, 1, 1, 1, 1)
    e = ((f >> 23) & 0xFF) - 112 + base
    bits = ((f >> 31) << 15) | (e << 7) | ((f >> 16) & 0x7F)
    bits = bits.reshape(n, k)
    bits[enc["esc_row"], enc["esc_k"]] = enc["esc_bits"]
    return bits


def check_roundtrip(dec8):
    allbits = torch.arange(65536, dtype=torch.int64).to(torch.int16)
    perm = torch.randperm(65536, generator=torch.Generator().manual_seed(0))
    w = allbits[perm].view(torch.bfloat16).view(16, 4096)
    enc = f12v_encode(w)
    want = w.view(torch.int16).to(torch.int64) & 0xFFFF
    assert torch.equal(reconstruct(enc, dec8, 16, 4096), want), "all-patterns reconstruction differs"
    print(f"ok: all 65,536 BF16 bit patterns round-trip exactly (escape rate {enc['n_esc'] / enc['n']:.3f} on this adversarial tensor)")
    # the record path: 7 escapes in a row are accepted, 8 fall back
    torch.manual_seed(5)
    w = (torch.randn(4, 1024) / 32).to(torch.bfloat16)
    w[0, :8] = 0
    w[1, 100:107] = 0
    enc = f12v_encode(w)
    assert f12v_records(enc, 4) is None and f12v_device_planes(w) is None
    w[0, 7] = w[0, 8]
    assert f12v_device_planes(w) is not None
    print("ok: records take 7 escapes per row and fall back at 8")
    torch.manual_seed(1)
    for name, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}.items():
        w = (torch.randn(n, k) / math.sqrt(k)).to(torch.bfloat16)
        enc = f12v_encode(w)
        want = w.view(torch.int16).to(torch.int64) & 0xFFFF
        assert torch.equal(reconstruct(enc, dec8, n, k), want), name
        pl = f12v_device_planes(w)
        assert pl is not None, name
        print(f"ok: {name:8s} Gaussian round-trip exact; escapes {pl['rate'] * 100:.4f}% (max {pl['max_row']} per row) "
              f"-> {12 + 32 * pl['rate']:.3f} bits/weight")


def kernel_text_checks():
    """The ring statements the emulation mirrors must be exactly these in the kernel."""
    for stmt in ("for (int s = 0; s < D; ++s) load(s, s);", "const int slot = s % D;",
                 "if (s + D < NS) load(slot, s + D);", "const int rg = warp / KSW, ks = warp % KSW;",
                 "const bool fin = tid < RB;", "const uint32_t base = rc0.x & 0xFFu, count = (rc0.x >> 8) & 0xFFu;",
                 "const uint32_t esc[7] = {rc0.y, rc0.z, rc0.w, rc1.x, rc1.y, rc1.z, rc1.w};",
                 "for (int p = 0; p < KSW; ++p) sum += red[p][tid];",
                 "if (lane == 0) red[ks][rg * R + r] = v;",
                 "const float xk = __uint_as_float(static_cast<uint32_t>(xh[esc[j] & 0xFFFFu]) << 16);",
                 "sum = fmaf(-xk, wdec, sum);", "sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);",
                 "out[f12v_row(bx, 0, tid, RB, R)] = __float2bfloat16_rn(sum);",
                 "ld_x(rec + f12v_ridx(row, 0), rc0);", "ld_x(rec + f12v_ridx(row, 1), rc1);",
                 "const long long row = f12v_row(bx, 0, tid, RB, R);",
                 "for (int j = 0; j < 3; ++j) ld_w(wp + f12v_widx(row, KBS, kb, j, lane), wv[slot][r][j]);",
                 "for (int g = 0; g < 4; ++g) ld_x(x + f12v_xidx(kb, lane, g), xv[slot][g]);",
                 "for (int g = 0; g < 4; ++g) xs[f12v_xidx(f12v_kb(ks, s, NS), lane, g)] = xv[slot][g];",
                 "const long long row = f12v_row(bx, rg, r, RB, R);", "const int kb = f12v_kb(ks, s, NS);",
                 "for (int t = 0; t < 4; ++t) f12v_xpair(word(xv[slot][g], t), xf[2 * t], xf[2 * t + 1]);",
                 "const uint4& lv = wv[slot][r][f12v_lslot(g)];",
                 "f12v_decode8(word(wv[slot][r][0], g), word(lv, f12v_lword(g, 0)), word(lv, f12v_lword(g, 1)), f);",
                 "for (int i = 0; i < 8; ++i) acc[r][g] = fmaf(xf[i], f[i], acc[r][g]);",
                 "if (rg == 0) {"):
        assert SRC.count(stmt) >= 1, f"kernel text changed, update the emulation: {stmt}"


def emulate(inst, planes, x, n_rows, h):
    """The kernel per its text (h: parsed helpers), in float64. Returns out [n_rows] float64."""
    r_, rg_, ksw, ns, d = inst
    rb, kbs = r_ * rg_, ns * ksw
    k = 1024 * kbs
    grid = n_rows // rb
    assert n_rows % rb == 0
    # ring: step s computes on the slot that holds step s
    ring = {}
    for s in range(d):
        ring[s] = s
    for s in range(ns):
        slot = s % d
        assert ring[slot] == s, (inst, s)
        if s + d < ns:
            ring[slot] = s + d
    ar = torch.arange
    bx = ar(grid).view(-1, 1, 1, 1, 1, 1)
    rg = ar(rg_).view(1, -1, 1, 1, 1, 1)
    r = ar(r_).view(1, 1, -1, 1, 1, 1)
    ks = ar(ksw).view(1, 1, 1, -1, 1, 1)
    s = ar(ns).view(1, 1, 1, 1, -1, 1)
    lane = ar(32).view(1, 1, 1, 1, 1, -1)
    row = h["row"](bx, rg, r, rb, r_).expand(grid, rg_, r_, ksw, ns, 32)
    kb = h["kb"](ks, s, ns).expand_as(row)
    lanes = lane.expand_as(row)
    p4 = u32(planes["p"].view(-1, 4))
    widx = [h["widx"](row, kbs, kb, j, lanes) for j in range(3)]
    reads = torch.cat([wi.flatten() for wi in widx])
    assert torch.equal(torch.sort(reads).values, ar(n_rows * kbs * 96)), (inst, "weight slots not read exactly once")
    xw = u32(x.contiguous().view(torch.int32)).view(-1, 4)       # x as uint4 words
    gs = ar(4)
    xidx = torch.stack([h["xidx"](kb, lanes, g) for g in range(4)], dim=-1)   # [..., 4]
    xs = torch.full_like(xw, -1)
    rg0 = (rg.expand_as(row) == 0)
    xs[xidx[rg0].flatten()] = xw[xidx[rg0].flatten()]
    assert torch.equal(xs, xw), (inst, "x copy in shared memory incomplete")
    for rgv in range(rg_):   # every warp reads all of its K range of x
        sel = xidx[:, rgv].flatten()
        assert torch.equal(torch.sort(sel).values, ar(k // 8).repeat_interleave(grid * r_)), (inst, "x reads")
    acc = torch.zeros(n_rows, dtype=torch.float64)
    for g in range(4):
        q = p4[widx[0].flatten(), g]
        lv = p4[widx[h["lslot"](g)].flatten()]
        l0, l1 = lv[:, h["lword"](g, 0)], lv[:, h["lword"](g, 1)]
        f = h["dec8"](q, l0, l1)
        xv = xw[xidx[..., g].flatten()]
        xf = []
        for t in range(4):
            lo, hi = h["xpair"](xv[:, t])
            xf += [lo, hi]
        part = sum(bits_to_f64(xf[i]) * bits_to_f64(f[i]) for i in range(8))
        acc.index_add_(0, row.flatten(), part)
    out_rows = h["row"](ar(grid).view(-1, 1), 0, ar(rb).view(1, -1), rb, r_).flatten()
    assert torch.equal(torch.sort(out_rows).values, ar(n_rows)), (inst, "output rows not written exactly once")
    rec4 = u32(planes["rec"].view(-1, 4))
    rc0, rc1 = rec4[h["ridx"](out_rows, 0)], rec4[h["ridx"](out_rows, 1)]
    base, count = rc0[:, 0] & 0xFF, (rc0[:, 0] >> 8) & 0xFF
    scale = torch.tensor([struct_f32(h["scale"](int(b))) for b in base], dtype=torch.float64)
    wdec = torch.tensor([struct_f32(h["wdec"](int(b))) for b in base], dtype=torch.float64)
    total = acc[out_rows] * scale
    esc = torch.stack((rc0[:, 1], rc0[:, 2], rc0[:, 3], rc1[:, 0], rc1[:, 1], rc1[:, 2], rc1[:, 3]), dim=-1)
    xh = u32(xs.reshape(-1).to(torch.int64)).view(-1)
    xbits = torch.stack(((xh & 0xFFFF), (xh >> 16)), dim=-1).flatten()   # bf16 bits by k
    for j in range(F12V_SLOTS):
        on = j < count
        kk = esc[:, j] & 0xFFFF
        xk = bits_to_f64(xbits[kk] << 16)
        wt = bits_to_f64(esc[:, j] & 0xFFFF0000)
        total = torch.where(on, total - xk * wdec + xk * wt, total)
    out = torch.empty(n_rows, dtype=torch.float64)
    out[out_rows] = total
    return out


def helpers():
    """The parsed CUDA helpers the emulation uses (also the dry runs' mock kernel)."""
    if SRC is None:
        load_src()
    kernel_text_checks()
    return {"row": parse("f12v_row"), "kb": parse("f12v_kb"), "widx": parse("f12v_widx"), "xidx": parse("f12v_xidx"),
            "ridx": parse("f12v_ridx"), "lslot": parse("f12v_lslot"), "lword": parse("f12v_lword"),
            "dec8": parse("f12v_decode8"), "xpair": parse("f12v_xpair"), "scale": parse("f12v_scale"),
            "wdec": parse("f12v_wdec")}


def check_kernel(f12v):
    sys.path.insert(0, str(F12V))
    import f12v_configs
    h = helpers()
    torch.manual_seed(3)
    full = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    for layer, (n_all, k) in full.items():
        w = (torch.randn(n_all, k) / math.sqrt(k)).to(torch.bfloat16)
        # planted escapes: zeros, a huge and a tiny weight, 5 in one row, two in column 77
        w[0, :5] = 0
        w[1, 3] = 1e4
        w[2, 77] = 1e-30
        w[5, 77] = 0
        w[n_all // 2 + 1, k - 1] = -3e3
        planes = f12v_device_planes(w)
        assert planes is not None, layer
        n_rows = f12v_configs.ROWS_M1[layer][0]
        wd = w[:n_rows].double()
        gen = torch.Generator().manual_seed(11)
        x_norm = torch.randn(1, k, generator=gen).to(torch.bfloat16)
        x_probe = torch.zeros(1, k, dtype=torch.bfloat16)
        x_probe[0, 77] = -32.0
        insts = sorted({f12v_configs.instance(c, layer) for c in f12v_configs.candidates(layer)})
        for inst in insts:
            for x, exact in ((x_norm, False), (x_probe, True)):
                got = emulate(inst, planes, x.view(-1), n_rows, h)
                ref = (x.double() @ wd.T).view(-1)
                if exact:
                    assert torch.equal(got, ref), (layer, inst, "probe row not exact")
                else:
                    scale = (x.double().abs() @ wd.abs().T).view(-1)
                    err = float(((got - ref).abs() / scale.clamp(min=1e-30)).max())
                    assert err < 1e-12, (layer, inst, err)
        print(f"ok: {layer:8s} kernel emulation exact for {len(insts)} configs (normal: fp64 rounding only; probe: bit exact)")


def main():
    f12v = load_src()
    check_decode(parse("f12v_decode8"))
    check_scale(parse("f12v_scale"), parse("f12v_wdec"))
    check_roundtrip(parse("f12v_decode8"))
    check_kernel(f12v)


if __name__ == "__main__":
    main()

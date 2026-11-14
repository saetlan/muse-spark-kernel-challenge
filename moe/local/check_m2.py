"""Local (CPU) checks for M2 (moe/m2/m2.py, built into submission_m2.py).

Triton can't run on a Mac, so these emulate the kernels in PyTorch with the kernels' own index
math, and check it against an FP32 reference from nvfp4.dequantize:
1. the decode asm (prmt + Marlin's shifts + mul.rn.bf16x2), bit for bit, on every byte and every
   E4M3 scale, element by element as Triton packs them (pack=4);
2. m2_repack + the tile addressing: every weight element recovered exactly;
3. the pair bookkeeping (active experts, items, persistent loop, 16-pair chunks): every local pair
   computed once per column tile and K split, nothing else written;
4. the full M2 pipeline (bf16 act, bf16 y or split partials, post-norm) vs the FP32 reference.

Run: .venv/bin/python moe/local/check_m2.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_v0_lab as base  # noqa: E402

TOP_K = 8


def install_stubs():
    base.stub_modules()

    class _Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            raise RuntimeError("Triton kernels can't run locally")

    def jit(fn=None, **_):
        return _Kernel(fn) if fn is not None else _Kernel

    sys.modules["triton"].jit = jit


# ---------------------------------------------------------------------------- 1. decode asm
def _u32(x):
    return x & 0xFFFFFFFF


def dequant_asm_text(path):
    """The asm string literal of _m2_dequant's tl.inline_asm_elementwise call, from the source."""
    import ast
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_m2_dequant")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "inline_asm_elementwise")
    return next(k.value.value for k in call.keywords if k.arg == "asm")


def _bf16_pair(reg32):
    lo16 = (reg32 & 0xFFFF).to(torch.int32).to(torch.int16)
    hi16 = ((reg32 >> 16) & 0xFFFF).to(torch.int32).to(torch.int16)
    return torch.stack((lo16, hi16), -1).view(torch.bfloat16)  # [n, 2]: low lane, high lane


def _pair_bits(bf2):
    bits = bf2.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return bits[:, 0] | (bits[:, 1] << 16)


def run_ptx(asm, regs):
    """Interpret the few PTX instructions the decode uses, on int64 tensors holding 32-bit values.
    regs maps "$N" (and named registers as they're written) to values."""
    def val(tok):
        tok = tok.strip()
        if tok in regs:
            return regs[tok]
        return int(tok, 0)

    for line in asm.replace("{", ";").replace("}", ";").split(";"):
        line = line.strip()
        if not line or line.startswith(".reg"):
            continue
        op, args = line.split(None, 1)
        a = [t.strip() for t in args.split(",")]
        if op == "prmt.b32":  # generic mode: result byte i <- byte c[4i+3:4i] of {b, a}
            src = val(a[1]) | (val(a[2]) << 32)
            ctl = val(a[3])
            out = 0
            for i in range(4):
                sel = (ctl >> (4 * i)) & 0x7
                out = out | (((src >> (8 * sel)) & 0xFF) << (8 * i))
            regs[a[0]] = out
        elif op == "and.b32":
            regs[a[0]] = val(a[1]) & val(a[2])
        elif op == "or.b32":
            regs[a[0]] = val(a[1]) | val(a[2])
        elif op == "shr.b32":
            regs[a[0]] = _u32(val(a[1])) >> val(a[2])
        elif op == "shl.b32":
            regs[a[0]] = _u32(val(a[1]) << val(a[2]))
        elif op == "mul.rn.bf16x2":  # exact product of two bf16 fits fp32, then one rounding
            prod = (_bf16_pair(val(a[1])).float() * _bf16_pair(val(a[2])).float()).to(torch.bfloat16)
            regs[a[0]] = _pair_bits(prod)
        else:
            raise SystemExit(f"run_ptx: unsupported instruction {op!r}")
    return regs


def emulate_m2_dequant(asm, q4, s_lo4, s_hi4):
    """_m2_dequant's asm on one pack, operands numbered as Triton 3.8 packs them (outputs, then
    inputs, each in 32-bit registers, element 0 in the low bits): q4 uint8 [n, 4] -> $4;
    s_lo4, s_hi4 bf16 [n, 4] -> $5,$6 / $7,$8; returns (lo, hi) bf16 [n, 4] from $0,$1 / $2,$3."""
    b = q4.to(torch.int64)
    regs = {"$4": b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24),
            "$5": _pair_bits(s_lo4[:, 0:2]), "$6": _pair_bits(s_lo4[:, 2:4]),
            "$7": _pair_bits(s_hi4[:, 0:2]), "$8": _pair_bits(s_hi4[:, 2:4])}
    regs = run_ptx(asm, regs)
    lo = torch.cat((_bf16_pair(regs["$0"]), _bf16_pair(regs["$1"])), -1)
    hi = torch.cat((_bf16_pair(regs["$2"]), _bf16_pair(regs["$3"])), -1)
    return lo, hi


def check_dequant_asm(source):
    from nvfp4 import unpack_e2m1
    asm = dequant_asm_text(source)
    scales = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    s_bf = (scales * 2.0**64).to(torch.bfloat16)
    assert torch.equal(s_bf.float(), scales * 2.0**64), "scale * 2^64 must be exact in bf16"
    # Every byte value at every position of a pack, against every scale.
    byte = torch.arange(256, dtype=torch.int64)
    q = torch.stack([(byte + 37 * i) % 256 for i in range(4)], -1).to(torch.uint8)  # [256, 4]
    q = q.repeat_interleave(127, 0)
    si = torch.arange(127).repeat(256)
    s_lo = torch.stack([s_bf[(si + 5 * i) % 127] for i in range(4)], -1)
    s_hi = torch.stack([s_bf[(si + 11 * i + 3) % 127] for i in range(4)], -1)
    lo, hi = emulate_m2_dequant(asm, q, s_lo, s_hi)
    codes = unpack_e2m1(q.reshape(-1)).view(-1, 4, 2)  # [n, 4, (lo, hi)]
    want_lo = codes[..., 0] * s_lo.float() * 2.0**-126
    want_hi = codes[..., 1] * s_hi.float() * 2.0**-126
    assert torch.equal(lo.float(), want_lo) and torch.equal(hi.float(), want_hi), "decode asm mismatch"
    assert (lo.float()[want_lo != 0].abs() >= 2.0**-126).all(), "a decoded product fell to a bf16 subnormal"
    print(f"ok: _m2_dequant asm in {Path(source).name} (interpreted bit for bit) = code * scale * 2^-62, "
          "every byte, position and scale")


# ------------------------------------------------------------------ 2-4. emulated M2 pipeline
def emu_weights(bank, name, bk, e, rows):
    """Dequantize rows of local expert e from M2's repacked layout with the kernel's addressing:
    tile k0, byte j: low nibble -> k0 + j, high nibble -> k0 + bk/2 + j; scale group k // 16."""
    from nvfp4 import unpack_e2m1
    w = bank[(name, bk)][e][rows]  # [r, K/2]
    sf = bank[name + "_sf"][e][rows].float()  # [r, K/16]
    k = w.shape[1] * 2
    hb = bk // 2
    codes = unpack_e2m1(w).view(w.shape[0], k // bk, hb, 2)  # [r, tiles, j, (lo, hi)]
    vals = torch.cat((codes[..., 0], codes[..., 1]), -1).reshape(w.shape[0], k)  # k order
    s = (sf * 2.0**64).to(torch.bfloat16).float()[:, torch.arange(k) // 16]
    return ((vals * 2.0**-126).to(torch.bfloat16).float() * s).to(torch.bfloat16).float()  # code * scale * 2^-62


def emu_gemm(a, bank, name, ids_local, n_pairs, gated, a_per_pair, cfg, sms, cover):
    """The kernel's work split, item by item; returns what it writes: {pair: row} per split."""
    w_key = (name, cfg["bk"])
    rows_total, k = bank[w_key].shape[1], bank[w_key].shape[2] * 2
    n = rows_total // 2 if gated else rows_total
    local = bank["count"]
    bn, split, tok = cfg["bn"], cfg["split"], 16
    items = (n // bn) * split
    counts = torch.stack([(ids_local == e).sum() for e in range(local)])
    active = (counts > 0).nonzero().flatten().tolist()
    if cfg["persist"]:
        progs = min(local * items, cfg["progs_per_sm"] * sms)
        work = [(it // items, it % items) for p in range(progs) for it in range(p, len(active) * items, progs)]
        work = [(active[a], rem) for a, rem in work]
    else:
        work = [(e, rem) for e in range(local) for rem in range(items) if counts[e] > 0]
    c = bank["c13" if gated else "c2"]
    out = torch.zeros((split, n_pairs, rows_total if split > 1 else n))
    deq = {}
    for e, rem in work:
        pid_n, pid_s = rem // split, rem % split
        cols = torch.arange(pid_n * bn, pid_n * bn + bn)
        if e not in deq:
            r = torch.arange(rows_total)
            deq[e] = emu_weights(bank, name, cfg["bk"], e, r)
        wg = deq[e][cols]
        wu = deq[e][cols + n] if gated else None
        ks = slice(pid_s * (k // split), (pid_s + 1) * (k // split))
        pairs = (ids_local == e).nonzero().flatten()
        for c0 in range(0, len(pairs), tok):
            chunk = pairs[c0:c0 + tok]
            arows = chunk if a_per_pair else chunk // TOP_K
            av = a[arows].float()[:, ks]
            g = av @ wg[:, ks].T * c[e]
            for p in chunk.tolist():
                cover[(p, pid_n, pid_s)] = cover.get((p, pid_n, pid_s), 0) + 1
            if split == 1:
                if gated:
                    u = av @ wu[:, ks].T * c[e]
                    res = g * torch.sigmoid(g) * u
                else:
                    res = g
                out[0, chunk[:, None], cols[None, :]] = res.to(torch.bfloat16).float()
            else:
                out[pid_s, chunk[:, None], cols[None, :]] = g
                if gated:
                    out[pid_s, chunk[:, None], (cols + n)[None, :]] = av @ wu[:, ks].T * c[e]
    return out


def emu_m2(x, ids, tw, bank, norm_w, cfg1, cfg2, sms):
    n_pairs = ids.numel()
    ids_local = ids.reshape(-1).long() - bank["start"]
    local_mask = (ids_local >= 0) & (ids_local < bank["count"])
    ids_local = torch.where(local_mask, ids_local, torch.full_like(ids_local, -1))
    cover1, cover2 = {}, {}
    o1 = emu_gemm(x, bank, "w13", ids_local, n_pairs, True, False, cfg1, sms, cover1)
    inter = bank[("w13", cfg1["bk"])].shape[1] // 2
    if cfg1["split"] == 1:
        act = o1[0]
    else:
        g, u = o1[..., :inter].sum(0), o1[..., inter:].sum(0)
        act = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
    act = act.to(torch.bfloat16)
    o2 = emu_gemm(act, bank, "w2", ids_local, n_pairs, False, True, cfg2, sms, cover2)
    y = o2[0] if cfg2["split"] == 1 else o2.sum(0).to(torch.bfloat16).float()
    for cover, cfg, n in ((cover1, cfg1, inter), (cover2, cfg2, bank[("w2", cfg2["bk"])].shape[1])):
        want = {(p, t, s) for p in local_mask.nonzero().flatten().tolist()
                for t in range(n // cfg["bn"]) for s in range(cfg["split"])}
        assert set(cover) == want and set(cover.values()) == {1}, "pair coverage is wrong"
    out = torch.zeros((x.shape[0], y.shape[1]))
    nw = norm_w.float()
    for p in local_mask.nonzero().flatten().tolist():
        v = y[p] * nw
        out[p // TOP_K] += v * torch.rsqrt(v.pow(2).mean() + 1e-8) * tw.reshape(-1)[p]
    return out.to(torch.bfloat16)


def fp32_reference(x, ids, tw, weights, start, count, post_norm_w):
    from nvfp4 import dequantize
    out = torch.zeros(x.shape, dtype=torch.float32)
    xf = x.float()
    inter = weights["w13"].shape[1] // 2
    for e in torch.unique(ids).tolist():
        if not start <= e < start + count:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xf[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter_v = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        y = inter_v @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        y = y * (1.0 + post_norm_w.float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * tw[tok, slot].unsqueeze(1))
    return out


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def small_weights(experts, hidden, inter, seed):
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * inter, hidden, generator=gen, device="cpu", leading=(experts,))
    w2, w2_sf, gs2 = random_weights(hidden, inter, generator=gen, device="cpu", leading=(experts,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(experts).contiguous(),
            "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(experts).contiguous(),
            "post_norm_w": (0.1 * torch.randn((hidden,), generator=gen)).to(torch.bfloat16)}


def check_pipeline(m2):
    hidden = inter = 512  # same index math as 4096, small enough for the CPU
    experts, start, count = 64, 0, 32
    weights = small_weights(experts, hidden, inter, seed=3)
    bank = m2.m2_bank(weights, start, count, (128, 256))
    norm_w = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)
    for name, bk in (("w13", 128), ("w2", 256)):  # 2. every element through the kernel's addressing
        from nvfp4 import dequantize
        for e in (0, 17, 31):
            got = emu_weights(bank, name, bk, e, torch.arange(bank[(name, bk)].shape[1]))
            want = dequantize(weights[name][e], weights[name + "_sf"][e], 1.0)
            assert torch.equal(got * 2.0**62, want), f"repack/addressing mismatch {name} bk={bk}"
    print("ok: m2_repack + tile addressing recover every weight exactly (bk 128 and 256)")
    gen = torch.Generator().manual_seed(5)
    cfgs = [dict(persist=True, swap=True, bn=64, bk=256, split=1, warps=4, stages=3, progs_per_sm=2),
            dict(persist=False, swap=False, bn=128, bk=128, split=2, warps=4, stages=3, progs_per_sm=2),
            dict(persist=True, swap=False, bn=64, bk=128, split=4, warps=4, stages=3, progs_per_sm=1)]
    worst = 0.0
    for tokens, skew in ((1, False), (3, False), (8, False), (20, False), (32, True)):
        x = torch.randn((tokens, hidden), generator=gen).to(torch.bfloat16)
        if skew:  # every token on experts 0-7: 32 pairs per expert, two 16-pair chunks
            ids = torch.arange(TOP_K, dtype=torch.int32).repeat(tokens, 1)
        else:
            ids = torch.stack([torch.randperm(experts, generator=gen)[:TOP_K] for _ in range(tokens)]).to(torch.int32)
        tw = torch.rand((tokens, TOP_K), generator=gen)
        tw = tw / tw.sum(-1, keepdim=True)
        ref = fp32_reference(x, ids, tw, weights, start, count, weights["post_norm_w"])
        for cfg1, cfg2 in zip(cfgs, cfgs[1:] + cfgs[:1]):
            got = emu_m2(x, ids, tw, bank, norm_w, cfg1, cfg2, sms=4)
            if ref.norm() == 0:
                assert got.float().abs().max() == 0
                continue
            err = rel_l2(got, ref)
            worst = max(worst, err)
            assert err < 1e-2, (tokens, skew, cfg1, cfg2, err)
    print(f"ok: emulated M2 pipeline vs FP32 reference, 5 routings x 3 config pairs: worst rel L2 {worst:.2e} (judge allows 1.5e-2)")


def check_submission_m2():
    mod = base.load("submission_m2")
    assert (base.MOE / "submission_m2.py").read_text().splitlines()[0] == base.PROBLEM_LINE
    for name in ("prepare", "kernel_fn", "_v1_prepare", "_v1_kernel_fn", "m2_repack", "m2_bank"):
        assert callable(getattr(mod, name)), name
    v1 = (base.MOE / "submission.py").read_text()
    v1 = v1.replace("\ndef prepare(weights", "\ndef _v1_prepare(weights", 1)
    v1 = v1.replace("\ndef kernel_fn(data, state", "\ndef _v1_kernel_fn(data, state", 1)
    assert (base.MOE / "submission_m2.py").read_text().startswith(v1.rstrip()), "v1 part must be byte for byte"
    print("ok: submission_m2.py = submission.py (v1, renamed entry points) + M2 section")
    return mod


if __name__ == "__main__":
    install_stubs()
    m2 = check_submission_m2()
    check_dequant_asm(sys.argv[1] if len(sys.argv) > 1 else base.MOE / "submission_m2.py")
    check_pipeline(m2)

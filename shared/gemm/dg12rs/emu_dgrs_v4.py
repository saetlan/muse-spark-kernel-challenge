"""CPU emulation of dgrs.py (no GPU here), with bf16_proj's own F12 encoder (read-only: f12m_encode from
proj/f12m/f12m_layout.py and the F12V helpers from proj/submission_v10.py, pulled out by AST).

1. Pack + decode: dgrs_pack's chunk-major planes read at the kernel's offsets (stage block of row block rb and chunk
   kc, row tile 4c + w, slots +512 / +1024, lane * 16), decoded with a port of f12m_decode, placed at wgmma RS's A
   positions (row 16 rt + g + 8 (reg & 1), k 16 step + 2 t + 8 (reg >> 1) + half) == W * 2^(112 - base) exactly,
   escapes = 2^-15.
2. x tile: the producer's swizzled cp.async destinations read back through the SW128 descriptor rule == x.
3. Whole GEMM per layer (qkv, o, pooler, unpooler at m = 32 / 128 / 256 with the split counts below): per CTA slice
   fp32 partials, last arriver sums slices in order, F12M epilogue (scale, escapes from rec, x from global), BF16;
   vs FP32 x @ W^T: relative error and worst row.
4. v3 split-K finalize map: every (row, token) exactly once over cluster ranks and threads, S = 1..8.
5. Pins: the CUDA lines these ports mirror.

Run: .venv/bin/python shared/gemm/dg12rs/emu_dgrs.py [quick]
"""

import ast
import random
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"

# ---- bf16_proj's encoder, read-only
f12 = {"torch": torch, "M32": 0xFFFFFFFF}
tree = ast.parse((ROOT / "proj" / "submission_v10.py").read_text())
want = {"f12v_bases", "f12v_words", "as_int32", "f12v_records"}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module([node], []), "submission_v10.py", "exec"), f12)
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("F12V_SLOTS", "M32"):
        exec(compile(ast.Module([node], []), "submission_v10.py", "exec"), f12)
assert want <= set(f12), set(f12)
lay = (ROOT / "proj" / "f12m" / "f12m_layout.py").read_text()
lay = lay.replace("from f12v_layout import as_int32, f12v_bases, f12v_records, f12v_words\n", "")
lay = lay.replace("from p9_layout import k_perm, p9_pack\n", "")
exec(compile(lay, "f12m_layout.py", "exec"), f12)

sys.modules.setdefault("torch.utils.cpp_extension", types.SimpleNamespace(load_inline=None))
dg = {"__name__": "dgrs_emu"}
exec(compile((HERE / "dgrs.py").read_text().replace("from torch.utils.cpp_extension import load_inline",
                                                     "load_inline = None"), "dgrs.py", "exec"), dg)
CUDA = dg["dgrs_source"]()

# ---------------------------------------------------------------------------------------------- 5. pins
PINS = [
    "  const uint32_t r4 = __funnelshift_l(q, q, 4);",
    "  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;",
    "  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;",
    "  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),",
    "                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));",
    "  const uint4 l01 = *reinterpret_cast<const uint4*>(sw + woff + 512);",
    "  const uint4 l23 = *reinterpret_cast<const uint4*>(sw + woff + 1024);",
    "    const uint4& l = j < 2 ? l01 : l23;",
    "    const uint4 r = f12m_decode(word(q, j), word(l, 2 * (j & 1)), word(l, 2 * (j & 1) + 1));",
    "    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024",
    "    const uint8_t* wsrc = wp + (static_cast<size_t>(kc0) * (N / 16) + static_cast<size_t>(rb) * 8) * 1536;",
    "    const size_t wstride = static_cast<size_t>(N / 16) * 1536;  // one chunk of all rows",
    "        if (!(DIAG & 4)) tma_2d(smem_u32(sX + s * X_BYTES), &xmap, (kc0 + i) * 64, tt * BM, full0 + 8 * s);",
    "  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(bm)};",
    "  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};",
    "  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |",
    "         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);",
    "    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \\",
    "  const int per = (KC + splits - 1) / splits;",
    "  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)",
    "          mt[t * 128 + r0] = acc[4 * j + h];",
    "          mt[t * 128 + r0 + 8] = acc[4 * j + 2 + h];",
    "      float* mt = tile + p * (BM * 128);",
    "      for (int idx = tid; idx < BM * 128; idx += THREADS) tile[idx] += tile[BM * 128 + idx];",
    "      const int r0 = 64 * c + 16 * w + g;  // rows r0, r0 + 8; tokens 8j + 2q (+1)",
    "    const int nr = glob ? 1 : splits, rank = glob ? 0 : sl;",
    "                           ? __uint_as_float(static_cast<uint32_t>(xh[static_cast<size_t>(tok[u]) * K + (esc[e] & 0xFFFFu)]) << 16)",
    "    const int RS = (128 + nr - 1) / nr;",
    "    const int rlo = rank * RS;",
    "        part[(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N + rb * 128 + (idx & 127)] = tile[idx];",
    "        const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;",
    "          atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again",
    "                   : glob ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row)",
    "    if (rn > 0 && tid < TL * rn && (!glob || s_last)) {",
    "    const int rhi = rlo + RS < 128 ? rlo + RS : 128;",
    "    const int TL = rn > 0 ? THREADS / rn : 0;",
    "      const int ri = tid % rn, tl = tid / rn;",
    "      for (int t0 = tl; t0 < BM; t0 += TL * FU) {",
    "          const int t = t0 + TL * u;",
    "            const uint32_t a = tile0 + static_cast<uint32_t>(((t0 + TL * u) * 128 + rlo + ri) * 4);",
    "        for (int s2 = 0; s2 < splits; ++s2) {  // rank order: deterministic",
    "              acc_s = fmaf(-xe[u][e], wdec, acc_s);",
    "              acc_s = fmaf(xe[u][e], __uint_as_float(esc[e] & 0xFFFF0000u), acc_s);",
    "          out[static_cast<size_t>(tok[u]) * N + row] = __float2bfloat16_rn(acc_s);",
    "__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }",
    "__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }",
    "      mbar_init(full0 + 8 * s, 1);  // thread 0's expect_tx arrive (bytes: the weight copy + the x TMA)",
    "  constexpr uint32_t TX = ((DIAG & 8) ? 0u : W_CHUNK) + ((DIAG & 4) ? 0u : X_BYTES);",
    "      if (i >= NP && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - NP) % S));",
    "      if (i + NP < nk) load(i + NP, nxt);",
    "    const int mw = (tid >> 7) - 1, c = mw & 1, p = mw >> 1;",
    "  if (MATH == 4) S &= ~1;  // even: slot s only ever holds chunks of parity s % 2, i.e. of pair s % 2",
    "      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == p);",
]
missing = [p for p in PINS if CUDA.count(p + "\n") < 1]
assert not missing, f"dgrs.py's CUDA changed; update the ports: {missing[:3]}"
print(f"ok 5: {len(PINS)} CUDA lines mirrored by the ports are unchanged")

M32 = 0xFFFFFFFF


def rotl(v, n):
    return ((v << n) | (v >> (32 - n))) & M32


def byte_perm(a, b, sel):
    """__byte_perm(a, b, s): bytes of {b, a} (a = bytes 0-3, b = 4-7) picked by the 4 selector nibbles."""
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(a)
    for i in range(4):
        sidx = (sel >> (4 * i)) & 0x7
        out |= pool[sidx] << (8 * i)
    return out


def f12m_decode(q, l0, l1):
    r4 = rotl(q, 4)
    ha = (q & 0x87878787) | 0x38383838
    hb = (r4 & 0x87878787) | 0x38383838
    return (byte_perm(l0, ha, 0x5140), byte_perm(l0, ha, 0x7362), byte_perm(l1, hb, 0x5140), byte_perm(l1, hb, 0x7362))


def bf16_of(bits16):
    return (bits16.to(torch.int32) << 16).view(torch.float32)


def decode_all(wcm, n, k):
    """Chunk-major planes [K/64][N/16][3][32][4] -> fp32 [N, K] of decoded A values, read at the kernel's offsets
    (stage = (kc, rb): byte (kc * N/16 + 8 rb) * 1536; row tile 4c + w; slot s at +512 s; lane * 16)."""
    words = wcm.to(torch.int64) & M32                     # [KC, RT, 3, 32, 4]
    kc_n, rt_n = words.shape[0], words.shape[1]
    q = words[:, :, 0]                                      # [KC, RT, 32, 4]: Q word of step j
    l01, l23 = words[:, :, 1], words[:, :, 2]
    out = torch.zeros(n, k)
    lane = torch.arange(32)
    g, t = lane >> 2, lane & 3
    for j in range(4):
        l = l01 if j < 2 else l23
        regs = f12m_decode(q[..., j], l[..., 2 * (j & 1)], l[..., 2 * (j & 1) + 1])
        for r, reg in enumerate(regs):
            for half in range(2):
                val = bf16_of((reg >> (16 * half)) & 0xFFFF)  # [KC, RT, 32]
                row = 16 * torch.arange(rt_n).view(1, -1, 1) + g.view(1, 1, -1) + 8 * (r & 1)
                col = (64 * torch.arange(kc_n).view(-1, 1, 1) + 16 * j + 2 * t.view(1, 1, -1) + 8 * (r >> 1) + half)
                row, col = torch.broadcast_tensors(row, col)
                out[row.reshape(-1), col.reshape(-1)] = val.reshape(-1)
    return out


def read_x_tile(x, tt, bm, kc):
    """Producer: unit u of token t -> byte t*128 + ((u ^ (t & 7)) << 4); wgmma B read through the SW128 descriptor:
    element (n, kk) of k16 step ks at ks*32 + (n//8)*1024 + (n%8)*128 + (kk//8)*16 + (kk%8)*2, XOR bits [4:6] with [7:9]."""
    m, k = x.shape
    buf = torch.full((bm * 128 // 2,), float("nan"))
    for t in range(bm):
        tok = tt * bm + t
        if tok >= m:  # TMA's out-of-bounds rows read as zeros
            off = t * 128
            buf[off // 2: off // 2 + 64] = 0.0
            continue
        for u in range(8):
            off = t * 128 + ((u ^ (t & 7)) << 4)
            buf[off // 2: off // 2 + 8] = x[tok, kc * 64 + u * 8: kc * 64 + u * 8 + 8]
    nn = torch.arange(bm).view(-1, 1, 1)
    ks = torch.arange(4).view(1, -1, 1)
    kk = torch.arange(16).view(1, 1, -1)
    addr = ks * 32 + (nn // 8) * 1024 + (nn % 8) * 128 + (kk // 8) * 16 + (kk % 8) * 2
    phys = addr ^ (((addr >> 7) & 7) << 4)
    return buf[(phys // 2).reshape(-1)].view(bm, 64)


def make_w(n, k, gen):
    w = (torch.randn(n, k, generator=gen) * 0.02)
    # plant out-of-window weights (escapes) on top of the Gaussian's own: 2 per row in 1/8 of the rows (F12 allows 7)
    rows = torch.randperm(n, generator=gen)[: n // 8]
    for r in rows.tolist():
        cols = torch.randint(0, k, (2,), generator=gen)
        w[r, cols] = torch.tensor([8.0, -40.0])
    return w.to(torch.bfloat16)


def emulate_layer(name, n, k, m, bm, splits, gen):
    w = make_w(n, k, gen)
    enc = f12["f12m_encode"](w, natural=True)
    rec = f12["f12v_records"](enc, n)
    assert rec is not None
    p = f12["as_int32"](enc["p"]).contiguous()
    wcm = dg["dgrs_pack"](p)
    base = (rec[:, 0] & 0xFF).to(torch.int64)
    # 1. decode == W * 2^(112 - base), escapes 2^-15
    dec = decode_all(wcm, n, k)
    scale = torch.pow(2.0, (112 - base).double()).view(-1, 1)
    want = (w.double() * scale)
    escm = torch.zeros(n, k, dtype=torch.bool)
    cnt_r = (rec[:, 0] >> 8) & 0xFF
    for r in range(n):
        for e in range(int(cnt_r[r])):
            escm[r, int(rec[r, 1 + e] & 0xFFFF)] = True
    want = torch.where(escm, torch.full_like(want, 2.0 ** -15), want)
    assert torch.equal(dec.double(), want), f"{name}: decoded A values differ from W * 2^(112 - base)"
    # 2. x tiles
    x = (torch.randn(m, k, generator=gen)).to(torch.bfloat16).float()
    tiles = -(-m // bm)
    for tt in range(tiles):
        for kc in (0, k // 64 - 1):
            got = read_x_tile(x, tt, bm, kc)
            v = min(bm, m - tt * bm)
            assert torch.equal(got[:v], x[tt * bm: tt * bm + v, kc * 64: kc * 64 + 64]), (name, tt, kc)
    # 3. split-K GEMM + last-arriver epilogue
    kc_n = k // 64
    per = -(-kc_n // splits)
    assert (splits - 1) * per < kc_n, (name, splits)
    mp = tiles * bm
    part = torch.zeros(splits, mp, n)
    xp = torch.zeros(mp, k)
    xp[:m] = x
    for sl in range(splits):
        kc0 = sl * per
        nk = min(per, kc_n - kc0)
        ks = slice(64 * kc0, 64 * (kc0 + nk))
        part[sl] = xp[:, ks] @ dec[:, ks].T       # [mp, n] in fp32 (MMA order differs: tolerance below)
    out = torch.zeros(m, n)
    tot = part.sum(0)                              # slices in order s = 0..S-1 (the finalize loop)
    tot = tot * torch.pow(2.0, (base - 112).float()).view(1, -1)
    wdec = torch.pow(2.0, (base - 127).float())
    for r in range(n):
        for e in range(int(cnt_r[r])):
            kk = int(rec[r, 1 + e] & 0xFFFF)
            true = bf16_of(torch.tensor([(int(rec[r, 1 + e]) >> 16) & 0xFFFF]))[0]
            tot[:, r] = tot[:, r] + xp[:, kk] * (true - wdec[r])
    out = tot[:m].to(torch.bfloat16).float()
    ref = x @ w.float().T
    err = float((out - ref).norm() / ref.norm())
    row = float(((out - ref).norm(dim=0) / ref.norm(dim=0).clamp(min=1e-30)).max())
    assert err < 1e-2 and row < 5e-2, (name, err, row)
    return err, row, int(escm.sum())


def counter_protocol():
    rng = random.Random(3)
    for rbs, tiles, splits in ((18, 1, 7), (64, 2, 2), (32, 1, 4), (8, 2, 1)):
        cnt = [0] * (rbs * tiles)
        for call in range(3):
            done = {}
            ctas = [(rb, tt, sl) for rb in range(rbs) for tt in range(tiles) for sl in range(splits)]
            rng.shuffle(ctas)
            finished = set()
            for rb, tt, sl in ctas:
                finished.add((rb, tt, sl))
                i = tt * rbs + rb
                old = cnt[i]
                cnt[i] += 1
                if old == splits - 1:
                    cnt[i] = 0
                    assert all((rb, tt, s2) in finished for s2 in range(splits)), "finalize before all slices"
                    assert (rb, tt) not in done, "finalized twice"
                    done[(rb, tt)] = True
            assert len(done) == rbs * tiles and not any(cnt), (len(done), call)
    return True


def finalize_coverage():
    """v3 finalize: rank sl of S takes rows [sl * RS, min(128, (sl + 1) * RS)), RS = ceil(128 / S); thread tid <
    TL * rn -> (row ri = tid % rn, token lane tl = tid / rn), TL = 384 // rn, tokens t = tl + TL (FU g + u) < BM:
    every (row, token) of the tile exactly once over all ranks and threads, for S = 1..8 and BM 32/64/128."""
    for threads, bm in ((384, 32), (384, 64), (384, 128), (640, 32), (640, 64)):
        for splits in range(1, 9):
            rs = -(-128 // splits)
            seen = {}
            for sl in range(splits):
                rlo = sl * rs
                rhi = min(128, rlo + rs)
                rn = max(0, rhi - rlo)
                if rn == 0:
                    continue
                tl_n = threads // rn
                for tid in range(tl_n * rn):
                    ri, tl = tid % rn, tid // rn
                    for t0 in range(tl, bm, tl_n * 4):
                        for u in range(4):
                            tt = t0 + tl_n * u
                            if tt < bm:
                                seen[(rlo + ri, tt)] = seen.get((rlo + ri, tt), 0) + 1
            assert len(seen) == 128 * bm and set(seen.values()) == {1}, (threads, bm, splits, len(seen))


def math4_protocol(S, nk, seed, np_=2, rel_off=None):
    """Producer (thread 0: chunk i -> slot i % S after the slot's previous phase was released; full barrier count 1)
    and NP pairs (pair p: chunks p, p + NP, ...; full wait (i / S) & 1; after its step for chunk i, release chunk
    i - NP; both warpgroups of a pair arrive: empty count 2) in random interleavings. Checks: each chunk consumed once,
    by pair i % NP, only after it landed and before its slot is refilled; no deadlock."""
    rng = random.Random(seed)
    full_ph = [0] * S      # completed phases
    empty_ph = [0] * S
    empty_cnt = [0] * S
    slot_chunk = [None] * S
    reading = [set() for _ in range(S)]
    consumed = {}
    prod_i = 0
    pairs = [[p, 0] for p in range(np_)]   # next chunk, step state
    steps = 0
    while True:
        steps += 1
        if steps > 100000:
            raise AssertionError("livelock")
        opts = []
        if prod_i < nk:
            s, rnd = prod_i % S, prod_i // S
            # try_wait.parity((rnd - 1) & 1) passes once the barrier's completed-phase count has the other parity
            if rnd == 0 or (empty_ph[s] & 1) != ((rnd - 1) & 1):
                opts.append(("prod",))
        for p in range(np_):
            i = pairs[p][0]
            if i < nk and (full_ph[i % S] & 1) != ((i // S) & 1):   # parity wait, as the hardware does it
                opts.append(("cons", p))
        if not opts:
            assert prod_i >= nk and all(pr[0] >= nk for pr in pairs), (S, nk, "deadlock")
            break
        o = rng.choice(opts)
        if o[0] == "prod":
            s = prod_i % S
            assert not reading[s], f"slot {s} refilled while chunk {slot_chunk[s]} is read"
            assert prod_i < S or empty_ph[s] == prod_i // S, f"slot {s}: producer passed a stale empty phase"
            slot_chunk[s] = prod_i
            full_ph[s] += 1
            prod_i += 1
        else:
            p = o[1]
            i = pairs[p][0]
            assert i % np_ == p and i not in consumed
            assert slot_chunk[i % S] == i and full_ph[i % S] == i // S + 1, f"pair {p} read chunk {i} from a stale phase"
            consumed[i] = p
            reading[i % S].add(i)
            prev = i - (np_ if rel_off is None else rel_off)
            if prev >= 0:   # after wait_group<1>: the pair's previous chunk is done; both warpgroups arrive
                reading[prev % S].discard(prev)
                for _wg in range(2):
                    empty_cnt[prev % S] += 1
                    if empty_cnt[prev % S] == 2:
                        empty_cnt[prev % S] = 0
                        empty_ph[prev % S] += 1
            pairs[p][0] = i + np_
    assert sorted(consumed) == list(range(nk))


for S_ in (4, 6, 8):
    for nk_ in range(1, 23):
        for sd in range(20):
            math4_protocol(S_, nk_, sd)
            math4_protocol(S_, nk_, sd, np_=1)
for S_ in (5, 7):  # an odd ring is safe too: each pair and the producer go in order, so a slot's completed-phase
    for sd in range(20):  # count is k or k + 1 when a pair waits for phase k (no parity aliasing); kernel keeps S even
        math4_protocol(S_, 17, sd)
bad_caught = False
for sd in range(200):
    try:
        math4_protocol(4, 17, sd, rel_off=1)  # planted: release the chunk just before (the other pair's), not i - NP
    except AssertionError:
        bad_caught = True
        break
assert bad_caught, "releasing the other pair's chunk must be caught"
print("ok 4b: MATH 4 / MATH 2 chunk schedule with hardware parity waits: every chunk consumed once by its pair after "
      "it landed, no refill while read, no stale phase, no deadlock (S 4-8, 1-22 chunks, random orders); a planted "
      "wrong release is caught")
finalize_coverage()
print("ok 4: v3 finalize map (rank row ranges x thread (row, token lane)) covers every (row, token) once for "
      "S = 1..8, BM 32/64/128 (384 threads) and BM 32/64 (640 threads, MATH 4); slices are summed in rank order (deterministic)")

gen = torch.Generator().manual_seed(0)
LAYERS = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pooler", 4096, 8192), ("unpooler", 8192, 4096)]
CASES = [(32, 32, {"qkv": 7, "o": 2, "pooler": 4, "unpooler": 2}),
         (128, 128, {"qkv": 7, "o": 2, "pooler": 4, "unpooler": 2}),
         (256, 128, {"qkv": 4, "o": 1, "pooler": 2, "unpooler": 1})]
if QUICK:
    LAYERS = [("o", 1024, 1024), ("qkv-small", 256, 2048)]
    CASES = [(32, 32, {"o": 2, "qkv-small": 7}), (200, 128, {"o": 1, "qkv-small": 3})]
lines = []
for m, bm, splits in CASES:
    for name, n, k in LAYERS:
        e, r, ne = emulate_layer(name, n, k, m, bm, splits[name], gen)
        lines.append(f"{name} m{m} bm{bm} S{splits[name]}: err {e:.1e} row {r:.1e} ({ne} escapes)")
print("ok 1-3: decode == W * 2^(112 - base) exactly (escapes 2^-15), x tiles read back exactly, split-K GEMM + "
      "F12M epilogue vs FP32 x @ W^T:\n  " + "\n  ".join(lines))

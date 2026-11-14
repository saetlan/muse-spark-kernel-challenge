
# ---------------------------------------------------------------------------------------------
# v3 (R3): fused NVFP4 grouped expert GEMMs in Triton, for prefill token counts (local/make_v3.py
# inserts this after v2/gmm.py).
#
# The local routes are sorted by expert (as v2) and cut into tiles of BT rows, one expert per tile;
# a device-side tile table (no host sync) maps each program to (expert, rows). Each program decodes
# BN weight rows of its expert straight from NVFP4 per BK step and multiplies them with its BT token
# rows: the decode is paid once per BT tokens, which is what makes it cheap at prefill sizes (at
# decode sizes it isn't: MoE F45). GEMM1 does gate and up together with SwiGLU in the epilogue;
# GEMM2 writes per-route rows for v2's post-norm combine kernel.
#
# The decode (_r3_dequant / _r3_scales / _r3_tile) and the nibble repack (r3_repack) are copied from
# our MoE session's M2 (moe/m2/m2.py: _m2_dequant, _m2_scales, _m2_tile, m2_repack), whose asm was
# interpreted bit for bit locally and ran correctly on the H200 (MoE E2). Weights stay NVFP4:
# r3_repack only permutes nibbles (lossless).
# ---------------------------------------------------------------------------------------------

# Decoded weights are code * scale * 2^-62 (Marlin's bit trick with scales raised by 2^64);
# epilogues multiply by 2^62.
R3_OUT_SHIFT = 2.0 ** 62
# Token classes: (fewer than this many tokens, class); the rest is "ft" (first-turn 16K chunks).
R3_CLASSES = ((1200, "t1"), (8192, "t4"))
# class -> (GEMM1 config, GEMM2 config); a config is (swap, bt, bn, bk, warps, stages, scale2d).
# Empty until a lab measures them; prepare() checks each against Marlin and drops any that disagree.
R3_CONFIGS = {}
R3_REP_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
R3_MAX_ERR = 1e-2
# v2's BF16 grouped path serves from this many tokens when no R3 config does (it lost at 4096: RE2).
GMM_FROM_TOKENS = 8192


def _r3_class(tokens):
    for bound, name in R3_CLASSES:
        if tokens < bound:
            return name
    return "ft"


def r3_repack(packed, bk):
    """uint8 [..., K/2] NVFP4 codes (element 2i in the low nibble) -> the same bytes permuted so
    that within each bk-wide K tile, byte j holds element j (low nibble) and j + bk/2 (high)."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    t = codes.unflatten(-1, (codes.shape[-1] // bk, 2, bk // 2))
    return (t[..., 0, :] | (t[..., 1, :] << 4)).flatten(-2).contiguous()


@triton.jit
def _r3_dequant(q, s_lo, s_hi):
    """Per byte of q: its low and high E2M1 nibbles as bf16 (value * 2^-126, Marlin's bit trick),
    times s_lo / s_hi with mul.rn.bf16x2. prmt first pairs bytes (0,1) and (2,3) into the 16-bit
    lanes, so output element i comes from input byte i whatever elements Triton packs together."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 w, t, s, m;
        prmt.b32 w, $4, $4, 0x3120;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $3, t, $8;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $1, t, $6;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $2, t, $7;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $0, t, $5;
        }
        """,
        constraints="=r,=r,=r,=r,r,r,r,r,r", args=[q, s_lo, s_hi],
        dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)


@triton.jit
def _r3_scales(ptrs):
    """float8_e4m3fn block scales -> bf16 scale * 2^64 (exact)."""
    return (tl.load(ptrs).to(tl.float32) * 18446744073709551616.0).to(tl.bfloat16)


@triton.jit
def _r3_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN: tl.constexpr, HB: tl.constexpr,
             G: tl.constexpr, SCALE2D: tl.constexpr):
    """Dequantized [BN, HB] bf16 halves of the BK = 2 * HB tile at k0: (elements k0 + j, k0 + HB + j)."""
    if SCALE2D:
        q = tl.load(w_rows[:, None] + (k0 // 2 + offs_j)[None, :])
        s_lo = _r3_scales(s_rows[:, None] + ((k0 + offs_j) // 16)[None, :])
        s_hi = _r3_scales(s_rows[:, None] + ((k0 + HB + offs_j) // 16)[None, :])
        lo, hi = _r3_dequant(q, s_lo, s_hi)
    else:
        s_lo = tl.broadcast_to(_r3_scales(s_tile + k0 // 16)[:, :, None], (BN, G, 16))
        s_hi = tl.broadcast_to(_r3_scales(s_tile + (k0 // 16 + G))[:, :, None], (BN, G, 16))
        lo, hi = _r3_dequant(tl.load(w_tile + k0 // 2), s_lo, s_hi)
        lo = tl.reshape(lo, (BN, HB))
        hi = tl.reshape(hi, (BN, HB))
    return lo, hi


@triton.jit(do_not_specialize=["max_tiles"])
def _r3_gemm_kernel(a_ptr, w_ptr, sf_ptr, c_ptr, table_ptr, out_ptr, max_tiles,
                    N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, GATED: tl.constexpr,
                    SWAP: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SCALE2D: tl.constexpr):
    """Output columns [nb * BN, +BN) for the rows of tile t (one expert, rows row0 .. row_end - 1 of
    the expert-sorted activations a). Weight rows: n (gate or plain) and N + n (up, GATED)."""
    pid = tl.program_id(0)
    nb = pid // max_tiles
    t = pid - nb * max_tiles
    e = tl.load(table_ptr + t)
    if e < 0:
        return
    row0 = tl.load(table_ptr + max_tiles + t)
    row_end = tl.load(table_ptr + 2 * max_tiles + t)
    HB: tl.constexpr = BK // 2
    G: tl.constexpr = BK // 32
    e64 = e.to(tl.int64)
    c_e = tl.load(c_ptr + e)
    offs_n = nb * BN + tl.arange(0, BN)
    w_rows = w_ptr + e64 * (ROWS * (K // 2)) + offs_n.to(tl.int64) * (K // 2)
    s_rows = sf_ptr + e64 * (ROWS * (K // 16)) + offs_n.to(tl.int64) * (K // 16)
    offs_g = tl.arange(0, G)
    offs_i = tl.arange(0, 16)
    offs_j = tl.arange(0, HB)
    w_tile = w_rows[:, None, None] + (offs_g[:, None] * 16 + offs_i[None, :])[None, :, :]
    s_tile = s_rows[:, None] + offs_g[None, :]
    a_row = (row0 + tl.arange(0, BT)).to(tl.int64)
    valid = a_row < row_end
    if SWAP:
        a_tile = a_ptr + a_row[None, :] * K + offs_j[:, None]
        a_mask = valid[None, :]
        acc = tl.zeros((BN, BT), dtype=tl.float32)
        acc_u = tl.zeros((BN, BT), dtype=tl.float32)
    else:
        a_tile = a_ptr + a_row[:, None] * K + offs_j[None, :]
        a_mask = valid[:, None]
        acc = tl.zeros((BT, BN), dtype=tl.float32)
        acc_u = tl.zeros((BT, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a_lo = tl.load(a_tile + k0, mask=a_mask, other=0.0)
        a_hi = tl.load(a_tile + (k0 + HB), mask=a_mask, other=0.0)
        lo, hi = _r3_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN, HB, G, SCALE2D)
        if SWAP:
            acc = tl.dot(lo, a_lo, acc)
            acc = tl.dot(hi, a_hi, acc)
        else:
            acc = tl.dot(a_lo, tl.trans(lo), acc)
            acc = tl.dot(a_hi, tl.trans(hi), acc)
        if GATED:
            lo, hi = _r3_tile(w_rows + N * (K // 2), s_rows + N * (K // 16), w_tile + N * (K // 2),
                              s_tile + N * (K // 16), offs_j, k0, BN, HB, G, SCALE2D)
            if SWAP:
                acc_u = tl.dot(lo, a_lo, acc_u)
                acc_u = tl.dot(hi, a_hi, acc_u)
            else:
                acc_u = tl.dot(a_lo, tl.trans(lo), acc_u)
                acc_u = tl.dot(a_hi, tl.trans(hi), acc_u)
    if SWAP:
        rows = a_row[None, :]
        cols = offs_n[:, None]
    else:
        rows = a_row[:, None]
        cols = offs_n[None, :]
    if GATED:
        g = acc * c_e
        res = g * tl.sigmoid(g) * (acc_u * c_e)
    else:
        res = acc * c_e
    tl.store(out_ptr + rows * N + cols, res.to(tl.bfloat16), mask=a_mask)


def _r3_route(x, topk_ids, state):
    """v2's GPU-side sort of the local routes by expert, plus the gathered activation rows."""
    bank = state["bank"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    device = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
    _, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,
                                  num_warps=4)
    return local_ids, counts, offs, inv, a


def _r3_max_tiles(slots, bt):
    return -(-slots // bt) + LOCAL_EXPERTS


def _r3_tiles(counts, offs, bt, max_tiles):
    """int32 [3, max_tiles]: (expert or -1, first row, end row) of each BT-row tile, experts in order."""
    c = counts[:LOCAL_EXPERTS]
    tiles = (c + (bt - 1)) // bt
    t_end = tiles.cumsum(0, dtype=torch.int32)
    t_start = t_end - tiles
    p = torch.arange(max_tiles, device=c.device, dtype=torch.int32)
    e = torch.searchsorted(t_end, p, right=True)
    e_c = e.clamp(max=LOCAL_EXPERTS - 1)
    r_start = (offs - c)[e_c] + (p - t_start[e_c]) * bt
    expert = torch.where(e < LOCAL_EXPERTS, e, torch.full_like(e, -1))
    return torch.stack((expert.to(torch.int32), r_start.to(torch.int32), offs[e_c].to(torch.int32))).contiguous()


def _r3_gemm(a, q, sf, c, table, max_tiles, out, cfg, gated):
    swap, bt, bn, bk, warps, stages, scale2d = cfg
    rows_w = q.shape[1]
    n = rows_w // 2 if gated else rows_w
    k = a.shape[1]
    return _r3_gemm_kernel[((n // bn) * max_tiles,)](
        a, q, sf, c, table, out, max_tiles, N=n, K=k, ROWS=rows_w, GATED=gated, SWAP=swap, BT=bt, BN=bn,
        BK=bk, SCALE2D=scale2d, num_warps=warps, num_stages=stages)


def _experts_r3(x, topk_weights, topk_ids, state, cfg1, cfg2):
    """Same contract as _experts."""
    r3, gmm = state["r3"], state["gmm"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    local_ids, counts, offs, inv, a = _r3_route(x, topk_ids, state)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    mt1, mt2 = _r3_max_tiles(slots, cfg1[1]), _r3_max_tiles(slots, cfg2[1])
    table1 = _r3_tiles(counts, offs, cfg1[1], mt1)
    table2 = table1 if cfg2[1] == cfg1[1] else _r3_tiles(counts, offs, cfg2[1], mt2)
    act = torch.empty((slots, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _r3_gemm(a, r3["w13"][cfg1[3]], gmm["w13_sf"], r3["c1"], table1, mt1, act, cfg1, True)
    y2 = torch.empty((slots, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _r3_gemm(act, r3["w2"][cfg2[3]], gmm["w2_sf"], r3["c2"], table2, mt2, y2, cfg2, False)
    out = torch.empty((tokens, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe_v3(x, topk_weights, topk_ids, state, tokens):
    cfg = state["r3"]["cfg"].get(_r3_class(tokens))
    if cfg is not None:
        return _experts_r3(x, topk_weights, topk_ids, state, *cfg)
    if tokens >= GMM_FROM_TOKENS and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def r3_add_bk(state, bk):
    """Repacked copies of the expert weights for BK = bk (lossless nibble permutation), per expert."""
    r3, gmm = state["r3"], state["gmm"]
    for name in ("w13", "w2"):
        if bk not in r3[name]:
            src = gmm[name + "_q"]
            r3[name][bk] = torch.stack([r3_repack(src[e], bk) for e in range(src.shape[0])])


def r3_check(state, cfg1, cfg2, tokens, seed=2):
    """(rel err, worst row err) of _experts_r3 against SGLang's Marlin path on a GPU routing."""
    device = state["bank"]["expert_map"].device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=device) * 0.5).to(torch.bfloat16)
    ids, w = _gmm_selftest_routing(tokens, device, seed=seed)
    ref = _experts(x, w, ids, state).float()
    out = _experts_r3(x, w, ids, state, cfg1, cfg2).float()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
    row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
    return err, row


def _r3_prepare(state, weights):
    """Global scales for the epilogues, repacked weights for the configured BKs, and a GPU check of
    each configured class against Marlin (which also compiles its kernels); failures fall back."""
    gmm = state["gmm"]
    state["r3"] = {
        "w13": {}, "w2": {}, "cfg": {},
        "c1": (gmm["w13_gs"] * R3_OUT_SHIFT).contiguous(),
        "c2": torch.full_like(gmm["w2_gs"], R3_OUT_SHIFT),
    }
    for cls, (cfg1, cfg2) in R3_CONFIGS.items():
        try:
            r3_add_bk(state, cfg1[3])
            r3_add_bk(state, cfg2[3])
            err, row = r3_check(state, cfg1, cfg2, R3_REP_TOKENS[cls])
            ok = err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
            print(f"[v3] {cls}: fused vs Marlin rel err {err:.2e}, worst row {row:.2e} -> {'fused' if ok else 'fallback'}",
                  flush=True)
            if ok:
                state["r3"]["cfg"][cls] = (cfg1, cfg2)
        except Exception as exc:
            print(f"[v3] {cls}: fused path failed, falling back: {type(exc).__name__}: {str(exc)[-300:]}", flush=True)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# v2: the routed experts as BF16 grouped GEMMs from 4096 tokens up (local/make_v2.py inserts this).
#
# Each call, the NVFP4 weights of every local expert that got a route are decoded into a BF16
# scratch: E2M1 code x E4M3 block scale, exact in BF16 (at most 6 significant bits); the FP32
# global scale is applied to the GEMM outputs instead, so no weight is rounded. The local routes
# are sorted by expert on the GPU (no host sync), gathered, and run through torch._grouped_mm
# (CUTLASS wgmma) twice, with SiLU-and-mul and the post-expert norm weighted sum as Triton
# kernels that read the sorted rows. Below 4096 tokens SGLang's Marlin path runs unchanged.
# prepare() checks the path against Marlin on the GPU and keeps Marlin if it disagrees.
# ---------------------------------------------------------------------------------------------

import triton
import triton.language as tl

GMM_MIN_TOKENS = 4096
GMM_SELFTEST_TOKENS = 4096
GMM_SELFTEST_MAX_ERR = 2e-2
DEQ_BLOCK_N = 32      # rows per dequant program
DEQ_BLOCK_KB = 128    # packed bytes per dequant program (256 values)


@triton.jit
def _e2m1_value(code):
    """E2M1 code (low 4 bits of an int32) -> float32 value."""
    sign = (code >> 3) & 1
    exp = (code >> 1) & 3
    man = (code & 1).to(tl.float32)
    normal = (1.0 + 0.5 * man) * tl.where(exp == 3, 4.0, tl.where(exp == 2, 2.0, 1.0))
    mag = tl.where(exp == 0, 0.5 * man, normal)
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _nvfp4_dequant_kernel(q_ptr, sf_ptr, counts_ptr, out_ptr, N: tl.constexpr, KB: tl.constexpr,
                          BN: tl.constexpr, BKB: tl.constexpr):
    """out[e, n, :] = e2m1(q[e, n, :]) * sf[e, n, :/16] in BF16 (element 2i = low nibble), for experts
    with at least one local route."""
    e = tl.program_id(0)
    if tl.load(counts_ptr + e) == 0:
        return
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.program_id(2) * BKB + tl.arange(0, BKB)
    e64 = e.to(tl.int64)
    q = tl.load(q_ptr + e64 * (N * KB) + rows[:, None] * KB + cols[None, :]).to(tl.int32)
    sf = tl.load(sf_ptr + e64 * (N * (KB // 8)) + rows[:, None] * (KB // 8) + (cols // 8)[None, :]).to(tl.float32)
    lo = _e2m1_value(q & 15) * sf
    hi = _e2m1_value(q >> 4) * sf
    vals = tl.reshape(tl.join(lo, hi), (BN, 2 * BKB))
    out_cols = tl.program_id(2) * (2 * BKB) + tl.arange(0, 2 * BKB)
    tl.store(out_ptr + e64 * (N * 2 * KB) + rows[:, None] * (2 * KB) + out_cols[None, :], vals.to(tl.bfloat16))


@triton.jit
def _gather_rows_kernel(x_ptr, perm_ptr, offs_ptr, out_ptr, H: tl.constexpr, TOPK: tl.constexpr,
                        NE: tl.constexpr):
    """out[r] = x[perm[r] // TOPK] for the local routes (r < offs[NE - 1])."""
    r = tl.program_id(0)
    if r >= tl.load(offs_ptr + NE - 1):
        return
    tok = (tl.load(perm_ptr + r) // TOPK).to(tl.int64)
    cols = tl.arange(0, H)
    tl.store(out_ptr + r.to(tl.int64) * H + cols, tl.load(x_ptr + tok * H + cols))


@triton.jit
def _gmm_act_kernel(h_ptr, keys_ptr, gs_ptr, offs_ptr, out_ptr, I: tl.constexpr, NE: tl.constexpr):
    """out[r] = silu(gs * gate) * (gs * up) in BF16, gs = the row's expert's global scale."""
    r = tl.program_id(0)
    if r >= tl.load(offs_ptr + NE - 1):
        return
    gs = tl.load(gs_ptr + tl.load(keys_ptr + r))
    cols = tl.arange(0, I)
    base = r.to(tl.int64) * (2 * I)
    gate = tl.load(h_ptr + base + cols).to(tl.float32) * gs
    up = tl.load(h_ptr + base + I + cols).to(tl.float32) * gs
    tl.store(out_ptr + r.to(tl.int64) * I + cols, (gate * tl.sigmoid(gate) * up).to(tl.bfloat16))


@triton.jit
def _gmm_combine_kernel(y_ptr, ids_ptr, inv_ptr, gs_ptr, s_ptr, w_ptr, out_ptr, eps, H: tl.constexpr,
                        TOPK: tl.constexpr):
    """SGLang's post-expert norm weighted sum (weight first), reading each local slot's row of the
    sorted GEMM output and applying its expert's global scale."""
    tok = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        expert = tl.load(ids_ptr + tok * TOPK + k)
        if expert >= 0:
            row = tl.load(inv_ptr + tok * TOPK + k)
            x = tl.load(y_ptr + row * H + offs).to(tl.float32) * tl.load(gs_ptr + expert)
            y = x * w
            var = tl.sum(y * y, axis=0) / H
            y = y * tl.rsqrt(var + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(tl.bfloat16))


def _dequant(q, sf, counts, out):
    experts, n, kb = q.shape
    _nvfp4_dequant_kernel[(experts, n // DEQ_BLOCK_N, kb // DEQ_BLOCK_KB)](
        q, sf, counts, out, N=n, KB=kb, BN=DEQ_BLOCK_N, BKB=DEQ_BLOCK_KB, num_warps=4)


def _experts_gmm(x, topk_weights, topk_ids, state):
    """Same contract as _experts: x [T, 4096] BF16, topk_ids in this GPU's numbering."""
    gmm, bank = state["gmm"], state["bank"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    device = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
    sorted_keys, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,
                                  num_warps=4)
    _dequant(gmm["w13_q"], gmm["w13_sf"], counts, gmm["w13"])
    _dequant(gmm["w2_q"], gmm["w2_sf"], counts, gmm["w2"])
    h1 = torch._grouped_mm(a, gmm["w13"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    act = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w13_gs"], offs, act, I=D_EXPERT, NE=LOCAL_EXPERTS,
                              num_warps=4)
    y2 = torch._grouped_mm(act, gmm["w2"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    out = torch.empty((tokens, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe(x, topk_weights, topk_ids, state, tokens):
    if tokens >= GMM_MIN_TOKENS and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def _gmm_selftest_routing(tokens, device, seed=0):
    """Random top-8 over 256 experts, tilted so ~1/4 of routes hit experts 0-31 (like the busiest GPU)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
    scores[:, :LOCAL_EXPERTS] += 0.35
    vals, ids = scores.topk(TOP_K, dim=-1)
    return ids.to(torch.int32), (vals / vals.sum(-1, keepdim=True)).float()


def _gmm_prepare(state, weights):
    """Raw NVFP4 weights (same tensors, same precision), BF16 scratch for the per-call decode, and
    a GPU check of the grouped path against SGLang's Marlin path (which also compiles every kernel)."""
    device = weights["w13"].device
    gmm = {
        "w13_q": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
        "w2_q": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].float().contiguous(), "w2_gs": weights["w2_gs"].float().contiguous(),
        "ok": False,
    }
    experts, n13, kb13 = gmm["w13_q"].shape
    _, n2, kb2 = gmm["w2_q"].shape
    gmm["w13"] = torch.empty((experts, n13, 2 * kb13), device=device, dtype=torch.bfloat16)
    gmm["w2"] = torch.empty((experts, n2, 2 * kb2), device=device, dtype=torch.bfloat16)
    state["gmm"] = gmm
    try:
        gen = torch.Generator(device=device).manual_seed(1)
        x = (torch.randn((GMM_SELFTEST_TOKENS, D_EXPERT), generator=gen, device=device) * 0.5).to(torch.bfloat16)
        ids, w = _gmm_selftest_routing(GMM_SELFTEST_TOKENS, device)
        ref = _experts(x, w, ids, state).float()
        out = _experts_gmm(x, w, ids, state).float()
        torch.cuda.synchronize()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
        row_err = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        finite = bool(torch.isfinite(out).all())
        gmm["ok"] = finite and err < GMM_SELFTEST_MAX_ERR and row_err < 5 * GMM_SELFTEST_MAX_ERR
        print(f"[v2] grouped path vs Marlin at {GMM_SELFTEST_TOKENS} tokens: rel err {err:.2e}, worst row "
              f"{row_err:.2e}, finite {finite} -> {'grouped' if gmm['ok'] else 'Marlin'} from {GMM_MIN_TOKENS} tokens",
              flush=True)
    except Exception as exc:
        print(f"[v2] grouped path failed its check, keeping Marlin: {type(exc).__name__}: {str(exc)[-400:]}",
              flush=True)
        gmm["ok"] = False
    torch.cuda.synchronize()

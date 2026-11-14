"""PE18: the router with its FP32 weights split once, in prepare(), into contiguous BF16 hi and lo [256, 8192]
(SGLang's split above 8 tokens, allowed at any batch size by the task) and read by two Triton kernels that
feed both straight into tl.dot: _hl_gemm_kernel (tokens x weights, m = 8..256) and _hl_swap_kernel (weights x
tokens, m <= 32). v7a's Triton router kernels load FP32 tiles and split them in registers for every tile, so
the operands take a register -> shared-memory detour before each dot; here the loads go straight to the
MMA. Same bytes (2 + 2 per weight), same math as SGLang (BF16 products, FP32 accumulation), FP32 output,
split-K with FP32 atomics. Sweeps router m = 256/128/32/16/8 against SGLang and the base; winners >= 3%
enter CHOICE (run_choice dispatches the new kinds), printed as "V14 CHOICE" lines."""

import math
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


@triton.jit(do_not_specialize=["M"])
def _hl_gemm_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * k_per_split
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_off = offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        x = tl.load(x_ptrs + kk, mask=m_mask[:, None], other=0.0)
        hi = tl.load(hi_ptr + w_off + kk, mask=n_mask[None, :], other=0.0, eviction_policy="evict_first")
        lo = tl.load(lo_ptr + w_off + kk, mask=n_mask[None, :], other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, hi, acc)
        acc = tl.dot(x, lo, acc)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    else:
        tl.store(out_ptrs, acc, mask=mask)


@triton.jit(do_not_specialize=["M"])
def _hl_swap_kernel(hi_ptr, lo_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    m_mask = offs_m < M
    k0 = pid_k * k_per_split
    w_off = offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        hi = tl.load(hi_ptr + w_off + kk, mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        lo = tl.load(lo_ptr + w_off + kk, mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        acc = tl.dot(hi, xt, acc)
        acc = tl.dot(lo, xt, acc)
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    else:
        tl.store(out_ptrs, acc, mask=mask)


def hl_setup(state):
    """The router's BF16 hi + lo, contiguous [N, K] (SGLang's split; state keeps the transposed copies too)."""
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16).contiguous()


def run_hl(c, x, layer, n, state):
    """c: dict(kind="hl_gemm" | "hl_swap", BM, BN, BK, split, warps, stages). FP32 out; split-K adds with FP32
    atomics into zeros. Without the hi/lo copies (or for another layer) it runs SGLang's path."""
    hi, lo = state.get("router_hi"), state.get("router_lo")
    m, k = x.shape
    if layer != "router" or hi is None or (c["kind"] == "hl_swap" and m > c["BM"]):
        return sglang_path(x, layer, n, state)
    split = c["split"]
    atomic = split > 1
    out = (torch.zeros if atomic else torch.empty)((m, n), dtype=torch.float32, device=x.device)
    kps = k // split
    x = x.contiguous()
    if c["kind"] == "hl_swap":
        _hl_swap_kernel[(triton.cdiv(n, c["BN"]), split)](
            hi, lo, x, out, m, n, kps, hi.stride(0), x.stride(0), out.stride(0),
            BM=c["BM"], BN=c["BN"], BK=c["BK"], ATOMIC=atomic, num_warps=c["warps"], num_stages=c["stages"])
    else:
        _hl_gemm_kernel[(triton.cdiv(m, c["BM"]), triton.cdiv(n, c["BN"]), split)](
            x, hi, lo, out, m, n, kps, x.stride(0), hi.stride(0), out.stride(0),
            BM=c["BM"], BN=c["BN"], BK=c["BK"], ATOMIC=atomic, num_warps=c["warps"], num_stages=c["stages"])
    return out


_run_choice_base = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] in ("hl_gemm", "hl_swap"):
        return run_hl(c, x, layer, n, state)
    return _run_choice_base(c, x, layer, n, state)


def hl_cands(m, max_n=44):
    """Configs for router m: programs 64..528, Triton's shared-memory ring <= 200 KB, ranked by token-tile
    re-reads of the weights, wave balance over 132 SMs, then deeper K tiles."""
    out = []
    mp = max(16, 1 << (m - 1).bit_length())                 # m rounded up to a power of two, at least 16
    for kind in ("hl_gemm", "hl_swap"):
        if kind == "hl_swap" and m > 32:
            continue
        bms = (mp,) if kind == "hl_swap" else tuple(b for b in (256, 128, 64, 32, 16) if b <= mp and b >= min(mp, 64))
        for bm in bms:
            for bn in (16, 32, 64, 128):
                if kind == "hl_swap" and bn < 32:
                    continue
                for split in (4, 8, 16, 32, 64):
                    for bk in (64, 128, 256):
                        tiles = math.ceil(m / bm) * (256 // bn) * split if kind == "hl_gemm" else (256 // bn) * split
                        if not 64 <= tiles <= 528 or 8192 % (split * bk) or 8192 // split < bk:
                            continue
                        for warps in (4, 8):
                            if bm * bn // (warps * 32) > 128 or bm * bn // (warps * 32) < 4:
                                continue
                            for stages in (3, 4):
                                if stages * (bm * bk * 2 + 2 * bk * bn * 2) > 200 * 1024:
                                    continue
                                reread = math.ceil(m / bm) if kind == "hl_gemm" else 1
                                waves = tiles / 132
                                balance = waves / math.ceil(waves)
                                out.append(((reread, -round(balance, 1), -bk, kind != "hl_swap", warps, stages),
                                            dict(kind=kind, BM=bm, BN=bn, BK=bk, split=split, warps=warps, stages=stages)))
    out.sort(key=lambda e: e[0])
    picked, seen = [], set()
    for _, c in out:                                          # variety first: one per (kind, BN, split)
        key = (c["kind"], c["BN"], c["split"])
        if key not in seen:
            seen.add(key)
            picked.append(c)
    picked += [c for _, c in out if c not in picked]
    return picked[:max_n]


def hl_name(c):
    return f"{'g' if c['kind'] == 'hl_gemm' else 's'}{c['BM']}/{c['BN']}/{c['BK']}/{c['split']}/{c['warps']}/{c['stages']}"


def sec_hl(lab, sweep, m):
    cands = hl_cands(m)
    if not cands:
        return
    sweep.shape(lab, "router", m, cands)
    rec = sweep.results[("router", m)]
    tb = 256 * 8192 * 4 / 1e6
    times = " ".join(f"{hl_name(c)}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0])[:10])
    log(f"hl router m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong; {tb:.1f} MB hi+lo) kind BM/BN/BK/split/warps/stages:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    hl_setup(state)
    log(f"base prepare + hi/lo split took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for m in (256, 128, 32, 16, 8):
        lab.section(f"hl router m{m}", sec_hl, sweep, m, min_left=12.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") in ("hl_gemm", "hl_swap"):
            log(f"V14 CHOICE {layer} m{m}: " + repr(c))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state

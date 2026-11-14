

# =============================================================================================
# fa3t: SGLang's FA3 main kernel (sgl_kernel flash_attn_with_kvcache, FP8 Q/K/V as SGLang runs it)
# behind one Triton prologue instead of SGLang's QK-norm, query/key/value casts and store_cache
# kernels (L16 profile, Fv16: ~8 us of 204 around the 187 us main kernel), with FA3's own tuning
# knobs (num_splits, pack_gqa; sgl_kernel/flash_attn.py: "Can be tuned for speed"). Routes
# FA3T_ROUTE[class] = (num_splits, pack_gqa) come from a lab; default (0, None) passes the step's
# precomputed scheduler_metadata exactly as SGLang does. Checked at warm-up against the Triton
# path like the CUDA routes (FP8 queries: tolerance SELF_CHECK_TOL_FP8).
# =============================================================================================
FA3T_ROUTE = {}


@triton.jit
def _prologue_f8(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
                 k_scale_ptr, v_scale_ptr,
                 HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr):
    # _prologue with the query stored as E4M3 of the normalized (RoPE'd) query, no qk factor: the
    # factor joins FA3's softmax scale, as in SGLang's call and the reference's FP8 rounding
    t = tl.program_id(0)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    base = qkv_ptr + t.to(tl.int64) * ROW
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    i = tl.arange(0, 32)
    q_e = tl.load(base + hq[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    q_o = tl.load(base + hq[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    k_e = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    k_o = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    v_e = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    v_o = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    q_r = tl.rsqrt((tl.sum(q_e * q_e, 1) + tl.sum(q_o * q_o, 1)) / 64.0 + EPS)
    k_r = tl.rsqrt((tl.sum(k_e * k_e, 1) + tl.sum(k_o * k_o, 1)) / 64.0 + EPS)
    q_e = q_e * q_r[:, None]
    q_o = q_o * q_r[:, None]
    k_e = k_e * k_r[:, None]
    k_o = k_o * k_r[:, None]
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(inv_freq_ptr + i)
        c = libdevice.cos(ang)[None, :]
        s = libdevice.sin(ang)[None, :]
        q_e, q_o = q_e * c - q_o * s, q_e * s + q_o * c
        k_e, k_o = k_e * c - k_o * s, k_e * s + k_o * c
    q_dst = q_out_ptr + t.to(tl.int64) * (HQ * 64) + hq[:, None] * 64 + 2 * i[None, :]
    tl.store(q_dst, q_e.to(tl.float8e4nv))
    tl.store(q_dst + 1, q_o.to(tl.float8e4nv))
    k_inv = 1.0 / tl.load(k_scale_ptr)
    v_inv = 1.0 / tl.load(v_scale_ptr)
    row = tl.load(loc_ptr + t)
    off = row * (HKV * 64) + hk[:, None] * 64 + 2 * i[None, :]
    tl.store(k_cache_ptr + off, (k_e * k_inv).to(tl.float8e4nv))
    tl.store(k_cache_ptr + off + 1, (k_o * k_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off, (v_e * v_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off + 1, (v_o * v_inv).to(tl.float8e4nv))


def run_fa3t(data, num_splits=0, pack_gqa=None):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    rope = bool(cfg["rope_theta"])
    q8 = torch.empty((tokens, hq, 64), dtype=FP8, device=device)
    _prologue_f8[(tokens,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                            q8, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                            HQ=hq, HKV=hkv, ROPE=rope, EPS=RMS_EPS, num_warps=4)
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    window = cfg["window"]
    default = num_splits == 0 and pack_gqa is None
    out = flash_attn_with_kvcache(
        q=q8, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"] if default else None,
        num_splits=num_splits, pack_gqa=pack_gqa, **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * 64:]
    o = rms_norm_gate(out.reshape(tokens, hq * 64).to(torch.bfloat16), gate, None, head_dim=64, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * 64)


_warm_up_fa3t_prev = _warm_up


def _warm_up():  # noqa: F811 (then each FA3T route checked against the Triton path)
    _warm_up_fa3t_prev()
    dev = torch.device(_WARM_DEVICE)
    for cls, (ns, pg) in sorted(FA3T_ROUTE.items()):
        full, verify, bclass = cls
        try:
            data = _self_check_inputs(bclass, 8 if verify else 1, full, dev)
            ours = run_fa3t(data, ns, pg).float()
            want = run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-6))
            if not err < SELF_CHECK_TOL_FP8:
                raise RuntimeError(f"differs from the Triton path: rel. L2 {err:.2e}")
        except Exception as exc:
            print(f"fa3t: route {cls} -> {(ns, pg)} dropped: {type(exc).__name__}: {str(exc)[-200:]}")
            FA3T_ROUTE.pop(cls)


_kernel_fn_fa3t_prev = kernel_fn


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    route = FA3T_ROUTE.get(shape_class(data["config"]))
    if route is not None and CUDA_ROUTE.get(shape_class(data["config"])) is None:
        return run_fa3t(data, *route)
    return _kernel_fn_fa3t_prev(data)

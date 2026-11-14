

# ---------------------------------------------------------------------------------------------
# Lab L13 (check runs only): programmatic dependent launch (make_pdl.py). (A) v2.2's Triton chain
# vs its PDL variant (late trigger = after the producer's stores, early = first thing) on all 12
# shapes; (B) decode with CUDA attention: D2 / D3 between the plain Triton prologue and combine vs
# D2p / D3p between the PDL ones. Error vs the FP32 reference on two alternating input sets (a read
# before griddepcontrol.wait would see the other set's data), graph timing with a cold L2.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 420.0
_MODS = {}


def _build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


def _dec_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _cuda_dec_run(data, kind, cps, warps, stages, early=False):
    """kind: dec3 / dec4 (plain chain) or dec3p / dec4p (PDL chain, early or late triggers)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec_nsplit(cfg, max_pages, cps, warps)
    rope = bool(cfg["rope_theta"])
    pdl = kind.endswith("p")
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    pro = _prologue_p if pdl else _prologue
    extra = dict(TRIGGER_EARLY=early) if pdl else {}
    pro[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                  q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                  HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4, **extra)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
            cfg["sm_scale"], warps, stages)
    fn = getattr(_MODS[kind], "attend_" + kind)
    fn(*args, int(early)) if pdl else fn(*args)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    comb = _combine_p if pdl else _combine
    cextra = dict(launch_pdl=True) if pdl else {}
    comb[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                      HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4, **cextra)
    return out


def _err_two_sets(fn_of, d_a, d_b, ref_b):
    """Run on set A then set B (same table, different queries and new rows); error of B vs its FP32
    reference: catches a read that slipped ahead of griddepcontrol.wait (stale partials or q)."""
    fn_of(d_a)
    out = fn_of(d_b)
    return float((out.float() - ref_b).norm() / ref_b.norm())


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1313)
    for kind, src, decl, extra in (("dec3", CUDA_DEC3_SRC, CUDA_DEC3_DECL, ["attend_dec3_setup"]),
                                   ("dec3p", CUDA_DEC3P_SRC, CUDA_DEC3P_DECL, ["attend_dec3p_setup"]),
                                   ("dec4", CUDA_DEC4_SRC, CUDA_DEC4_DECL, ["attend_dec4_setup"]),
                                   ("dec4p", CUDA_DEC4P_SRC, CUDA_DEC4P_DECL, ["attend_dec4p_setup"])):
        try:
            mod = _build("okc_l13_" + kind, decl, src, ["attend_" + kind] + extra)
            rc = getattr(mod, extra[0])()
            if rc:
                _log(f"{kind} setup refused {rc}: skipped")
                continue
            _MODS[kind] = mod
        except Exception as exc:
            _log(f"{kind} build FAILED {_why(exc)}")
    rows = 32 * scored_max_pages(32, 8)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L13 start: budget {_LAB_BUDGET_S:.0f}s; line = shape path: us err(B after A)")
    shapes = [(1, 1, 2048), (16, 1, 2048), (32, 1, 2048), (1, 1, 0), (16, 1, 0), (32, 1, 0),
              (1, 8, 2048), (16, 8, 2048), (32, 8, 2048), (1, 8, 0), (16, 8, 0), (32, 8, 0)]
    dec_variants = {16: [(4, 4, 4), (4, 4, 3), (8, 4, 3)], 32: [(8, 4, 3), (4, 4, 3), (8, 4, 4)],
                    1: [(4, 4, 4), (8, 4, 3)]}
    for batch, length, window in shapes:
        name = f"{'S' if window else 'F'}{'v' if length > 1 else 'd'}{batch}"
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            d_b = _lab_inputs(batch, length, window, k_cache, v_cache, gen)
            # set A: same table and cache rows, other queries and new keys/values; A's new rows
            # land where B's go, and B's run writes them back before B reads them.
            d_a = dict(d_b)
            d_a["qkv"] = torch.randn(d_b["qkv"].shape, generator=gen, device=dev).to(torch.bfloat16)
            run(d_b)
            ref_b = _lab_reference(d_b)
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        paths = [("v2.2", lambda d: run(d)), ("pdl-late", lambda d: run_pdl(d, early=False)),
                 ("pdl-early", lambda d: run_pdl(d, early=True))]
        if length == 1:
            for cps, warps, stages in dec_variants[batch] if not window else [(4, 4, 3)]:
                for kind in ("dec3", "dec3p", "dec4", "dec4p"):
                    if kind not in _MODS:
                        continue
                    for early in ((False, True) if kind.endswith("p") else (False,)):
                        tag = f"{kind}{'-e' if early else ''} c{cps}w{warps}s{stages}"
                        paths.append((tag, lambda d, k=kind, c=cps, w=warps, s=stages, e=early:
                                      _cuda_dec_run(d, k, c, w, s, e)))
        for tag, fn in paths:
            if lab.left() < 15:
                break
            try:
                err = _err_two_sets(fn, d_a, d_b, ref_b)
                us = lab.time_graph(lambda: fn(d_b))
                _log(f"{name} {tag}: {us:.1f}us err {err:.1e}{'  <-- WRONG' if not err < 2e-2 else ''}")
            except Exception as exc:
                _log(f"{name} {tag}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

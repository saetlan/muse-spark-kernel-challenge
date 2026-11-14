

# ---------------------------------------------------------------------------------------------
# Lab L5 (check runs only): S1 = A5b (whole-row loads + stacked heads + rows two chunks ahead,
# attention/local/cuda_dec2.py appended above) against the submission's Triton path on the six
# decode shapes, then gather-engine microbenchmarks on the same random rows (A19:
# attention/local/cuda_gather.py): ld whole rows / 64 B halves / split-32, cp.async ring, bulk
# copies. Two load_inline modules, so a PTX problem in the bulk-copy code (built separately,
# last) can't take A5b down with it.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
_MODS = {}


def _build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


def _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _dec2_run(data, ctas_per_sm, warps):
    """Triton prologue, A5b attention, Triton combine (q_len 1, groups of 8)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _MODS["main"].attend_dec2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                              data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                              cfg["page_size"], cfg["window"], cfg["sm_scale"], warps)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _sec_a5b(lab, k_cache, v_cache, gen):
    m = _MODS["main"]
    _log(f"A5b attrs (regs, local B, smem B): w4 {m.attend_dec2_attrs(4)} w8 {m.attend_dec2_attrs(8)}")
    variants = [(2, 4), (4, 4), (8, 4), (2, 8), (4, 8), (8, 8)]
    for batch, window in [(16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)]:
        name = f"{'S' if window else 'F'}d{batch}"
        if lab.left() < 20:
            _log("A5b: out of budget")
            return
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} v2.2 path: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for cps, warps in variants:
            try:
                ns = _dec2_nsplit(data["config"], data["page_table"].shape[1], cps, warps)
                err = float((_dec2_run(data, cps, warps).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _dec2_run(data, cps, warps))
                _log(f"{name} A5b c{cps} w{warps}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
            except Exception as exc:
                _log(f"{name} A5b c{cps} w{warps}: FAILED {_why(exc)}")


def _sec_gather(lab, k_cache, v_cache, gen):
    m = _MODS["main"]
    rows = k_cache.shape[0]
    idx = torch.randperm(rows, generator=gen, device=k_cache.device).to(torch.int32)
    out = torch.zeros((4,), dtype=torch.int32, device=k_cache.device)
    nbytes = rows * 256
    for mode, label in ((0, "whole"), (1, "half64"), (2, "split32")):
        for unroll in (4, 8):
            us = lab.time_graph(lambda: m.gather_ld(k_cache, v_cache, idx, out, mode, unroll, 132 * 8, 256))
            _log(f"G ld {label} u{unroll}: {us:.1f}us {nbytes / us / 1e6:.2f}TB/s")
    _log(f"G setup {m.gather_setup(227 * 1024)}")
    for stages in (2, 4, 6):
        for warps, blocks in ((4, 132 * 2), (4, 132 * 4), (8, 132)):
            if warps * stages * 4096 > 227 * 1024:
                continue
            us = lab.time_graph(lambda: m.gather_cp(k_cache, v_cache, idx, out, stages, blocks, warps))
            _log(f"G cp s{stages} w{warps} b{blocks}: {us:.1f}us {nbytes / us / 1e6:.2f}TB/s")


def _sec_bulk(lab, k_cache, v_cache, gen):
    _MODS["bulk"] = _build("okc_l5_bulk", CUDA_BULK_DECL, CUDA_BULK_SRC, ["bulk_setup", "gather_bulk"])
    m = _MODS["bulk"]
    rows = k_cache.shape[0]
    idx = torch.randperm(rows, generator=gen, device=k_cache.device).to(torch.int32)
    out = torch.zeros((4,), dtype=torch.int32, device=k_cache.device)
    err = torch.zeros((1,), dtype=torch.int32, device=k_cache.device)
    _log(f"bulk setup {m.bulk_setup(227 * 1024)}")
    nbytes = rows * 256
    for stages in (2, 4):
        for warps, blocks in ((4, 132 * 2), (4, 132 * 4), (8, 132)):
            us = lab.time_graph(lambda: m.gather_bulk(k_cache, v_cache, idx, out, err, stages, blocks, warps))
            torch.cuda.synchronize()
            _log(f"G bulk s{stages} w{warps} b{blocks}: {us:.1f}us {nbytes / us / 1e6:.2f}TB/s timeout {int(err.item())}")


def _section(lab, name, fn, *args):
    if lab.left() < 15:
        _log(f"{name}: skipped, budget spent")
        return
    t = _time.monotonic()
    try:
        fn(lab, *args)
    except Exception as exc:
        _log(f"{name}: FAILED {_why(exc)}")
    torch.cuda.synchronize()
    _log(f"{name}: {_time.monotonic() - t:.1f}s")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5005)
    try:
        _MODS["main"] = _build("okc_l5_main", CUDA_DEC2_DECL + "\n" + CUDA_GATHER_DECL,
                               CUDA_DEC2_SRC + "\n" + CUDA_GATHER_SRC.replace("#include <torch/extension.h>", "")
                               .replace("#include <ATen/cuda/CUDAContext.h>", ""),
                               ["attend_dec2", "attend_dec2_attrs", "gather_ld", "gather_setup", "gather_cp"])
    except Exception as exc:
        _log(f"build main FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L5 start: budget {_LAB_BUDGET_S:.0f}s; pool {rows} rows")
    _section(lab, "A5b", _sec_a5b, k_cache, v_cache, gen)
    _section(lab, "gather ld/cp", _sec_gather, k_cache, v_cache, gen)
    _section(lab, "gather bulk", _sec_bulk, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()

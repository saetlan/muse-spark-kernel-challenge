"""CPU emulation of submission.py's (v2) three Triton kernels, program by program, checked against
reference.py on the judge's test cases (CPU, small enough ones) and the scored shapes' logic.

It imports submission.py with a stub `triton` (no Triton on the Mac), so `plan()` and the
constants are the submission's own. Each emulated kernel follows the Triton code line by line:
same index math, same masks, same roundings (BF16 queries, FP8 cache, BF16 P, FP32 rest).

Run: .venv/bin/python attention/local/emulate.py [--quick]
"""

import importlib.util
import math
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, PROB)


def _stub_triton():
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    for name in ("float32", "bfloat16", "float8e4nv", "int64"):
        setattr(tl, name, name)
    extra = types.ModuleType("triton.language.extra")
    extra.libdevice = types.ModuleType("triton.language.extra.libdevice")
    extra.cuda = types.ModuleType("triton.language.extra.cuda")       # PDL intrinsics: no-ops here
    extra.cuda.gdc_wait = lambda: None
    extra.cuda.gdc_launch_dependents = lambda: None
    tl.extra = extra
    tr = types.ModuleType("triton")

    def jit(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

    tr.jit = jit
    tr.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    tr.language = tl
    sys.modules.update({"triton": tr, "triton.language": tl, "triton.language.extra": extra,
                        "triton.language.extra.libdevice": extra.libdevice, "triton.language.extra.cuda": extra.cuda})
    # SGLang's ops (the hybrid fallback): not installed on the Mac; names only.
    for mod, names in {"sgl_kernel.flash_attn": ["flash_attn_with_kvcache"],
                       "sglang.kernels.ops.kvcache.kvcache": ["store_cache"],
                       "sglang.kernels.ops.muse_spark_v12.fused_qk_norm": ["FusedQKNorm"],
                       "sglang.kernels.ops.muse_spark_v12.norm_gate": ["rms_norm_gate"]}.items():
        parts = mod.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        for n in names:
            setattr(sys.modules[mod], n, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("SGLang stub called")))


def load_submission(path=os.path.join(PROB, "submission.py")):
    _stub_triton()
    spec = importlib.util.spec_from_file_location("v1_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prologue(data, sub):
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"].reshape(-1, (2 * hq + 2 * hkv) * 64).float()
    t = qkv.shape[0]
    q = qkv[:, :hq * 64].view(t, hq, 32, 2)
    k = qkv[:, hq * 64:(hq + hkv) * 64].view(t, hkv, 32, 2)
    v = qkv[:, (hq + hkv) * 64:(hq + 2 * hkv) * 64].view(t, hkv, 32, 2)
    q_e, q_o, k_e, k_o = q[..., 0], q[..., 1], k[..., 0], k[..., 1]
    q_r = torch.rsqrt(((q_e * q_e).sum(-1) + (q_o * q_o).sum(-1)) / 64.0 + sub.RMS_EPS)
    k_r = torch.rsqrt(((k_e * k_e).sum(-1) + (k_o * k_o).sum(-1)) / 64.0 + sub.RMS_EPS)
    q_e, q_o = q_e * q_r[..., None] * cfg["qk_scale_factor"], q_o * q_r[..., None] * cfg["qk_scale_factor"]
    k_e, k_o = k_e * k_r[..., None], k_o * k_r[..., None]
    if cfg["rope_theta"]:
        inv = sub._inv_freq(cfg["rope_theta"], "cpu")
        ang = data["positions"].float()[:, None] * inv[None, :]
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        q_e, q_o = q_e * c - q_o * s, q_e * s + q_o * c
        k_e, k_o = k_e * c - k_o * s, k_e * s + k_o * c
    q_out = torch.stack([q_e, q_o], -1).reshape(t, hq, 64).to(torch.bfloat16)
    kf = torch.stack([k_e, k_o], -1).reshape(t, hkv, 64) / data["k_scale"]
    vf = v.reshape(t, hkv, 64) / data["v_scale"]
    kc = data["k_cache"].view(-1, hkv, 64)
    vc = data["v_cache"].view(-1, hkv, 64)
    kc[data["cache_loc"]] = kf.to(torch.float8_e4m3fn)
    vc[data["cache_loc"]] = vf.to(torch.float8_e4m3fn)
    return q_out


def attend(data, q, sub, tune=None):
    """_attend, program by program: a pair of KV heads, one slice, per-head dots."""
    cfg = data["config"]
    hq, hkv, L, page, window = (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"], cfg["page_size"],
                                cfg["window"])
    table = data["page_table"]
    max_pages = table.shape[1]
    nsplit, BLOCK, _, _, cb, MP = sub.plan(cfg, max_pages, tune)
    G = hq // hkv
    ML = L * G
    T = cfg["batch"] * L
    o_part = torch.full((T * hq * nsplit, 64), float("nan"))   # NaN: any slot never written shows up
    m_part = torch.full((T * hq * nsplit,), float("nan"))
    l_part = torch.full_like(m_part, float("nan"))
    kc = data["k_cache"].view(-1, hkv * 64)
    vc = data["v_cache"].view(-1, hkv * 64)
    r = torch.arange(MP)
    r_ok = r < ML
    r_l, r_g = r // G, r % G
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * sub.LOG2E
    qf = q.reshape(T * hq, 64).float()
    for b in range(cfg["batch"]):
        n = int(data["cache_seqlens"][b])
        row_pos = n - L + r_l
        lo = max(n - L - window + 1, 0) if window > 0 else 0
        per = math.ceil(math.ceil((n - lo) / nsplit) / BLOCK) * BLOCK
        tok = b * L + r_l
        for pair in range(hkv // 2):
            kv0 = pair * 2
            heads = [kv0 * G + r_g, kv0 * G + r_g + G]
            qs = [torch.where(r_ok[:, None], qf[(tok * hq + h).clamp(max=T * hq - 1)], 0.0) for h in heads]
            for split in range(nsplit):
                start = lo + split * per
                end = min(start + per, n)
                st = [[torch.full((MP,), -1.0e30), torch.zeros(MP), torch.zeros(MP, 64)] for _ in range(2)]
                for s0 in range(start, end, BLOCK):
                    s = s0 + torch.arange(BLOCK)
                    live = s < end
                    pg = torch.where(live, table[b][(s // page).clamp(max=max_pages - 1)], 0).long()
                    assert bool((pg[live] >= 0).all()), "page table -1 read inside the live range"
                    rows = pg * page + s % page
                    vis = live[None, :] & (s[None, :] <= row_pos[:, None])
                    if window > 0:
                        vis &= s[None, :] > row_pos[:, None] - window
                    for j in range(2):
                        cols = slice((kv0 + j) * 64, (kv0 + j + 1) * 64)
                        k = torch.where(live[:, None], kc[rows][:, cols].float(), 0.0)
                        v = torch.where(live[:, None], vc[rows][:, cols].float(), 0.0)
                        m_i, l_i, acc = st[j]
                        sc = torch.where(vis, (qs[j] @ k.T) * qk_scale, float("-inf"))
                        m_new = torch.maximum(m_i, sc.max(1).values)
                        alpha = torch.exp2(m_i - m_new)
                        p = torch.exp2(sc - m_new[:, None])
                        st[j] = [m_new, l_i * alpha + p.sum(1), acc * alpha[:, None] + p.to(torch.bfloat16).float() @ v]
                for j in range(2):
                    part = ((tok * hq + heads[j]) * nsplit + split)[r_ok]
                    o_part[part] = st[j][2][r_ok]
                    m_part[part] = st[j][0][r_ok]
                    l_part[part] = st[j][1][r_ok]
    assert not torch.isnan(m_part).any(), "a partial slot was never written"
    return o_part, m_part, l_part, nsplit, cb


def combine(data, o_part, m_part, l_part, nsplit, cb, sub):
    """_combine: one pass in chunks of cb slices, running max and rescaled sums."""
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    T = cfg["batch"] * cfg["q_len"]
    m = m_part.view(T * hq, nsplit)
    lsum = l_part.view(T * hq, nsplit)
    o = o_part.view(T * hq, nsplit, 64)
    big = torch.full((T * hq,), -1.0e30)
    den = torch.zeros(T * hq)
    num = torch.zeros(T * hq, 64)
    for s0 in range(0, nsplit, cb):
        mc, lc, oc = m[:, s0:s0 + cb], lsum[:, s0:s0 + cb], o[:, s0:s0 + cb]
        new = torch.maximum(big, mc.max(1).values)
        scale = torch.exp2(big - new)
        w = torch.exp2(mc - new[:, None])
        den = den * scale + (w * lc).sum(1)
        num = num * scale[:, None] + (w[..., None] * oc).sum(1)
        big = new
    a = num / den[:, None] * float(data["v_scale"])
    a = a * torch.rsqrt((a * a).sum(-1, keepdim=True) / 64.0 + sub.RMS_EPS)
    gate = data["qkv"].reshape(T, -1)[:, (hq + 2 * hkv) * 64:].float().reshape(T * hq, 64)
    a = a / (1.0 + torch.exp(-gate))
    return a.reshape(cfg["batch"], cfg["q_len"], hq * 64).to(torch.bfloat16)


def run(data, sub, tune=None):
    q = prologue(data, sub)
    return combine(data, *attend(data, q, sub, tune), sub)


def window_rows_case(ref, page):
    """Sliding verify block where, for every even row l of the block, the key just outside that
    row's window (position p + l - 2048) points along the row's own queries (as the kernel sees
    them: normalized, RoPE'd). Including it by mistake dominates that row's softmax, so a
    per-row window off-by-one fails the row check. The judge's "edges" only plants keys for
    the block's first token."""
    data = ref.generate_input(batch=2, ctx=5000, heads="tp8", layer="sliding", q_len=8, page=page,
                              dist="soft", seed=77, device="cpu")
    cfg = data["config"]
    hq, hkv, L = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    q_in = ref.split_qkv(data)[0]                       # [B, L, Hq, 64]
    k_flat = data["k_cache"].view(-1, hkv, 64)
    for b in range(cfg["batch"]):
        p = int(data["seq_lens"][b])
        for l in range(0, L, 2):
            qn = ref._rope(ref._rms(q_in[b, l]), torch.tensor(p + l), ref.ROPE_THETA)   # [Hq, 64]
            t = p + l - ref.SLIDING_WINDOW
            row = int(data["page_table"][b, t // page]) * page + t % page
            k_flat[row] = ref._quant(qn[:: hq // hkv], data["k_scale"])
    return data


def late_key_case(ref):
    """bs1 full layer with more slices than one merge chunk, and a key aligned with each KV
    head's queries near the end of the context: the softmax maximum then sits in the merge's
    last chunk, so every earlier chunk must be rescaled."""
    data = ref.generate_input(batch=1, ctx=40000, heads="tp8", layer="full", page=1, dist="soft",
                              seed=78, device="cpu")   # 156 slices, 125 holding keys: 2 merge chunks
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qn = ref._rms(ref.split_qkv(data)[0][0, 0])             # [Hq, 64], no RoPE on full layers
    t = int(data["seq_lens"][0]) - 10
    row = int(data["page_table"][0, t])
    data["k_cache"].view(-1, hkv, 64)[row] = ref._quant(qn[:: hq // hkv], data["k_scale"])
    return data


def main():
    import reference as ref

    sub = load_submission()
    quick = "--quick" in sys.argv
    # The judge's 16 tests (TASK.md), minus the ones too big for a CPU run, plus small
    # stand-ins at the scored shapes' structure (batch 16/32, page 1, agentic spread).
    cases = [
        dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
        dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
        dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
        dict(batch=4, ctx=3000, heads="tp8", layer="full"),
        dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
        dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
        dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
        dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
        dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
        dict(batch=2, ctx=4000, heads="dp", layer="full", q_len=8, dist="soft"),
        dict(batch=2, ctx=3000, heads="dp", layer="sliding", dist="coherent"),
        dict(batch=16, ctx=2500, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
        dict(batch=2, ctx=12000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent"),
        # bs1 full with more slices than one merge chunk (78 > 64), as the scored bs1 full shape (312)
        dict(batch=1, ctx=20000, heads="tp8", layer="full", page=1, dist="coherent"),
    ]
    if not quick:
        cases += [dict(batch=8, ctx=8192, heads="dp", layer="full", spread="mixed"),
                  dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent")]
    bad = 0
    for i, case in enumerate(cases):
        data = ref.generate_input(seed=1000 + i, device="cpu", **case)
        out = run(data, sub)
        # As the judge does: collect the output and our new cache rows, then check them.
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + str(case) + " -> " + msg)
        bad += not ok
    data = late_key_case(ref)
    out = run(data, sub)
    ok, msg = ref.check_implementation(data, ref.collect(data, out))
    print(("PASS " if ok else "FAIL ") + "full bs1, maximum in the last merge chunk -> " + msg)
    bad += not ok
    for page in (1, 64):
        data = window_rows_case(ref, page)
        out = run(data, sub)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"verify window edges per row (page {page}) -> " + msg)
        bad += not ok
    for cfg_case in [(b, l, w) for b in (1, 16, 32) for l in (1, 8) for w in (2048, 0)]:
        b, l, w = cfg_case
        cfg = dict(batch=b, q_len=l, num_q_heads=16, num_kv_heads=2, page_size=1, window=w)
        print("plan (batch, q_len, window)", cfg_case, "-> (nsplit, block, warps, stages, cb, mp)",
              sub.plan(cfg, sub.scored_max_pages(b, l)))
    print("emulate:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

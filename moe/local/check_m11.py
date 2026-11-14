"""Local (CPU) checks for M11 (moe/m11/m11.py, built into submission_m11.py). No GPU here, so the
kernel's arithmetic is emulated with its own formulas:
1. m11_shuffle + the kernel's decode (dec<R>, scales(), hmul2) + the PTX m16n8k16 fragment rules +
   the kernel's activation addressing: every weight lands at the K position its activation is read
   from, exactly (value * 2^-62);
2. the routing scan and warp task loop: every local pair computed once per tile, nothing else;
3. the whole M11 pipeline (bf16 act and y, post-norm) vs an FP32 reference at the real K = 4096.

Run: .venv/bin/python moe/local/check_m11.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_v0_lab as base  # noqa: E402

TOP_K = 8
K = 4096
MAXP = 320  # set from the module in __main__ (m11.M11_MAX_TOKENS)


def bf16_pair(reg32):
    return check_m2._bf16_pair(reg32.to(torch.int64) & 0xFFFFFFFF)


def pair_bits(bf2):
    return check_m2._pair_bits(bf2)


def dec(w, r):
    """The kernel's dec<R>: x = w << 4R; (x & 0x80008000) | ((x >> 6) & 0x01C001C0)."""
    x = (w << (4 * r)) & 0xFFFFFFFF
    return (x & 0x80008000) | ((x >> 6) & 0x01C001C0)


def scale_pairs(v):
    """The kernel's scales(): low byte -> row g, next byte -> row g + 8; each (s * 2^64) in both lanes."""
    lo = (v & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn).float()
    hi = ((v >> 8) & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn).float()
    lo_b = (lo * 2.0**64).to(torch.bfloat16)  # f16 -> f32 -> * 2^64 -> bf16: exact, so one rounding of an exact value
    hi_b = (hi * 2.0**64).to(torch.bfloat16)
    return torch.stack((lo_b, lo_b), -1), torch.stack((hi_b, hi_b), -1)


def hmul2(a_bits, s_pair):
    return (bf16_pair(a_bits).float() * s_pair.float()).to(torch.bfloat16)


def dequant_from_fragments(words, scales):
    """words int32 [RT, KG, 32, 4], scales int16 [RT, KG, 32] -> W [RT*16, KG*64] float32 as the MMAs see it
    (value * 2^-62), placed at the activation K position each A element meets."""
    rt, kg = words.shape[:2]
    w = words.to(torch.int64) & 0xFFFFFFFF  # [rt, kg, lane, j]
    v = scales.to(torch.int64) & 0xFFFF  # [rt, kg, lane]
    sg, sg8 = scale_pairs(v)  # [rt, kg, lane, 2]
    out = torch.zeros((rt * 16, kg * 64))
    lane = torch.arange(32)
    g, t = lane // 4, lane % 4
    for j in range(4):
        regs = [hmul2(dec(w[..., j], r), sg if r % 2 == 0 else sg8).float() for r in range(4)]  # each [rt, kg, lane, 2]
        # PTX m16n8k16 A: reg0 = (row g, cols 2t, 2t+1), reg1 = (g+8, 2t..), reg2 = (g, 2t+8..), reg3 = (g+8, 2t+8..).
        # The kernel's B load: thread (n, t) reads activations 16 t + 4 j + {0,1} for cols 2t, 2t+1 and
        # 16 t + 4 j + {2,3} for cols 2t+8, 2t+9, so col 2t + h -> k 16 t + 4 j + h, col 2t + 8 + h -> + 2 + h.
        for r, (row_off, k_off) in enumerate(((0, 0), (8, 0), (0, 2), (8, 2))):
            for h in range(2):
                rows = (torch.arange(rt)[:, None, None] * 16 + g[None, None, :] + row_off)
                ks = (torch.arange(kg)[None, :, None] * 64 + 16 * t[None, None, :] + 4 * j + k_off + h)
                rows, ks = torch.broadcast_tensors(rows, ks)
                out[rows.reshape(-1), ks.reshape(-1)] = regs[r][..., h].reshape(-1)
    return out


def check_shuffle(m11):
    from nvfp4 import dequantize, random_weights
    gen = torch.Generator().manual_seed(11)
    packed, sf, _ = random_weights(256, K, generator=gen, device="cpu")
    sf = lab_all_scales(sf.shape)  # every E4M3 value incl. subnormals and zero
    words, scales = m11.m11_shuffle(packed, sf)
    assert words.shape == (16, 64, 32, 4) and scales.shape == (16, 64, 32)
    got = dequant_from_fragments(words, scales)
    want = dequantize(packed, sf, 1.0) * 2.0**-62
    assert torch.equal(got, want), f"shuffle/fragment mismatch, max diff {float((got - want).abs().max())}"
    print("ok: m11_shuffle + dec<R> + scales() + PTX fragments + activation addressing = every weight exactly, "
          "all 127 scales")


def lab_all_scales(shape):
    n = shape[0] * shape[1]
    return (torch.arange(n) % 127).to(torch.uint8).view(torch.float8_e4m3fn).view(shape)


def emulate_tasks(ids_flat, start, per_expert, grid, ks, nt, maxp):
    """The kernel's routing scan, group task loop and 8 * nt pair chunks -> (expert, tile, chunk pairs)
    in the order the groups run them; ks warps of a group share each task."""
    local = ids_flat.long() - start
    lists = [[] for _ in range(32)]
    for p, e in enumerate(local.tolist()):
        if 0 <= e < 32:
            lists[e].append(p)
    active = [e for e in range(32) if lists[e]]
    groups_per_cta = 4 // ks
    n_tasks = len(active) * per_expert
    out = []
    for cta in range(grid):
        for grp in range(groups_per_cta):
            for task in range(cta * groups_per_cta + grp, n_tasks, grid * groups_per_cta):
                e = active[task // per_expert]
                pairs = lists[e][:maxp]
                for c0 in range(0, len(pairs), 8 * nt):
                    out.append((e, task % per_expert, pairs[c0:c0 + 8 * nt]))
    return out


def emu_gemm(a, bank, name, ids, a_per_pair, gated, cfg, grid):
    """What one launch writes, with the kernel's tasks, chunks, fragment math and epilogue. cfg is
    s1 (gated) or s2: tt row tiles per task, nt pair tiles per chunk, ks warps splitting K."""
    rt_total = bank["rt_" + name]
    tt = 2 if gated else cfg["tt"]
    per_expert = rt_total // 2 if gated else rt_total // tt
    tasks = emulate_tasks(ids, bank["start"], per_expert, grid, cfg["ks"], cfg["nt"], MAXP)
    n_cols = bank["inter"] if gated else bank["hidden"]
    out = torch.zeros((ids.numel(), n_cols))
    seen = {}
    deq = {}
    c = bank["c13" if gated else "c2"]
    for e, tile, pairs in tasks:
        if e not in deq:
            deq[e] = dequant_from_fragments(bank[name][e], bank[name + "_s"][e])
        rows = [tile, tile + rt_total // 2] if gated else [tile * tt + i for i in range(tt)]
        p = torch.tensor(pairs)
        av = a[p if a_per_pair else p // TOP_K].float()
        # ks warps each sum a K share; their partials are added before the epilogue (same dot product).
        shares = torch.arange(K).chunk(cfg["ks"])
        accs = [sum(av[:, ks_] @ deq[e][16 * r:16 * r + 16, ks_].T for ks_ in shares) * c[e] for r in rows]
        if gated:
            res = [accs[0] * torch.sigmoid(accs[0]) * accs[1]]
            cols = [16 * tile + torch.arange(16)]
        else:
            res = accs
            cols = [16 * r + torch.arange(16) for r in rows]
        for rv, cv in zip(res, cols):
            out[p[:, None], cv[None, :]] = rv.to(torch.bfloat16).float()
        for q in pairs:
            for cv in cols:
                key = (q, int(cv[0]) // 16)
                seen[key] = seen.get(key, 0) + 1
    local = [q for q, e in enumerate((ids.long() - bank["start"]).tolist()) if 0 <= e < 32]
    n_tiles = n_cols // 16
    want = {(q, tl_) for q in local for tl_ in range(n_tiles)}
    assert set(seen) == want and set(seen.values()) == {1}, "task coverage is wrong"
    return out.to(torch.bfloat16)


def check_pipeline(m11):
    hidden = inter = K
    weights = check_m2.small_weights(32, hidden, inter, seed=13)
    bank = m11.m11_bank(weights, 0, 32)
    norm_w = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float()
    gen = torch.Generator().manual_seed(17)
    worst = 0.0
    cfgs = [(dict(nt=1, d=8, ks=1, c=1), dict(tt=1, nt=1, d=8, ks=1, c=1)),
            (dict(nt=1, d=4, ks=4, c=1), dict(tt=2, nt=1, d=4, ks=2, c=1)),
            (dict(nt=2, d=2, ks=2, c=1), dict(tt=2, nt=2, d=2, ks=4, c=1))]
    for (tokens, skew, grid), (s1, s2) in zip(((1, False, 132), (3, False, 7), (8, False, 132), (8, True, 2),
                                               (20, True, 3), (40, True, 5), (24, False, 132)), cfgs * 3):
        x = torch.randn((tokens, hidden), generator=gen).to(torch.bfloat16)
        if skew:  # every token on experts 0-7: tokens pairs per expert, several 8 * nt chunks
            ids = torch.arange(TOP_K, dtype=torch.int32).repeat(tokens, 1)
        else:
            ids = torch.stack([torch.randperm(64, generator=gen)[:TOP_K] for _ in range(tokens)]).to(torch.int32)
        tw = torch.rand((tokens, TOP_K), generator=gen)
        tw = tw / tw.sum(-1, keepdim=True)
        flat = ids.reshape(-1)
        act = emu_gemm(x, bank, "w13", flat, False, True, s1, grid)
        y = emu_gemm(act, bank, "w2", flat, True, False, s2, grid)
        out = torch.zeros((tokens, hidden))
        for p, e in enumerate(flat.tolist()):
            if e < 32:
                v = y[p].float() * norm_w
                out[p // TOP_K] += v * torch.rsqrt(v.pow(2).mean() + 1e-8) * tw.reshape(-1)[p]
        ref = check_m2.fp32_reference(x, ids, tw, weights, 0, 32, weights["post_norm_w"])
        if ref.norm() == 0:
            continue
        err = check_m2.rel_l2(out.to(torch.bfloat16), ref)
        worst = max(worst, err)
        assert err < 1e-2, (tokens, skew, err)
    print(f"ok: emulated M11 pipeline vs FP32 reference at K = 4096, 7 routings (skewed up to 40 pairs per "
          f"expert, odd grids, ks 1/2/4, tt 1/2, nt 1/2): "
          f"worst rel L2 {worst:.2e}")


def m12_sequences(counts, per_expert, grid, groups, grp, nt, steps, r, maxp):
    """One warp of tma_kernel, transliterated: the refill cursor's issue order (flat advance, as in
    issue()) and the consumer's nested-loop order, each as (task, c0, step, slot)."""
    active = [e for e in range(32) if counts[e] > 0]
    n_tasks = len(active) * per_expert
    first, stride = grp, grid * groups  # blockIdx.x = 0; other CTAs differ only by first
    issued = []
    p = {"task": first, "c0": 0, "step": 0}

    def issue(slot):
        if p["task"] >= n_tasks:
            return
        issued.append((p["task"], p["c0"], p["step"], slot))
        e = active[p["task"] // per_expert]
        p["step"] += 1
        if p["step"] == steps:
            p["step"] = 0
            p["c0"] += 8 * nt
            if p["c0"] >= min(counts[e], maxp):
                p["c0"] = 0
                p["task"] += stride

    for s in range(r):
        issue(s)
    consumed = []
    seq = 0
    task = first
    while task < n_tasks:
        e = active[task // per_expert]
        total = min(counts[e], maxp)
        for c0 in range(0, total, 8 * nt):
            for st in range(steps):
                slot = seq % r
                assert len(issued) > seq, "a warp would wait on a slot that was never filled (hang)"
                consumed.append((task, c0, st, slot))
                issue(slot)
                seq += 1
        task += stride
    return issued, consumed


def check_m12_sequence(m11):
    """For every M12 instance shape and random routings: the refill cursor issues exactly the slots
    the warp consumes, in the same order and slot, R ahead, and nothing is left in flight."""
    gen = torch.Generator().manual_seed(23)
    cases = 0
    for gated, tt, nt, ks, ch, r in m11.M12_INSTANCES:
        steps = (64 // ks) // ch
        per_expert = 256 if gated else 256 // tt
        for trial in range(6):
            counts = [0] * 32
            n_exp = int(torch.randint(1, 33, (1,), generator=gen))
            for e in torch.randperm(32, generator=gen)[:n_exp].tolist():
                counts[e] = int(torch.randint(1, 321, (1,), generator=gen)) if trial % 2 else int(torch.randint(1, 9, (1,), generator=gen))
            grid = [1, 7, 132, 528][trial % 4]
            groups = 4 // ks
            for grp in range(groups):
                issued, consumed = m12_sequences(counts, per_expert, grid, groups, grp, nt, steps, r, m11.M11_MAX_TOKENS)
                assert issued == consumed, (gated, tt, nt, ks, ch, r, trial, grp)
                cases += 1
    print(f"ok: M12 refill cursor = consumer order for {len(m11.M12_INSTANCES)} instances x 6 routings "
          f"({cases} warp sequences, up to 320 pairs per expert): every wait has its slot, none left in flight")


def check_submission_m11():
    mod = base.load("submission_m11")
    text = (base.MOE / "submission_m11.py").read_text()
    assert text.splitlines()[0] == base.PROBLEM_LINE
    v1 = (base.MOE / "submission.py").read_text()
    v1 = v1.replace("\ndef prepare(weights", "\ndef _v1_prepare(weights", 1)
    v1 = v1.replace("\ndef kernel_fn(data, state", "\ndef _v1_kernel_fn(data, state", 1)
    assert text.startswith(v1.rstrip()), "v1 part must be byte for byte"
    for name in ("prepare", "kernel_fn", "_v1_prepare", "_v1_kernel_fn", "m11_shuffle", "m11_bank", "m11_run"):
        assert callable(getattr(mod, name)), name
    print("ok: submission_m11.py = submission.py (v1, renamed entry points) + M11 section")
    return mod


def check_dispatch(m11):
    """With the file's M11_CONFIGS: which path each token count takes, and nt always covers it."""
    state = {"m11_configs": m11.M11_CONFIGS}
    routes = {}
    for tokens in range(1, 400):
        cfg = m11.m11_config_for(state, tokens)
        if cfg is not None:
            assert tokens <= m11.M11_MAX_TOKENS, (tokens, cfg)
            for key, inst in (("s1", (1, 2, cfg["s1"]["nt"], cfg["s1"]["d"], cfg["s1"]["ks"])),
                              ("s2", (0, cfg["s2"]["tt"], cfg["s2"]["nt"], cfg["s2"]["d"], cfg["s2"]["ks"]))):
                assert inst in m11.M11_INSTANCES, (tokens, key, inst)
        routes[tokens] = "v1" if cfg is None else "M11"
    m11_counts = sorted(t for t, r in routes.items() if r != "v1")
    print(f"ok: dispatch with M11_CONFIGS: M11 for {m11_counts[0]}-{m11_counts[-1]} tokens, every config a "
          f"compiled instance; v1 otherwise" if m11_counts else "ok: dispatch: M11_CONFIGS empty, v1 everywhere")
    return routes


if __name__ == "__main__":
    check_m2.install_stubs()
    sys.modules["torch.utils.cpp_extension"] = type(sys)("torch.utils.cpp_extension")
    sys.modules["torch.utils.cpp_extension"].load_inline = None
    m11 = check_submission_m11()
    MAXP = m11.M11_MAX_TOKENS
    check_dispatch(m11)
    check_m12_sequence(m11)
    check_shuffle(m11)
    check_pipeline(m11)

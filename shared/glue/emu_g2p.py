"""CPU emulation of shared/glue/g2p.py's fused M11 GEMM2 + post-norm (no GPU here).

1. Protocol: the kernel's task split (task = blockIdx * GROUPS + grp + i * grid * GROUPS over active experts x
   per_expert tiles), with groups interleaved in random orders, the y stores of a task landing before its counter
   update, and the last-arriver epilogue ported line for line (expert counter, its reset, the per-token counters
   with each token's local-route count, their reset, the finished-token list). Checks over 3 consecutive calls on
   the same counters (self-reset) with different routings, incl. tokens without local routes: every token with a
   local route is finished exactly once, only after every tile of each of its local experts was stored; tokens
   without one get the zero row (CTA b < tokens); all counters are zero after each call.
2. Numerics: g2p_token's math (two passes, fp32 products of bf16 y and bf16 nw, per-thread partial sums of squares
   over its uint4 columns, warp then group reduction, rsqrt, weighted sum in k order, bf16 RNE) vs the two-kernel path
   (M13's post_norm_kernel math, 512 threads) at K = 4096 on random bf16 y: max relative error and bf16-ulp count.
3. Pins: the CUDA lines these ports mirror must be present in g2p.py's source unchanged.

Run: .venv/bin/python shared/glue/emu_g2p.py
"""

import random
import tempfile
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ns = {"__name__": "g2p_emu", "torch": torch, "load_inline": None, "tempfile": tempfile, "M11_MAX_TOKENS": 320,
      "POST_NORM_EPS": 1e-8}
exec(compile((HERE / "g2p.py").read_text(), "g2p.py", "exec"), ns)
CUDA = ns["g2p_source"]()
LOCAL, TOP_K, K = 32, 8, 4096

# ---------------------------------------------------------------------------------------------- 3. pins
PINS = [
    "  for (int zt = blockIdx.x; zt < n_pairs / TOP_K; zt += gridDim.x) {  // G2P: tokens without a local route: zero rows",
    "      any |= (ze >= 0 && ze < LOCAL) ? 1 : 0;",
    "    if (!any)",
    "            __threadfence();  // acquire: every other task's y stores of expert e",
    "            __threadfence();  // acquire: the other experts' y rows of token tk",
    "        reinterpret_cast<uint4*>(pout + static_cast<size_t>(zt) * K)[i] = make_uint4(0u, 0u, 0u, 0u);",
    "      const int tig = kpart * 32 + lane;  // thread in group",
    "        __threadfence();  // release: this lane's y stores before the counter",
    "          fin = atomicAdd(cnt + e, 1) == per_expert - 1;",
    "            atomicExch(cnt + e, 0);  // self-reset: no other task of this call touches cnt[e] again",
    "        for (int i = tig; i < total; i += KS * 32) {",
    "          const int tk = s_list[e][i] / TOP_K;",
    "            nl += (e2 >= 0 && e2 < LOCAL) ? 1 : 0;",
    "          __threadfence();  // release by this thread: expert e's y rows (acquired by the group above) before the token counter",
    "          if (atomicAdd(cnt + LOCAL + tk, 1) == nl - 1) {",
    "            atomicExch(cnt + LOCAL + tk, 0);",
    "            s_done[grp][atomicAdd(&s_ndone[grp], 1)] = tk;",
    "          g2p_token<KS>(s_done[grp][d], tig, grp, ids, local_start, a_y(out), nw, tw, pout, eps, s_pss, s_rs);",
    "  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {",
    "  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;",
    "    G2P_TASK_DONE" if False else "    {  // G2P: this task's y tiles are written (warp kpart 0 of the group stored them)",
    # g2p_token
    "  constexpr int PER = (K / 8) / NT_;  // uint4 columns per thread",
    "    const int col = tig + c * NT_;",
    "      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);",
    "        const float v0 = __uint_as_float(u << 16) * __uint_as_float(wu << 16);",
    "        const float v1 = __uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u);",
    "        ss[k] += v0 * v0 + v1 * v1;",
    "    for (int i = 0; i < KS; ++i) s += s_pss[grp][i][tig];",
    "    s_rs[grp][tig] = rsqrtf(s / static_cast<float>(K) + eps);",
    "  for (int k = 0; k < TOP_K; ++k) f[k] = loc[k] ? tw[tk * TOP_K + k] * s_rs[grp][k] : 0.0f;",
    "        o[2 * j] += f[k] * (__uint_as_float(u << 16) * __uint_as_float(wu << 16));",
    "        o[2 * j + 1] += f[k] * (__uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u));",
    "    reinterpret_cast<uint4*>(pout + static_cast<size_t>(tk) * K)[col] = make_uint4(pk[0], pk[1], pk[2], pk[3]);",
]
missing = [p for p in PINS if CUDA.count(p + "\n") < 1]
assert not missing, f"g2p.py's CUDA changed; update the ports: {missing[:3]}"
moe_ns = {"__name__": "g2p_moe_emu", "torch": torch, "M11_MAX_TOKENS": 320, "POST_NORM_EPS": 1e-8}
exec(compile((HERE / "g2p_moe.py").read_text(), "g2p_moe.py", "exec"), moe_ns)
MOE = moe_ns["g2p_moe_source"]()
missing = [p for p in PINS if MOE.count(p + "\n") < 1]
assert not missing, f"g2p_moe.py's CUDA changed; update the ports: {missing[:3]}"
assert "namespace m11f {" in MOE and "namespace m11 {" not in MOE and "#include" not in MOE.split("namespace m11f {")[0]
print(f"ok 3: {len(PINS)} CUDA lines mirrored by the ports are unchanged (decode_layer and MoE variants)")


# ------------------------------------------------------------------------------------------ 1. protocol
class Violation(Exception):
    pass


def routing(tokens, rng, start):
    """ids [tokens*8] global ids: distinct experts per token; some tokens without local routes."""
    ids = []
    for t in range(tokens):
        nloc = rng.choice([0, 0, 1, 2, 3, 5, 8]) if tokens > 1 else rng.choice([1, 2, 3])
        loc = rng.sample(range(LOCAL), nloc)
        rest = rng.sample([e for e in range(256) if not (start <= e < start + LOCAL)], TOP_K - nloc)
        row = [start + e for e in loc] + rest
        rng.shuffle(row)
        ids += row
    return ids


def one_call(ids, cnt, tokens, TT, KS, grid, start, rng, rt_total=16):
    groups_per_cta = 4 // KS
    per_expert = rt_total // TT
    lists = {}
    for p, gid in enumerate(ids):  # the CTA scan (s_cnt / s_list / s_active, ascending expert order)
        e = gid - start
        if 0 <= e < LOCAL:
            lists.setdefault(e, []).append(p)
    active = sorted(lists)
    n_tasks = len(active) * per_expert
    stored = set()  # (e, tile) whose y stores landed
    finished = {}
    out_zero = set()
    # CTA b: tokens b, b + grid, ...: zero row for a token without local routes
    for b in range(grid):
        for zt in range(b, tokens, grid):
            if not any(0 <= ids[zt * TOP_K + k] - start < LOCAL for k in range(TOP_K)):
                out_zero.add(zt)
    workers = []
    for b in range(grid):
        for grp in range(groups_per_cta):
            mine = list(range(b * groups_per_cta + grp, n_tasks, grid * groups_per_cta))
            if mine:
                workers.append(mine)

    def run_task(task):
        e = active[task // per_expert]
        tile = task % per_expert
        stored.add((e, tile))
        yield  # other workers may run between the stores and the counter update (only after the fence)
        cnt[e] += 1
        fin = cnt[e] - 1 == per_expert - 1
        if fin:
            if any((e, t) not in stored for t in range(per_expert)):
                raise Violation(f"expert {e} finished before all its tiles were stored")
            cnt[e] = 0
            done = []
            for p in lists[e]:
                tk = p // TOP_K
                nl = sum(1 for k in range(TOP_K) if 0 <= ids[tk * TOP_K + k] - start < LOCAL)
                old = cnt[LOCAL + tk]
                cnt[LOCAL + tk] += 1
                if old == nl - 1:
                    cnt[LOCAL + tk] = 0
                    done.append(tk)
            for tk in done:
                for k in range(TOP_K):
                    e2 = ids[tk * TOP_K + k] - start
                    if 0 <= e2 < LOCAL and any((e2, t) not in stored for t in range(per_expert)):
                        raise Violation(f"token {tk} finished before expert {e2}'s tiles were stored")
                if tk in finished:
                    raise Violation(f"token {tk} finished twice")
                finished[tk] = True
        yield

    gens = [iter(sum(([t] for t in w), [])) for w in workers]
    pending = [None] * len(workers)
    alive = list(range(len(workers)))
    while alive:
        i = rng.choice(alive)
        if pending[i] is None:
            nxt = next(gens[i], None)
            if nxt is None:
                alive.remove(i)
                continue
            pending[i] = run_task(nxt)
        try:
            next(pending[i])
        except StopIteration:
            pending[i] = None
    return finished, out_zero


def protocol_checks(mutate=None):
    rng = random.Random(7)
    runs = 0
    for TT, KS, c in ((1, 4, 4), (2, 4, 8), (2, 2, 3), (2, 1, 2)):
        for tokens in (1, 2, 5, 8, 16, 32):
            start = 32 * rng.randrange(8)
            cnt = [0] * (LOCAL + 320)
            for call in range(3):  # the same counters, three calls: self-reset
                ids = routing(tokens, rng, start)
                finished, zero = one_call(ids, cnt, tokens, TT, KS, c * 3, start, rng)
                for tk in range(tokens):
                    has = any(0 <= ids[tk * TOP_K + k] - start < LOCAL for k in range(TOP_K))
                    if has != (tk in finished):
                        raise Violation(f"token {tk} (local routes: {has}) finished: {tk in finished}")
                    if (not has) != (tk in zero):
                        raise Violation(f"token {tk} zero row: {tk in zero}, local routes: {has}")
                if any(cnt):
                    raise Violation(f"counters not zero after call {call}: {[i for i, v in enumerate(cnt) if v]}")
                runs += 1
    return runs


runs = protocol_checks()
print(f"ok 1: last-arriver protocol, {runs} calls (3 per counter set; TT 1/2, KS 4/2/1, 1-32 tokens, random task "
      "orders): each token finished once after all its experts' tiles, zero rows for tokens without local routes, "
      "counters back to zero")


# ------------------------------------------------------------------------------------------ 2. numerics
def bf(x):
    return x.to(torch.bfloat16).float()


def ref_post_norm(y, nw, tw, loc, eps=1e-8):
    """M13 post_norm_kernel (512 threads x 8 values): v = y * nw, ss per 512-thread layout, rs, weighted sum."""
    v = y.float() * nw.float()  # [8, K]
    part = (v * v).view(TOP_K, 512, 8).sum(-1)  # per thread
    ss = part.view(TOP_K, 16, 32).sum(-1).sum(-1)  # warp, then 16 warps
    rs = torch.rsqrt(ss / K + eps)
    o = torch.zeros(K)
    for k in range(TOP_K):
        if loc[k]:
            o += (tw[k] * rs[k]) * v[k]
    return o.to(torch.bfloat16)


def g2p_post_norm(y, nw, tw, loc, KS=4, eps=1e-8):
    """g2p_token: thread tig owns uint4 columns tig + c * KS*32 (8 values each), two passes."""
    nt = KS * 32
    per = (K // 8) // nt
    v = y.float() * nw.float()
    vv = v.view(TOP_K, per, nt, 8)  # [k, c, tig, 8]: column (tig + c * nt) * 8 + j
    th = (vv * vv).sum(-1).sum(1)  # per thread [k, tig], c then j order
    th = torch.where(torch.tensor(loc).view(-1, 1), th, torch.zeros_like(th))
    ss = th.view(TOP_K, KS, 32).sum(-1).sum(-1)
    rs = torch.rsqrt(ss / K + eps)
    f = [tw[k] * rs[k] if loc[k] else 0.0 for k in range(TOP_K)]
    o = torch.zeros(K)
    for k in range(TOP_K):
        if loc[k]:
            o += f[k] * v[k]
    return o.to(torch.bfloat16)


gen = torch.Generator().manual_seed(3)
worst, ulps = 0.0, 0
for trial in range(200):
    y = (torch.randn(TOP_K, K, generator=gen) * (0.01 + trial / 50)).to(torch.bfloat16)
    nw = (1 + 0.1 * torch.randn(K, generator=gen)).to(torch.bfloat16)
    tw = torch.softmax(torch.randn(TOP_K, generator=gen), 0).tolist()
    loc = [bool(b) for b in (torch.rand(TOP_K, generator=gen) < 0.4).tolist()]
    if not any(loc):
        loc[trial % TOP_K] = True
    a = ref_post_norm(y, nw, tw, loc).float()
    for KS in (4, 2, 1):
        b = g2p_post_norm(y, nw, tw, loc, KS).float()
        worst = max(worst, float((a - b).norm() / a.norm()))
        ulps += int((a != b).sum())
assert worst < 1e-3, worst
print(f"ok 2: g2p_token math vs the two-kernel post-norm at K = 4096 (200 random cases x KS 4/2/1): max rel err "
      f"{worst:.1e}, {ulps} bf16 outputs differ by rounding of the sum of squares' order (of {200 * 3 * K})")

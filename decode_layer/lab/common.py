"""Lab helpers shared by decode_layer check-run labs (appended after the submission by
decode_layer/local/make_lab.py). Everything here runs inside prepare() on synthetic inputs; none
of it touches the timed calls."""

import math
import statistics
import time

LAB_FLUSH_BYTES = 512 << 20  # larger than the H200's 60 MB L2, like the judge's cold-L2 read


def log(msg):
    print(f"[lab] {msg}", flush=True)


def _error_text(exc, keep=400):
    """The end of an exception's message, where Triton puts the actual reason."""
    text = " ".join(str(exc).split())
    return text if len(text) <= keep else "... " + text[-keep:]


class Lab:
    def __init__(self, budget_s):
        self.t0 = time.monotonic()
        self.budget_s = budget_s
        self.flush = None

    def elapsed(self):
        return time.monotonic() - self.t0

    def left(self):
        return self.budget_s - self.elapsed()

    def section(self, name, fn, *args, until=None, min_left=5.0):
        """Run fn(lab, *args, deadline) where deadline is lab-elapsed seconds; failures only print."""
        if self.left() < min_left:
            log(f"{name}: skipped, {self.left():.0f}s of budget left")
            return
        deadline = self.budget_s if until is None else min(until, self.budget_s)
        t = time.monotonic()
        try:
            fn(self, *args, deadline)
        except Exception as exc:  # a lab failure must never fail the run
            log(f"{name}: FAILED {type(exc).__name__}: {_error_text(exc)}")
        torch.cuda.synchronize()
        log(f"{name}: {time.monotonic() - t:.1f}s")

    def cold_l2(self):
        if self.flush is None:
            self.flush = torch.empty(LAB_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        return self.flush.sum()

    def time_graph(self, fn, reps=15, cold=True):
        """Median microseconds of a CUDA graph replay of fn, each after an L2 flush if cold.
        Includes the replay's fixed cost (~4 us, which the judge subtracts)."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            if cold:
                self.cold_l2()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)

    def empty_graph_us(self):
        """A replay of a graph holding one trivial kernel: the fixed cost to subtract."""
        x = torch.zeros(1, device="cuda")
        return self.time_graph(lambda: x.add_(1), cold=True)


# The 12 scored shapes: (name, batch, q_len, layer, weight). All ctx 80000, page 1, dist soft,
# spread agentic above batch 1.
SCORED = (
    ("dec1 sw", 1, 1, "sliding", 46), ("dec1 full", 1, 1, "full", 16),
    ("dec16 sw", 16, 1, "sliding", 46), ("dec16 full", 16, 1, "full", 16),
    ("dec32 sw", 32, 1, "sliding", 46), ("dec32 full", 32, 1, "full", 16),
    ("ver1 sw", 1, 8, "sliding", 46), ("ver1 full", 1, 8, "full", 16),
    ("ver16 sw", 16, 8, "sliding", 46), ("ver16 full", 16, 8, "full", 16),
    ("ver32 sw", 32, 8, "sliding", 46), ("ver32 full", 32, 8, "full", 16),
)
LAB_CTX = 80000
LAB_SPREAD = 0.075  # agentic: lengths uniform in ctx * (1 +/- 0.075)
LAB_STREAM_SEED = 5354  # the judge's stream basis (a fixed mean direction the router aligns with)
LAB_STREAM_MEAN, LAB_STREAM_SHARED, LAB_STREAM_RANK, LAB_DRAFT_SHARE = 0.7, 0.9, 4, 0.6
LAB_CACHE_CHUNK = 1 << 20


def lab_stream_basis(device):
    gen = torch.Generator().manual_seed(LAB_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen)
    shared = torch.randn((LAB_STREAM_RANK, HIDDEN), generator=gen)
    return mean.to(device), shared.to(device)


def lab_input(batch, q_len, layer, seed, device="cuda", ctx=LAB_CTX, local_start=0):
    """Inputs shaped like the judge's for one scored case (page 1, kv scale 1). The cache holds
    N(0, 1) keys and values in FP8: their values don't change timing."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    if batch == 1:
        seq_lens = torch.full((1,), ctx, dtype=torch.int64, device=device)
        longest = ctx
    else:
        u = torch.rand((batch,), generator=gen, device=device)
        longest = math.ceil(ctx * (1 + LAB_SPREAD))
        seq_lens = (ctx * (1 + LAB_SPREAD * (2 * u - 1))).long().clamp(1, longest)
    max_pages = longest + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    pages_per = seq_lens + q_len
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b, (n, o) in enumerate(zip(pages_per.tolist(), offsets.tolist())):
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    cache_loc = torch.gather(page_table.long(), 1, positions.view(batch, q_len)).reshape(-1)
    k_cache = torch.empty((total, 1, 2, 64), dtype=FP8, device=device)
    v_cache = torch.empty((total, 1, 2, 64), dtype=FP8, device=device)
    for dst in (k_cache, v_cache):
        flat = dst.view(-1, 128)
        for at in range(0, total, LAB_CACHE_CHUNK):
            n = min(LAB_CACHE_CHUNK, total - at)
            flat[at:at + n] = torch.randn((n, 128), generator=gen, device=device).to(FP8)
    mean, shared = lab_stream_basis(device)
    coef = torch.randn((tokens, LAB_STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((batch, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = LAB_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (LAB_STREAM_MEAN * mean + math.sqrt(LAB_STREAM_SHARED / LAB_STREAM_RANK) * coef @ shared)
              + (1 - LAB_STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(torch.bfloat16)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    sched = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        sched = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=16,
            num_heads_k=2, headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
            cu_seqlens_q=cu_seqlens_q, page_size=1, causal=True, num_splits=0)
    sliding = layer == "sliding"
    return {
        "stream": stream, "attn_input": attn_input, "k_cache": k_cache, "v_cache": v_cache,
        "positions": positions, "cache_loc": cache_loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": sched,
        "k_scale": torch.tensor(1.0, device=device), "v_scale": torch.tensor(1.0, device=device),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=device),
        "config": {
            "batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN, "num_q_heads": 16,
            "num_kv_heads": 2, "head_dim": 64, "page_size": 1, "window": 2048 if sliding else 0,
            "rope_theta": ROPE_THETA if sliding else 0.0, "qk_scale_factor": 5.4730064863875,
            "sm_scale": 0.125, "kv_scale": 1.0, "n_experts": N_EXPERTS, "top_k": TOP_K,
            "local_experts": LOCAL_EXPERTS,
        },
    }


def kv_bytes(data):
    """KV bytes attention must read: each request's attended tokens x 256 B (K and V, 2 heads)."""
    cfg = data["config"]
    lens = data["cache_seqlens"].long()
    if cfg["window"]:
        lens = lens.clamp(max=cfg["window"] + cfg["q_len"] - 1)
    return int(lens.sum()) * 256

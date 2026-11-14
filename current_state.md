# Current state: where the time goes, per track (Nov 4, ~12:00 UTC)

Source for every number: the judge's per-kernel profile saved next to each track's **best scored run**
(`<track>/profiles/<file>.<job>.txt`), measured on the judge's H200 under CUDA graphs. "SoL" (speed of
light) is the judge's own lower bound for the shape: the time to stream the bytes the shape must read
from HBM at full bandwidth. Weight bytes below are computed from the problem's `reference.py` shapes.

## The chip, and the one rule that explains almost every number below

```
             H200 (one GPU, sm_90a)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  HBM3e 141 GB  ── ~4.5–4.8 TB/s ──►  L2 60 MB  ──►  132 SMs           │
 │  (weights, KV cache,                    (shared)      each: 228 KB    │
 │   activations live here)                              smem/L1, 4 tensor│
 │                                                       cores, regs      │
 │  every kernel launch in a graph: ~0.8 µs + ramp-up/tail of its waves   │
 └──────────────────────────────────────────────────────────────────────┘
```

* At decode sizes (1–256 tokens) **the work is tiny and the bytes are huge**: a 67 MB BF16 weight
  matrix takes ≥14 µs just to read once, while its math at m=1 is nanoseconds. So time ≈
  bytes moved ÷ achieved bandwidth + fixed costs (launch, ramp-up, tail, reductions, glue kernels).
* "% of SoL" = SoL ÷ our time. 100% means we stream the bytes as fast as the chip allows; the gap is
  everything else (low occupancy at small batch, waiting for the first bytes, extra kernels, decode of
  4-bit weights, split-K reductions).
* Only prefill's 16K-token shapes are compute-bound (tensor cores busy); everything else is
  memory-bound.

---

## 1. muse_bf16_proj — board 1.318× (3.67 ms), job `33748bdb`, `proj/scored/submission_v6.py`

**What it computes:** `out = x @ W.T` for the 5 projections of one layer. All shapes weigh the same.

```
  HBM: weight W (BF16)                       x (m×8192, tiny)
  qkv_gate 2304×8192 = 37.7 MB  ┐                 │
  o        8192×1024 = 16.8 MB  │  stream once     │ broadcast to every CTA
  pooler   4096×8192 = 67.1 MB  ├──────────────►  SMs: dot products (FMA or tensor core)
  unpooler 8192×4096 = 67.1 MB  │  (≥ 7.9 / 3.5 /  │  ──► split-K partials ──► reduce ──► out (m×n)
           (only 8.4 MB at m≤8) │   14 / 14 / 1.8  │
  router   256×8192 FP32 8.4 MB ┘   µs at SoL)     │
```

**Our kernels and times.** How to read a cell: each value is our time in µs for that projection at
that token count. Each cell reads `our time / SoL (%)`. SoL is the fastest it could ever run, just from reading its
weights once at full memory speed. The `(%)` is SoL ÷ our time, i.e. how close we are to that limit (100% = can't go faster; 50% = we take
twice the minimum). Example: qkv_gate at m=1 takes 9.6 µs, its limit is 7.9 µs → 82%.

| m (tokens) | qkv_gate | o | pooler | unpooler | router | kernels used |
|---|---|---|---|---|---|---|
| 1 | 9.6 / 7.9 (82%) | 5.8 / 3.5 (60%) | 14.9 / 14.0 (94%) | 4.2 / 1.8 (42%) | 4.0 / 1.8 (44%) | own CUDA `f12v` GEMV, Triton `_fma` |
| 8 | 10.4 / 7.9 (76%) | 6.0 / 3.5 (59%) | 14.4 / 14.0 (97%) | 5.2 / 1.8 (34%) | 5.6 / 1.8 (32%) | own CUDA `f12m`, `p9` |
| 16 | 12.5 / 7.9 (63%) | 6.3 / 3.6 (56%) | 15.4 / 14.1 (91%) | 16.7 / 14.1 (84%) | 5.9 / 1.8 (31%) | `f12m`, Triton `_swap`, `p9` |
| 32 | 14.7 / 8.0 (55%) | 6.7 / 3.6 (54%) | 20.1 / 14.1 (71%) | 18.9 / 14.1 (75%) | 6.4 / 1.9 (29%) | **cuBLAS nvjet** (= SGLang), split-K + reduce |
| 128 | 16.4 / 8.4 (51%) | 7.5 / 4.0 (53%) | 21.4 / 14.6 (69%) | 20.0 / 14.6 (73%) | 10.2 / 2.2 (22%) | cuBLAS, DeepGEMM (= SGLang) |
| 256 | 19.9 / 9.8 (49%) | 9.6 / 4.5 (47%) | 30.8 / 17.4 (56%) | 24.4 / 17.4 (71%) | 13.2 / 2.7 (20%) | cuBLAS (= SGLang) |

**Where the time goes:**
* **Moving weights:** ~70–95% of each kernel at m ≤ 16 — we're close to SoL on the big matrices.
* **Fixed overhead on small matrices:** `o` (16.8 MB) and router/unpooler-slice (8.4 MB) take 4–6 µs
  for 1.8–3.5 µs of bytes: launch + ramp-up + tail + split-K reduction is ~2 µs per kernel.
* **m = 32, 128, 256 are still SGLang's own cuBLAS kernels** (speedup 1.00×): 9 of 30 shapes earn
  nothing. At m=32 cuBLAS reaches only 55% of SoL on qkv_gate (8.6 µs GEMM + 1.9 µs split-K
  reduce kernel). **This is the biggest lever left** (a custom tensor-core GEMM for m=32–256).

---

## 2. muse_attention_decode — board 1.459× (2.58 ms), job `f9b4a77b`, `attention/submission_v6.py`

**What it computes:** one decode step (or an 8-token verify block) of GQA attention over a paged FP8
KV cache: 16 query heads, 2 KV heads, head dim 64; sliding layer = last 2,048 tokens with RoPE, full
layer = whole ~80K context.

```
  q (BF16) ──► _prologue: QK-norm, RoPE, write new k/v to FP8 cache (1.3–2.5 µs)
                   │
  HBM: FP8 KV cache, 256 B per token per request (K+V, 2 heads × 64)
     full layer: 80K tokens × 256 B = 20.5 MB per request  ── 4.3 µs at SoL each
     sliding   :  2K tokens × 256 B = 0.5 MB per request    ── 0.1 µs at SoL each
                   │  paged gather (page = 1 token → scattered 128 B rows)
                   ▼
  attend kernel: split the context across many CTAs (flash-decoding), each does
     QK^T (FP8→tensor core), online softmax, ·V  → partial (max, sum, acc)
                   ▼
  _combine: merge the partials, gate, normalize → out (1.8–7 µs)
```

| load | sliding (ours / SoL) | full (ours / SoL) | kernels |
|---|---|---|---|
| decode 1 | **9.3** / 0.1 | **19.5** / 4.3 (22%) | own CUDA `ud3`; own `dec4` 12.1 + combine 3.5 + prologue 1.3 |
| decode 16 | 14.4 / 1.7 | 109 / 63 (58%) | Triton `_attend_p`; `dec4` 102.7 |
| decode 32 | 21.0 / 3.3 | 210 / 126 (60%) | Triton; `dec4` 200 |
| verify 1 | 13.9 / 0.1 | 31.5 / 4.3 | Triton; own `ver4` 24.5 |
| verify 16 | 22.1 / 1.9 | **205.7 / 63 (31%)** | Triton; **SGLang FA3 (1.00×)** 194 |
| verify 32 | 35.1 / 3.7 | **390 / 127 (33%)** | Triton; **SGLang FA3 (1.00×)** 373 |

**Where the time goes:**
* **Full-attention layers at batch 16–32 are ~90% of all our time** (109 + 210 + 206 + 390 µs).
  It's reading the FP8 cache: 330–655 MB per call. Decode reaches ~60% of SoL; **verify 16/32 still
  run SGLang's FA3 at ~32% of SoL** → the single biggest lever on this track (the session's L16 FP8
  verify kernel targets exactly these).
* **Sliding layers and batch 1 are fixed-cost bound**: 0.1–3.7 µs of bytes, 9–35 µs of time — three
  kernels (prologue, attend, combine) each with launch + ramp, plus the paged gather latency.

---

## 3. muse_moe_w4a16 — board 1.443× (9.32 ms), job `2e2f531a`, `moe/submission_m13t_e14.py`

**What it computes:** for each token, its top 8 of 256 experts (32 are local to this GPU):
`y = norm( silu(x·W1) * (x·W3) · W2 )`, weighted sum. Weights are NVFP4 (4-bit + FP8 scale per 16).

```
  HBM: NVFP4 expert weights, per expert: W13 8192×4096 + W2 4096×4096
       = 25.2 MB of 4-bit values + 1.6 MB of scales ≈ 26.7 MB per expert
       (bs 1 touches ~2 local experts, bs 256 touches all 32 ≈ 855 MB)
                │ stream, pre-shuffled into MMA-fragment order (lossless repack in prepare)
                ▼
  m11 gemm_kernel<…, gated=true>: x · W13 → decode 4-bit→BF16 in registers → tensor-core MMA
                → SiLU·mul (fused epilogue) → h (BF16, tokens × 4096)
                ▼
  m11 gemm_kernel<…, gated=false>: h · W2 → y
                ▼
  _m11_post_norm_kernel: per-expert norm + top-k weighted sum → out (3 µs)
```

| load (tokens) | ours | SoL | % SoL | split |
|---|---|---|---|---|
| decode 1 (1) | 37.6 | 11.4 | 30% | W13 25.7 + W2 16.6 + norm 3.0 |
| decode 16 | 129.6 | 52.5 | 41% | 53.7 + 29.7 + 3.5 |
| decode 32 | 195.4 | 70.8 | 36% | 133.1 + 68.2 + 3.6 |
| verify 1 (8) | 93.1 | 37.3 | 40% | 68.3 + 37.8 + 2.9 |
| verify 16 (128) | 331.2 | 101.3 | 31% | **SGLang Marlin** 345 + 6 small glue kernels ~20 |
| verify 32 (256) | 469.6 | 108.9 | 23% | **SGLang Marlin** 489 + glue ~25 |

**Where the time goes:**
* **Moving 4-bit weights + decoding them in registers**: we reach only 23–41% of SoL. At full
  bandwidth there are just ~3.7 thread-instructions per 4-bit weight (F45), so the decode, the
  barrier waits and the few warps in flight at small batch are the bottleneck, not HBM.
* **128 and 256 tokens still run SGLang's Marlin** (the same kernel we're compared against, plus a
  tuned config): 2 of 6 loads. The session's next kernel (E12 / RS-wgmma for 128/256) targets them.
* Glue (routing sort, align, act-and-mul, gather) is small: 3–25 µs.

---

## 4. muse_decode_layer — board 1.364× (16.58 ms), job `0dcf0c8e`, `decode_layer/submission_v7.py`

**What it computes:** one whole decoder layer for a decode step = everything above, chained.

```
 residual h ─► norm+residual ─► qkv_gate proj ─► attention (prologue, attend, combine)
  (BF16)       (SGLang, 4 µs)   (bf16_proj)      (attention_decode track)
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
  o proj ─► norm+residual ─► router proj ─► routing (top-8) ─► pooler proj (8192→4096)
 (bf16_proj) (SGLang, 4 µs)  (bf16_proj)    (SGLang, 2.5 µs)    (bf16_proj)
                                                                     │
     ┌───────────────────────────────────────────────────────────────┘
     ▼
  experts W13 ─► experts W2 ─► post-norm + weighted sum ─► unpooler proj (4096→8192) ─► h'
  (MoE track: m11 or SGLang Marlin)       (3–6 µs)          (bf16_proj)
  every arrow = a kernel boundary: activations go back to HBM/L2 and ~1–3 µs of launch/ramp
```

**Time split by part (µs, from the profile):**

| load | total (SGLang) | experts | attention | 5 projections | norms, routing, glue |
|---|---|---|---|---|---|
| decode 1 sliding | 103 (170) | 42 (41%) | 8 | 45 | 13 |
| decode 1 full | 118 (180) | 42 | 22 | 45 | 13 |
| decode 16 full | 298 (435) | 94 (31%) | 121 (41%) | 64 | 15 |
| decode 32 full | 412 (617) | 124 (30%) | 212 (51%) | 69 | 16 |
| verify 1 full | 181 (270) | 72 (40%) | 33 | 61 | 15 |
| verify 16 sliding | 424 (525) | **~290 (68%)** Marlin | 20 | 76 | ~30 |
| verify 32 full | 945 (1054) | **~440 (47%)** Marlin | **381 (40%)** FA3 | 134 | ~30 |

**Where the time goes:**
* **Experts (MoE) are the largest part everywhere except full attention at batch 16–32.** At
  verify 16/32 they still run SGLang's Marlin (1.0×) → the MoE track's 128/256-token kernel lands here
  directly.
* **Full attention at verify 16/32 still runs SGLang's FA3** → the attention track's verify kernel
  lands here directly.
* **Projections are 25–45% at small batch** (≥ 4 kernels, ~45 µs at m=1 for ~37 µs of SoL bytes).
* **Glue (norms, routing, combine, ~15–30 µs)** and kernel boundaries are where fusion pays
  (qkv epilogue → attention prologue, router + routing, post-norm + unpooler; PDL already overlaps
  some: "GPU busy 966 µs" vs the 945 µs critical path at verify 32).
* **Note:** the profile above is the scored v7 run; checks since then (DE8–DE11) sit at ~1.37×.

---

## 5. muse_prefill_layer — board 1.344× (288 ms), job `c8adfdf2`, `prefill_layer/v5_scored.py`

**What it computes:** the same layer as decode_layer, but for prompt processing: 753 new tokens on an
80K conversation (×1 or ×4), or a first turn of 16,384 tokens. Doesn't count toward the overall board.

```
  tokens (753 … 16,384) × 8192 BF16
     │
     ├─► dense projections (cuBLAS nvjet, tensor cores; compute-bound at 16K)
     ├─► attention (SGLang FA3, causal over new block + cached context)
     ├─► routing ─► gather rows per expert ─► experts:
     │       753 / 3,012 tokens : own CUDA R14 W4A16 (TMA + wgmma, 4-bit decode in registers)
     │       16K tokens         : _nvfp4_dequant to BF16 (805 µs) + CUTLASS grouped GEMM (4,137 µs)
     └─► combine (weighted sum) ─► norms + residual (SGLang, 400 µs at 16K)
```

**Time split by load (µs; load weight = how many times a prefill runs it):**

| load (weight) | ours (SGLang) | experts | dense proj | attention | norms + glue |
|---|---|---|---|---|---|
| turn 1 sliding (46) | 839 (1,059) | **473** R14 (56%) | ~225 | 33 | ~110 |
| turn 1 full (16) | 1,194 (1,406) | 454 | ~285 | **423** | ~100 |
| turn 4 sliding (46) | 2,425 (3,362) | **1,364** R14 (56%) | ~867 | 62 | ~200 |
| turn 4 full (16) | 3,993 (4,894) | 1,312 | ~861 | **1,655** (41%) | ~190 |
| first turn 32K sliding (**138**) | 11,122 (17,171) | **4,942** dequant+CUTLASS (44%) | **4,229** (38%) | 309 | ~1,640 |
| first turn 64K full (16) | 19,165 (24,294) | ~4,700 | ~4,230 | **8,103** (42%) | ~1,600 |

**Where the time goes:**
* **First-turn 16K shapes dominate the score** (weight 138 on the sliding one). There the experts
  still dequantize NVFP4 → BF16 into HBM (805 µs) and run a BF16 grouped GEMM (4.1 ms): a fused
  FP4-decode GEMM (R14 at 16K, which RE9 already started picking) removes the round trip.
* **Dense projections (cuBLAS) are 38% at 16K** — that's SGLang's own kernel, compute-bound; gains
  need better tiling or overlap (activations stay BF16 per rule 6, so no FP8 shortcut), not less memory
  traffic.
* **Turn 1 (753 tokens): the R14 MoE is ~470 µs vs a ~250 µs target** (track note: a 250 µs MoE takes
  turn 1 to ~1.6× and the geomean to ~1.5×).
* **Full attention on long contexts (64K) is SGLang FA3: 8.1 ms, 42%** of that shape.

---

## Summary: the biggest remaining lever per track

| track | % of time still on SGLang's own kernel | biggest single lever |
|---|---|---|
| bf16_proj | m = 32/128/256 (9 of 30 shapes, cuBLAS) | own GEMM for m = 32–256, kill split-K reduce kernels |
| attention | full verify 16/32 (FA3, ~32% of SoL) | FP8 verify kernel (L16, built, waiting for a check) |
| MoE | 128/256 tokens (Marlin) + small-batch at 23–41% SoL | E12 / RS-wgmma kernel for 128/256 tokens |
| decode_layer | Marlin at verify 16/32, FA3 at full verify 16/32 | inherits MoE E12 + attention L16, then fusion of the glue |
| prefill | 16K experts (dequant + CUTLASS), all dense proj, FA3 | fused FP4 GEMM at 16K; faster R14 at 753 tokens |

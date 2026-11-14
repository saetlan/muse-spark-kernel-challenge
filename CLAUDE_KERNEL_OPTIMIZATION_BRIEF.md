# Open Kernel Challenge — Optimization Handoff for Claude

**Date of snapshot:** 2026-11-04, approximately 16:37 UTC  
**Hardware:** NVIDIA H200, 132 SMs, ~60 MB L2, HBM3e (4.8 TB/s peak; 4.49 TB/s measured streaming)  
**Deadline from the supplied brief:** 2026-11-05 07:59 UTC  
**Goal:** Reach **#1 in at least one leaderboard track**, or maximize improvement toward it before the deadline.

> **Instructions to Claude:** Treat this as an engineering handoff, not a request for generic CUDA advice. Analyze the accompanying Python files and their actual final dispatch paths before proposing modifications. Challenge reported results below are **measured**; all new strategies are **hypotheses** until benchmarked. Favor a few high-upside, falsifiable experiments over broad parameter sweeps. Preserve full correctness and the competition rules. If you have code execution / GPU access, implement candidates in **new files** without overwriting known-good submissions, then benchmark them. If GPU access is unavailable, deliver a concrete, code-level patch plan and expected failure modes instead of claiming measured improvements.

## Files to provide alongside this brief

| File | Role |
|---|---|
| `standings_brief.md` | Leaderboard snapshot, track histories, bottlenecks, unanswered questions; **source of truth for recorded performance** |
| `bf16_proj_v6_1.318x.py` | Best scored BF16 projections (F12V/F12M + fallbacks) |
| `attention_decode_v8_1.484x.py` | Best scored attention decode (multiple Triton/CUDA/FA3 paths) |
| `moe_w4a16_E16_1.459x.py` | Best scored W4A16 MoE (Marlin/M11/M13/M18 etc.) |
| `decode_layer_v10_1.400x.py` | Best scored full decode layer (older component integration) |
| `prefill_layer_v5_1.344x.py` | Best scored prefill layer (R14 and BF16 expanded-weight grouped path) |

**Important:** These Python files contain appended versions, embedded CUDA source strings, and multiple redefinitions / dispatch layers. Do not assume the first function named `kernel_fn`, `prepare`, or `CHOICE` is what ultimately executes. Inspect the **last definitions, route table updates, and warm-up self-checks**.

## Rules / non-negotiable constraints

- Submission replaces SGLang's kernel for one of five problems; score is speedup over SGLang measured through CUDA graphs on the same inputs.
- NVFP4 expert weights must remain NVFP4 as the weight representation. **Lossless repacking in `prepare()` is allowed**; do not silently use an illegal permanent BF16/FP8 expanded representation.
- Keep BF16 activations where the baseline keeps BF16; uphold router and reference arithmetic/accuracy requirements.
- Never cache **input-derived** results across calls; do not move runtime computation outside the timed call.
- Correctly process **every** required KV position and expert route; do not exploit test seeds or shortcuts.
- `prepare()` can build legal weight-only representations and precompile candidates; verify its time budget and graph capture behavior.
- The brief reports **40 unscored checks / 20 scored runs per track per UTC day** (remaining balances were as of the snapshot; recheck current limits).
- Different shapes and route distributions can produce different winners; keep safe fallbacks.

## Scoreboard at snapshot

| Track | Ours | #1 | Factor needed beyond current | Our rank | Main bottleneck |
|---|---:|---:|---:|---:|---|
| `bf16_proj` | 1.3179× | 1.7111× | **1.30×** | 622 / 1713 | M=32/128/256 largely baseline/cuBLAS |
| `attention_decode` | 1.4836× | 3.3389× | **2.25×** | 958 / 1345 | Full-attention verify 16/32, FP8 KV pipeline |
| `moe_w4a16` | 1.4587× | 2.7817× | **1.91×** | 1000 / 1400 | Low-utilization NVFP4 decode / large-batch Marlin |
| `decode_layer` | 1.4005× | 2.5762× | **1.84×** | 899 / 1271 | Verify experts and attention, kernel boundaries |
| `prefill_layer` | 1.3442× | 2.1471× | **1.60×** | 623 / 876 | 16K expert dequantization, projections, long attention |

These are scores, not kernel microbenchmark speedups. We must **inspect the judge's aggregation formula** before ranking per-case improvements. At the recorded snapshot, run-to-run noise was around ±1% and local checks can differ from scored runs.

---

# PRIORITY 1 — Win the `bf16_proj` track (comparatively achievable)

### Known facts

`bf16_proj_v6` uses lossless ~12-bit BF16 repacking for some streaming kernels, specialized GEMV for M=1, and MMA-based kernels for M=8/16; many M=32/128/256 choices remain SGLang-like. Nine of 30 shapes are noted as not improving appreciably. For M=256, the brief estimates a tensor-core floor ~50 µs and HBM floor ~52 µs for the relevant large case. Gains must therefore come from better scheduling / occupancy / data reuse, **not assuming arbitrary 2× room**.

### First experiment (recommended first action)

Implement a **native BF16 small-M GEMM** for M=32 that avoids a separate split-K reduction:

- Start with `BM=16`, `BN=64` output tiling; complete K accumulation inside a CTA; FP32 accumulators and correct BF16 output.
- Compute launch grids **separately for each projection's output width**; e.g., N=4096 gives `ceil(32/16)*ceil(4096/64)=128` CTAs, but N=2304 yields only 72 CTAs. Do not claim one tile covers all shapes equally well.
- Compare against incumbent cuBLAS and existing `swap` / `gemm` / `f12m` paths under CUDA graph replay.
- Explore `BM=16/32`, `BN=32/64/128`, warp count, stages, M/N tile ordering; only explore split-K or Stream-K when the output grid is underoccupied.
- For M=128/256, try regular WGMMA with an efficient producer/consumer schedule **without** BF16 compression first. Compression helps bandwidth-limited GEMV but may increase issue pressure for compute-heavy GEMM.
- Treat the **router** (`N=256`, FP32 weights) as its own class: benchmark merging BF16-high + BF16-low accumulation into **one** kernel where baseline uses two products; verify numerical tolerance and exact routing consequences.

**Success criterion:** consistent, correct, ≥3–5% gain on several currently weak scored cases with an improved *whole-track* score. Prefer a modest win on multiple shapes over one narrow spectacular microbenchmark.

**Potential failure modes:** duplicated weight loads when reducing BM, register spilling, insufficient CTAs for narrow N, atomics/reduction overhead, numerical mismatch for router, memory floor already near peak.

# PRIORITY 2 — Attention verification for batch 16/32

### Known facts

Per GPU: `num_q_heads=16`, `num_kv_heads=2`, head dim 64, GQA ratio 8, verify `q_len=8`, FP8 E4M3 KV cache, scored page size 1 (other tests can use 64), sliding window 2048 or full context, sometimes ~80K cached tokens. Current v8 has different routes by shape, including `ver5` at full verify batch 1 and `fa3t` at full verify batch 16; batch 32 still needs a genuinely winning specialized full-verify path. The `fa3t` update appears at the end of `attention_decode_v8_1.484x.py`; inspect final dispatch.

### Key architectural hypothesis

A verify block provides **8 query tokens × 8 query heads per KV head = 64 query rows** sharing the same KV stream. Consider attending these 64 rows together so each fetched K/V tile is reused across many query rows.

- Test Q tile M=64 with K/V tile N=64 or 128; compare to smaller query tiles.
- Keep stable online softmax in FP32, exact masking (including causal positions within the verification block), head-to-KV mapping, RoPE, cache scales, and output norm/gate.
- Determine whether page table gathering, FP8 layout transformations, softmax consumer work, or global loads are limiting.
- Compare `ver4` / `ver5` / `ver5t` and FA3 on identical graph and cache conditions; `ver5t` attempts producer-side V re-layout, so check whether it actually addresses the bottleneck.
- Test a producer/consumer FP8 pipeline and **measure real overlap**, occupancy, and register pressure. Do not assume asynchronous instructions imply latency hiding.
- Change scheduling only with a specific hypothesis. Test whole-wave/static splits against length-aware/persistent scheduling; consider schedule overhead and split-merge traffic.

### Useful lower bound (illustrative, NOT per-case measured)

At 80K cached tokens, two KV heads × 64 dims × (K+V) × 1 byte ≈ **20.48 MB/request**. At measured 4.49 TB/s bandwidth, an ideal once-only KV stream costs ~**4.6 µs/request**, ~**73 µs at B=16**, ~**146 µs at B=32**. Actual time is higher because of page lookups, softmax, MMA, output, traffic duplication, sequence variation, etc. Use this to sanity-check proposed speedups.

**Risks:** FP8 query rounding differs from BF16-Triton self-check; a loose internal self-check threshold can mask bugs. Validate against the official oracle and long-context, short-context, sliding, full, page-size-64, non-unit KV-scale, and causal boundary cases. Do not skip required keys.

# PRIORITY 3 — MoE W4A16: redesign route/weight reuse

### Known facts

`moe_w4a16_E16` already tests compressed fragment-ordered weights, register decoding, MMA/WGMMA variants, PDL, and alternatives. M18 (shared-memory decode + SS-WGMMA) did not beat the incumbent for large batches; reported batch-256 GEMM1 ~312–319 µs, GEMM2 ~165–170 µs, M18 issue utilization ~22%. RS-WGMMA ran into generated synchronization / operand-arrival issues. Large batch 128/256 can fall back to Marlin. Do **not** spend remaining time on more tile tuning of a proven weak M18 path without new profiling evidence.

### Critical geometry calculation

With top-k=8, 256 experts global, and 32 local experts (EP8), under roughly uniform random routing:

`expected local assignments = batch * 8 * (32/256) = batch`

`expected assignments per local expert = batch / 32`

At batch 256, that means only ~8 rows/expert **on average**; at batch 128, ~4. This makes naive 64-row WGMMA tiles heavily padded. Real routing distributions may be skewed and should be measured.

### Experiment A: Two algorithms, not one universal kernel

- **Small batches (1–32):** route-oriented / warp-level MMA, minimal setup, reduced decompression instruction count, direct/PDL tail where proven.
- **Batches 64–256:** grouped work that reuses decoded weight fragments across multiple token rows assigned to the same expert. Compare a compact route schedule generated once per call vs. M11-style in-CTA route discovery. A persistent expert/output-tile scheduler is plausible, **but only if real per-expert occupancy supports it**.
- Profile *useful* MMA rows / padded rows, decoded-fragment reuse count, issue utilization, stalls, registers/spills, global BW, and repeated route-discovery work.

### Experiment B: Isolate the bottleneck

1. Time *decode-only* weight-fragment preparation (with output consumed enough to prevent optimization away).
2. Time GEMM using diagnostic already-unpacked scratch BF16 weights to estimate math/scheduling ceiling. **Do not submit permanently expanded BF16 weights as a substitute for the required NVFP4 path.**
3. Benchmark register MMA vs. WGMMA on the same routing with **equal useful arithmetic**, including padding costs.
4. If decode dominates: decode each tile once and increase reuse. If scheduling dominates: compact per-expert work and eliminate redundant route scanning. If issue/sync dominates: examine actual SASS and ptxas behavior before rewriting.

**Risks:** preprocessing a route table may cost more than it saves at low batch; WGMMA granularity wastes rows; reusing fragments raises register pressure; persistent work queues cause imbalance/atomics; prefetch may regress (prior tuning found some prefetch variants slower).

# PRIORITY 4 — Prefill: avoid full per-call weight expansion

### Known facts

`prefill_layer_v5` uses R14 Hopper W4A16 grouped GEMM on select cases, decoding compressed NVFP4 inside the kernel, with producer/consumer pipeline, shared activation layout and fused SwiGLU. For some large cases (including 16K tokens) the v2 path materializes touched experts as BF16 scratch, then uses `torch._grouped_mm`.

The local expert matrices have W13 of 8192×4096 and W2 of 4096×4096. Full expansion for 32 local experts is about **3 GiB in BF16** across W13+W2. For an illustrative full set, a 3 GiB write + 3 GiB read implies ~1.4 ms *ideal* HBM transfer floor at 4.49 TB/s. Actual touched-expert subset/traffic must be measured; grouped BF16 GEMM may compensate for its extra traffic by faster compute.

### Experiment

Compare three versions at large prefill token counts:

1. Existing expansion + grouped BF16 GEMM (`v2`).
2. R14-like on-the-fly NVFP4 decode in the GEMM with larger M tiles / persistent grouped scheduling.
3. **Hybrid:** stage and decode only a weight tile, then reuse it across many token rows and consumer warpgroups without writing all expanded weights to HBM.

Measure whole-layer contribution; remember dense projections and long-context FA3 are also significant. Do not assume expert speedup translates 1:1 to prefill speedup.

# PRIORITY 5 — `decode_layer`: integrate and fuse

### Known facts

`decode_layer_v10` integrates attention v6 (not standalone v8's complete routing), BF16 projection v6, and an M11/E3c-derived expert path. The final `V10_PDL` choices are empty according to the snapshot; check actual runtime route setup. Verify 16/32 remain weak, largely experts/attention-bound.

### Experiments, in this order

1. Update decoder attention paths from standalone v8 where correctness and workload semantics match; benchmark **entire** graph, not only attention.
2. Test newer MoE variants / PDL selectively, with full-layer timing and fallback.
3. Fuse **small-batch unpooler tail + post-FFN boundary**: baseline may create an [M,8192] tensor by zero-padding a [M,1024] unpooler slice before norm/residual; represent zero components implicitly until the next required write. Preserve 8192-element normalization and residual semantics.
4. Explore one combined routing kernel that produces final per-expert local schedule directly from router logits and top-k selection, reducing intermediate buffers and launches. Preserve bias/tie handling and normalization semantics.

Be wary of changes that win standalone but lose in the full decoder because of L2 residency, streams, scratch buffers, or graph scheduling.

---

## Process for selecting experiments

1. **Determine score formula** from competition harness. Is it a geometric mean of per-case ratios, aggregate latency ratio, or something else? Calculate exact predicted score deltas from actual per-case latencies.
2. **Establish incumbents**: benchmark original submissions under judge-like CUDA graph replay; record per-case input geometry, SGLang µs, our µs, kernel launch count, correctness, confidence/noise.
3. **Find the critical path** using per-kernel timing and profiler metrics. A total kernel share is not automatically a critical-path share when streams overlap.
4. **Rank experiments** using `(expected track-score gain × probability of success) / engineering time`; maintain a safe fallback for every unproven shape.
5. **Change one variable** per experiment: tile/layout, pipeline, or schedule. Avoid changing all three at once.
6. **Use diagnostic microbenchmarks first**, then CUDA graph benchmark under controlled cache conditions, then unscored official check, then scored run. Reserve official quota for credible gains.
7. Prefer reproducible improvements **≥3–5%** for promoting a new complicated path; ±1% noise and tuner-to-judge regressions already occurred. Smaller gains can accumulate if repeatedly confirmed.
8. Keep output of existing winning files intact, ensure the first graph capture never incurs compile/load, and obey the `prepare()` budget.

### Suggested measurements table

| Track/case | Shape / route class | SGLang µs | Incumbent µs | Candidate µs | Rel. speedup | Predicted score change | Correct? | Dominant stall / bandwidth |
|---|---|---:|---:|---:|---:|---:|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Suggested priority if only one #1 finish matters

**Start with BF16 M=32/128/256.** The gap to #1 is 1.30× relative to current score and several scored cases still behave like baseline. Try a simple native-BF16 custom GEMM with better output tiling *before* building sophisticated compression/cluster kernels. Then extend the winning strategy to related shapes and re-score the track.

**If the first BF16 experiments demonstrate no tractable gain**, pivot promptly to full-attention verify (largest architectural upside) or R14 large prefill (strong data-movement hypothesis). Avoid consuming the window on low-margin MoE M18 tuning.

## What I need back from Claude

Please return, in order:

1. **Code audit:** Identify the actual dispatch / bottlenecks in the final submitted files and any mistakes in the hypotheses above. Distinguish verified facts from inference.
2. **Ranked top 3 experiments** with (a) exact functions / kernels to alter, (b) new algorithm or schedule, (c) why current code is suboptimal, (d) rough attainable lower bound, (e) estimated impact on the official scoring formula, and (f) a precise benchmark plan.
3. **One executable candidate first** (if GPU/tooling available): prioritize the BF16 M=32 small-M GEMM outlined above, implemented in a new version file with correctness tests, fallback routing, and fair timing. If a stronger evidence-based opportunity exists, explain why and implement that instead.
4. **Ablation table** comparing incumbent vs candidate per relevant shape; list the exact candidate promoted to the scoreboard and why.
5. **Time-aware next steps:** explicitly state what to stop working on, what is feasible before 2026-11-05 07:59 UTC, and what may need more time.

### Please avoid

- Recommending techniques already present in the attached code without first identifying what is genuinely different.
- Suggesting native FP4 tensor-core MMA on H200 (Hopper has no native NVFP4 MMA path).
- Equating theoretical FLOP throughput with useful throughput for MoE experts averaging ~4–8 rows.
- Claiming a measured speedup without benchmarking.
- Tuning a broad grid of launch parameters before establishing a specific hardware bottleneck.
- Discarding difficult cases, caching results derived from timed inputs, changing required precision, or weakening validation to get a passing submission.

**Success means a correct, reproducible, competition-legal score improvement—not a fast-looking synthetic microbenchmark.**

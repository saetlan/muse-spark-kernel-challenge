# Open Kernel Challenge: standings and open questions

Snapshot of the okc leaderboards, Nov 4 2026 about 16:37 UTC. Team: saetlan. The kernel track closes Nov 5 07:59 UTC.

Context: five GPU-kernel problems on an H200 (132 SMs, 60 MB L2, HBM3e 4.8 TB/s peak, 4.49 TB/s measured streaming). A submission replaces SGLang's kernel for one problem; the score is the speedup over SGLang, both run as CUDA graphs on the same inputs. Rules: weights stay NVFP4 (lossless repack in prepare() is fine), BF16 activations where SGLang keeps BF16, nothing cached from the inputs, no work outside the timed call. Each problem allows 40 unscored checks and 20 scored runs per UTC day.

## Standings

| Track | Our rank | Our speedup | #1 speedup | Top 10 | Top 100 | Best unscored check | Checks left today | Scored left today |
|---|---|---|---|---|---|---|---|---|
| bf16_proj | 622 of 1713 | 1.3179x | 1.7111x | 1.6534x | 1.5871x | 1.3436x | 25 | 18 |
| attention_decode | 958 of 1345 | 1.4836x | 3.3389x | 3.0077x | 2.6963x | 1.4974x | 23 | 16 |
| moe_w4a16 | 1000 of 1400 | 1.4587x | 2.7817x | 2.6700x | 2.4322x | 1.4625x | 34 | 13 |
| decode_layer | 899 of 1271 | 1.4005x | 2.5762x | 2.4708x | 2.2973x | 1.4121x | 29 | 17 |
| prefill_layer | 623 of 876 | 1.3442x | 2.1471x | 2.0292x | 1.9157x | 1.3458x | 26 | 19 |

(Speedups of checks are computed against the leader's time and can differ from the judge's by ~0.1%. Run-to-run noise is about +-1%.)

## Per track: where we are, what we ship, where the time goes

### bf16_proj (ours 1.3179x, #1 1.7111x, rank 622 of 1713)

Scored runs so far (oldest first):
- v1 (1.117x): First scored entry, 9/9 tests, 1.117x (CI 1.100-1.134).
- v2 (1.192x): Scored 1.192x (CI 1.174-1.210), 9/9 tests; prepare 10.7 s.
- v4 (1.247x): Scored 1.247x (CI 1.228-1.265), 9/9 tests, prepare 54 s.
- v6 (1.318x): Scored 1.318x (CI 1.298-1.338), ~rank 599 of 1,700; 9/9 tests, prepare 114 s.

Biggest remaining lever: m = 32/128/256 still run SGLang's cuBLAS (9 of 30 shapes earn nothing); at m = 256 the tensor-core floor (~50 us) is close to the HBM floor (~52 us). Own tensor-core GEMM for m = 32-256, no separate split-K reduce kernel.

### attention_decode (ours 1.4836x, #1 3.3389x, rank 958 of 1345)

Scored runs so far (oldest first):
- v2.2 scored (1.218x): First scored run, 16/16 at 1.218× (CI 1.200-1.243), our board entry.
- v3 scored (1.277x): Our best board entry at the time; board still showed 1.218× right after.
- v5 scored (1.406x): 1.406× (CI 1.385-1.429), then our best.
- v6 scored (infra error) (no score): Refunded, did not count; resubmitted later with the user's go-ahead.
- v6 scored (1.459x): 1.459× (CI 1.437-1.481), our best at the time.
- v8 scored (1.484x): New board entry for the track.

Biggest remaining lever: Full-attention verify at 16/32 requests still runs SGLang's FA3 at ~32% of speed-of-light; decode 16/32 at ~60%. FP8 verify kernel is the biggest lever.

### moe_w4a16 (ours 1.4587x, #1 2.7817x, rank 1000 of 1400)

Scored runs so far (oldest first):
- v1 (1.270x): First scored entry; same kernels as SGLang with better call parameters.
- M11 hybrid (1.303x): First M11 scored run; modest gain over v1.
- M11 E3c (1.443x): Big jump from 1.303x; the main early scored result.
- E8 (1.440x): Tuner kept all incumbents; memory side at small-transfer ceiling, math is slow.
- E9 (1.441x): Incumbents kept again; M15 correct but not faster.
- E12 (1.439x): Switched classes didn't hold on the judge; 3% margin is inside tuner error.
- E13 (1.442x): RS-wgmma serialized by ptxas; incumbents kept.
- E14 (1.443x): ptxas injects wgmma arrives from operand copies; RS-wgmma dropped as a path.
- E15 (1.455x): PDL tail on M11 wins at bs1; best so far.
- E16 (1.459x): Small best-so-far gain; three different large-batch kernels all stall at ~310-320 us bs256.

Biggest remaining lever: 4-bit decode in registers keeps small batches at 23-41% of speed-of-light (decode ALU/issue-bound, few warps in flight, not HBM-bound). 128/256 tokens still run SGLang's Marlin.

### decode_layer (ours 1.4005x, #1 2.5762x, rank 899 of 1271)

Scored runs so far (oldest first):
- v2 scored (1.185x): First scored run; 1.185x would rank about 1,112 of 1,234 on the board.
- v4 scored (1.356x): Scored 1.356x, board rank about 918 vs leader 2.540x.
- v7 scored (1.364x): Marginal +0.6% over v4; the siblings' newer kernels were not yet embedded.
- v10 scored (1.400x): Best scored run: 1.400x, rank 898 of 1,272 vs leader 2.576x.

Biggest remaining lever: Experts (Marlin at verify 16/32) and attention (FA3 at full verify 16/32) dominate; then fusing the glue kernels (~1-3 us per kernel boundary).

### prefill_layer (ours 1.3442x, #1 2.1471x, rank 623 of 876)

Scored runs so far (oldest first):
- v2 (1.142x): Scored 1.142x, would rank ~705th of 855; leader 2.252x.
- v4 (1.214x): Scored 1.214x; would rank ~660 of 851.
- v5 (1.344x): Scored 1.344x; board still listed v4 1.214x at rank 677; leader 2.147x.

Biggest remaining lever: 16K-token experts still dequantize NVFP4 to BF16 in HBM then run a BF16 grouped GEMM; dense projections are compute-bound cuBLAS (38%); FA3 on 64K attention is 42% of that shape.

## What our best scored run contains (from the track notes)

### bf16_proj: v6 (1.318x)
- Implemented: v4 + F12V (12-bit GEMV) at pooler/qkv m1; F12M (12-bit mma.sync) at o m8, pooler m8/16, qkv m8/16, unpooler m16; v4 kernels kept as fallbacks
- Worked: decode 1 1.55x, verify 1 1.83x; decode 16 1.33x; 12-bit weights gave the gain in streaming kernels
- Did not work: m=32/128/256 only ~cuBLAS: decode 32 1.16x, verify 16 1.15x, verify 32 1.05x

### attention_decode: v8 scored (1.484x)
- Implemented: v6 base; Sd16 UD3 (8,4,3); Fv1 V5 (1,4,1); Fv16 fa3t (4, pack_gqa); Sv1 back on PDL
- Worked: 1.4836× (CI 1.461-1.510) vs 1.459× previous board entry

### moe_w4a16: E16 (1.459x)
- Implemented: M18: decode to shared memory + SS-wgmma, one asm statement per chunk; New row-chunk bank (m18_shuffle), A double-buffered; M16/M17 removed from search; bs8 switched to M11 + PDL post-norm at 2% margin
- Worked: 1.459x (CI 1.437-1.481), new best; bs8 97.5 us (1.548x, was 1.530x); bs1 35.9 us (2.141x)
- Did not work: M18 only ties M14: GEMM1 bs256 319 vs 312 us; GEMM2 170 vs 165 us; M18 at ~22% issue utilization; N<=64 variants >1.25x slower; M18 never selected, output unvalidated on GPU

### decode_layer: v10 scored (1.400x)
- Implemented: Clean v10 = DE12's timed code with its picks; Attention v6 + bf16_proj v6; M11 E3c with PDL path code (V10_PDL empty)
- Worked: Scored 1.400x (CI 1.379-1.421x), +2.6% over v7's 1.364x; dec1 1.695 / 1.627, dec16 1.460 / 1.603
- Did not work: ver16/ver32 stay 1.261/1.128 and 1.182/1.113 (Marlin-bound)

### prefill_layer: v5 (1.344x)
- Implemented: R14 CUDA W4A16 grouped GEMM on turn 1 and turn 4 (tiling mode 1, (2,8,1)); Division-free ring counters; First turn on v2 grouped path; prepare GPU self-check (err 5.6e-3)
- Worked: Turn 1 1.262x/1.178x, turn 4 1.386x/1.226x; First turn 1.501x/1.544x/1.346x/1.268x; CI 1.324-1.364

## Questions for brainstorming

1. Given these gaps to the leaders (1.9x on attention, 1.3x on MoE, 1.2x on decode_layer, 0.8x on prefill, 0.4x on bf16_proj), which techniques on Hopper would you expect the leaders to use that we are missing?
2. For W4A16 MoE at small batch on H100/H200: how do the fastest implementations get past the 4-bit decode instruction budget (about 3.7 instructions per weight at full bandwidth)? Anything beyond fragment-ordered weights, register decode, and wgmma RS?
3. For FP8 paged-KV decode/verify attention with 2 KV heads and head dim 64: what limits FA3-like kernels at 16-32 requests, and what layout or scheduling changes would approach the bandwidth floor?
4. For BF16 skinny GEMM (m = 32/128/256, K = 4096-8192) where cuBLAS reaches 50-75% of the memory floor: what split-K, cluster/DSMEM, TMA multicast, or stream-K schemes are worth trying?
5. Which cross-kernel fusions (norm + residual, routing, combine, PDL overlap) give the most in a whole decoder layer at decode batch 1-32?
6. What would you try in the remaining ~15 hours, given about 30 checks per problem, to climb the most places per hour of effort?

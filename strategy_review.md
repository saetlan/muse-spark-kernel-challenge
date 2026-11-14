# Strategy review: the ChatGPT brief vs where we are (Nov 4, ~17:55 UTC)

Source: `CLAUDE_KERNEL_OPTIMIZATION_BRIEF.md` (snapshot ~16:37 UTC). The track close is Nov 5 07:59 UTC, so ~14 h
are left. Status per idea: **done** = already measured by a session, **new** = not tried yet, **wrong** = the
brief's premise doesn't match our measurements.

## Board now (17:51 UTC)

| Track | Board at the brief | Board now | #1 | Rank now | Factor still needed for #1 |
|---|---|---|---|---|---|
| bf16_proj | 1.318× | **1.351×** | 1.718× | 560 / 1,720 | 1.27× |
| attention | 1.484× | **1.566×** | 3.339× | 886 / 1,349 | 2.13× |
| MoE | 1.459× | **1.488×** | 2.782× | 983 / 1,403 | 1.87× |
| decode_layer | 1.400× | **1.432×** | 2.576× | 882 / 1,273 | 1.80× |
| prefill | 1.344× | 1.344× (unchanged since the morning) | 2.147× | 627 / 878 | 1.60× |

The overall board ranks us 914th.

## Brief vs our measurements

### bf16_proj (brief priority 1)
| Brief idea | Status | Evidence |
|---|---|---|
| Native BF16 small-M GEMM at m = 32 (BM 16, BN 64, no split-K) | **done, lost** | Triton swap/gemm, SX at m 32 (PF35), token groups (PE21, PF39): every variant ≤ cuBLAS (m32 o 14.6 vs 10.8 µs). |
| Plain BF16 WGMMA at m = 128/256 before compression | **done** | PE22/PE23: our build of DeepGEMM's kernel = DeepGEMM's own time (pooler m128 26.37 vs 26.42 µs) once launched without the cluster attribute (PF41). |
| Router hi + lo in one kernel | **done, won** | PE18/PE20: hl_gemm/hl_swap, router m32/128/256 6.76/8.84/12.60 → 5.86/8.00/10.94 µs (in v10, 1.351×). |
| "m = 256 tensor-core floor ~50 µs" | **wrong** | The judge's floors for m 256 are 9.8 (qkv), 4.5 (o), 17.4 (pooler/unpooler) µs. Pooler math at peak = 17.6 µs. |
| Not in the brief: 12-bit weights in DeepGEMM's skeleton | **in progress** | PE24 correct but 2–5× slow (scattered TMA boxes, PF42). **PE25** (chunk-major planes, fix) is built and waits for the user's go. |

### attention (brief priority 2)
| Brief idea | Status | Evidence |
|---|---|---|
| 64 query rows (8 tokens × 8 heads) share one KV stream, Q tile M = 64 | **done** | W1/W1-N (wgmma FP8, M = 64) is live on Fv1/Fv16/Fv32/Sv16/Sv32 in v9 (scored 1.566×). |
| Measure real overlap / producer-consumer | **done** | AF50 anatomy: consumers are the bottleneck (softmax 835 + un-overlapped wgmma 638 cycles per stage). **W2b** (overlap Q·K(i+1) with softmax(i)) is running as L27. |
| KV lower bound 73 / 146 µs at B = 16 / 32 | agrees | AF43 uses ~78 / 156 µs. |
| **Missing in the brief:** sliding layers are 46 of the 62 calls per load | **new lever** | AF43: every sliding shape at ~6 µs alone → 2.00×; today Sv32 = 24.7 + 6.9 + 2.2 µs over 3 kernels. Full verify alone → 1.81×. |

### MoE (brief priority 3)
| Brief idea | Status | Evidence |
|---|---|---|
| Uniform routing → batch/32 rows per expert | **wrong** | MF19 (40 judge seeds): bs256 = one hot expert (~128 tokens) + ~24 cold ones (16.4 tokens on average). |
| Two algorithms (small vs large batch) | **done** | M11 ≤ 32 tokens; M19/M14/M18 hot/cold split at 128/256 (E18/E19, scored 1.488×). |
| Experiment B: isolate decode vs math vs sync | **done** | MF20: skeleton 186 µs, + decode 55 µs, + wgmma 64 µs at bs256; MF22: a shared-memory weight ring → 4 CTAs per SM, −5–8%. |
| Don't tune M18 further | agrees | The session moved to M19 and M11 task dealing (MF23). |

### prefill (brief priority 4)
| Brief idea | Status | Evidence |
|---|---|---|
| Hybrid: decode a weight tile once into smem, reuse across rows | **done, lost** | R53/R55 (RF27, RF29): shared-memory-bandwidth bound, ~800–1,100 cycles per chunk whatever the structure. |
| R14 on-the-fly decode at 16K | **done, judge loses** | RF26: every lab that moved 16K to R14 scored lower on the judge's routing. |
| Overlap expansion with other work | **done, lost** | RE20/RF30: overlap cost 6.7%. |
| Dense projections + long FA3 matter | **not attacked** | Turn 1 full: FA3 = 35% of time; 64K full: FA3 = 42%. The session is now helping on decode_layer glue. |

### decode_layer (brief priority 5)
| Brief idea | Status | Evidence |
|---|---|---|
| Use the standalone attention's latest routes | **partly** | v12/v13 embed attention **v8**; attention **v9** (W1-N on the verify classes, +5.5% on its own board) isn't in decode_layer yet. |
| Newer MoE path | **in progress** | v16 = v15 + MoE E19's large-batch path at 65–320 tokens (replaces Marlin at verify 16/32). |
| Pad-free unpooler tail | **done** | DE16 (`v13_pad`, one cat instead of F.pad's 3 kernels). |
| One routing kernel; GEMM2 + post-norm fused | **in progress** | D17 delegated to prefill (`shared/glue/`); MoE's T33 tracks the same kernel. |

## What the brief got right, in our terms
- **Pick fewer, falsifiable experiments.** The sessions already do this; the labs fit the method.
- **Pivot on failure.** That fits prefill, whose score hasn't moved in ~12 h.

## Recommended reorganization (for the user to decide)

1. **decode_layer: embed attention v9 (W1-N verify routes) next to the MoE E19 path (v16).** It's the
   cheapest large gain: its verify 16/32 shapes spend ~381 µs in FA3 and ~440 µs in Marlin at verify 32,
   and both siblings now have judged replacements. Later, W2b and bf16_proj's v10 routers follow the same path.
2. **bf16_proj: PE25 (12-bit DeepGEMM fork with the TMA fix).** It's the only path left with a large upside on
   m = 32/128/256 (score upper bound ~1.55×). If PE25 still lands below DeepGEMM's BF16 time, stop and keep v10.
3. **attention: W2b (running), then the sliding layers' fixed cost** (one kernel instead of attend + merge +
   prologue at Sv16/Sv32/Sd32). Per AF43 the sliding layers are worth as much as the full verify ones.
4. **MoE: keep going** (M11 task dealing at ≤ 32 tokens, M19 at 128/256); its results flow into decode_layer.
5. **prefill: decide.** Either (a) keep it as the glue helper (its own levers measured flat or negative), or
   (b) give it one last shot at its FA3 share (FA3 settings / split for turn 1 and 64K full).
   (a) seems the better use of the remaining ~14 h.

Not worth the remaining time: more tiling sweeps at m = 32 on bf16_proj (PF39), M18 tuning (MF20), more
prefill expert-decode structures (RF29).

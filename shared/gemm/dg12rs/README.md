# shared/gemm/dg12rs — DG12-RS: 12-bit F12 weights × BF16 x with register-A wgmma (TRACKER shared item 3)

Built by muse_prefill_layer for muse_bf16_proj (owner), Nov 4 18:20–18:40 UTC. Nothing in `proj/` was touched (only
read: `f12m/f12m_layout.py`, `f12m/f12m.py`'s decode/epilogue, `submission_v10.py`'s F12V helpers).

## Entry points (`dgrs.py`; v3 table, v4 / v5 additions below)

| Function | What |
|---|---|
| `dgrs_pack(p)` | your `f12m_device_planes(w, natural=True)["p"]` → chunk-major [K/64][N/16][3][32][4] (lossless) |
| `dgrs_buffers(m, n, bm, splits, device)` | err flag; for `red=1`: fp32 partials + int32 counters (zeroed once, zero after every call) |
| `dgrs_build(instances=None)` | one `load_inline` module; instances = (BM, CPS, DIAG) tuples (or plain BM ints); default (32,1,0), (64,1,0), (128,1,0) |
| `dgrs_run(ext, x, wp_cm, rec, out, bufs, bm, splits, smax=8, cps=1, diag=0, red=0)` | out = x @ Wᵀ (bf16); one launch |
| `ext.dgrs_attrs(bm, cps, diag)` | (registers, spill bytes) |

- `splits` 1..8 (every slice non-empty). `red=0`: S > 1 launches a cluster (1, 1, S) reducing through DSMEM; `red=1`:
  plain launch, global fp32 partials + self-resetting last-arriver counter (the fallback if real clusters are slow).
  S = 1 is always a plain launch with an in-CTA finalize.
- `cps` 1 or 2 CTAs per SM (`__launch_bounds__(384, cps)`, ring 200 / cps KB; cps 2 only for BM ≤ 64).
- `diag` bits (lab): 1 no finalize, 2 no decode (raw words as A), 4 no x TMA, 8 no weight copy.

## v4 additions (Nov 4 ~20:55 UTC, after bf16_proj's PE31: the math chain, not bandwidth, is the limit)

- `math` = 2 or 4 math warpgroups (4: 640 threads, 1 CTA/SM, BM ≤ 64; setmaxnreg producer 24 / math 112 within the
  96 × 640 launch allocation, F57/F58). With 4, pair p (warpgroups 2p+1, 2p+2) takes chunks p, p+2, … of the same
  128 rows (even ring), releases its own previous chunk, writes its own fp32 tile; tiles summed pair 0 + pair 1 before
  the finalize. Each math branch now writes its tile itself after a math-only named barrier (no accumulators live in
  the producer's code).
- `diag` bit 16: keep the math (all accumulators feed a store under a condition ptxas can't prove false), skip the
  finalize, so the per-chunk chain can be timed without ptxas deleting it (PE31's d1 build had 22 registers).
- `ext.dgrs_max_clusters(bm, cps, diag, math, splits)`: cudaOccupancyMaxActiveClusters for (1, 1, splits) clusters.
- API: instances (BM, CPS, DIAG[, MATH]); `dgrs_run(..., red=0, math=2)`; `ext.dgrs_attrs(bm, cps, diag, math)`.

## v10 (Nov 5 ~02:20 UTC): shape specialization

**Background.** bf16_proj's PE38 (job ed0cce4b) settled both v9 experiments:

| Build | unpooler m32 | o m32 | Verdict |
|---|---|---|---|
| Specialized d0 (RED 2, SPL 2) | 20.96 µs (SGLang 22.86, −8.3%) | 12.67 µs | **Win** |
| Generic d0 | 23.81 µs | 14.26 µs | |
| DIAG 512 (ring corrections) | | | **Loss, keep it off** |

DIAG 512 lost because the producer warps, as a third ring consumer, throttled the ring: the main loop got +2.8 µs. A d0 instance never compiles `pre_ring` (it sits behind a constexpr-false `RING`), so specialized d0 is the ship candidate.

**What v10 adds.** Four more optional template parameters, SMX / M / N / K (0 = runtime): instance tuples (BM, CPS, DIAG, MATH, RED, SPL, SMX, M, N, K).
- With `smax` constant, S and every `i % S` and `i / S` in the producer and math loops fold.
- With M, N and K constant, chunk counts, strides and live checks fold.
- `dgrs_gemm` matches each set field exactly and prefers the most specialized instance.
- `ext.dgrs_attrs_x(...)` returns the attrs of one exact 10-field instance.

**Checks.**
- Dispatch order: most specialized first, generic last.
- 184 pins, clang including the four shape instances.
- Full emulation: 16 cases at 1.6–1.7e-3.
- `mutate_dgrs.py`: 87/87 caught (v10 adds 3 dispatch / application mutants).

sha256 of `dgrs.py` v10: f4ff43fafffd5af2…; v9 kept as `dgrs_v9.py`.

## v9 (Nov 5 ~01:50 UTC): escape corrections from the ring (DIAG 512), after bf16_proj's user chose one more round

PE36 falsified v8's cold-code idea (DIAG 256 is slower; keep it off). This round targets the one measured cost with a known cause: `pre_corr`'s ~4 MB of 32-byte gathers stretch the main loop by ~0.9 µs on unpooler m32.

**DIAG 512 (any red; not with DIAG 32; ≤ 256 chunks per slice, host-checked).**
- **Escapes in its own K range.** Each slice CTA corrects only the escapes whose k falls in its own chunks, for all 128 rows of the row block, using x straight from the ring's swizzled tiles. There are no global gathers.
- **Producer warps 1–3 (`pre_ring`), before the loop:**
  - Rows 43·pw … (all 128): scales, indexed by absolute row.
  - A list of the in-range escapes, in row then record order.
  - Bucketing by chunk with `__match_any_sync` ranks, a per-lane scan and a scatter. This is a stable sort, so the result is deterministic.
- **Producer warps 1–3, per chunk in ring order:**
  - Wait on the slot's full barrier.
  - For each escape in the bucket, with lane = token: x = tile[t·128 + (((kk >> 3) ^ (t & 7)) << 4) + (kk & 7)·2], then corr[t][row] = fma(x, true, fma(−x, wdec, c)).
  - Arrive on the slot's empty barrier, whose count becomes 2 + 3.
- **Math warps, after the loop:** sync on bar 2. Pair 0 folds acc += corr · 2^(112 − base), where `__frcp_rn` of a power of two is exact. Reduction then runs as in v8 (red 0 / 1 / 2). The finalize is Σ slots × scale, with no `sCorr` reads.
- **Shared memory.** `sCorr` becomes [BM][129]. The buckets reuse the gather staging area. `ext.dgrs_stages_d(bm, cps, math, splits, red, smax, diag)` gives the stages for a DIAG.
- **Verified locally:**
  - `pre_ring` port vs the direct chain (chunk then record order), **bit for bit**: BM 32/64/128, slices of 7–256 chunks, lists up to 301. The bucketing equals a stable sort by chunk. x is read from byte-accurate swizzled tiles.
  - Ring protocol with the 3 producer warps as consumers (empty count 5); a planted missing full wait is caught.
  - Layer emulation with the fold, for the m32 cases.
  - Barrier sites: bar 2 is synced once per launch, before the fold or before the finalize.
  - 179 pins, clang including DIAG 512/528/576 and the specializations.
  - `mutate_dgrs.py`: 84/84 caught. v9 adds 11 ring mutants and 3 specialization mutants, and now checks every anchor before running.

**RED / SPL specializations (v9).** These are optional template parameters: an instance tuple (BM, CPS, DIAG, MATH, RED, SPL) fixes red and splits at compile time, so the row share, the thread map and the slot loop fold to constants, and the other modes' code drops out. bf16_proj's PE37 (job c7dde285) showed a ~0.8–1 µs fixed cost on entering the finalize. That cost is local: it shows at S1 and without the precompute, and pull and push cost the same. This specialization is a code-size test for it.
- Dispatch prefers a matching specialized instance; 4-tuples stay generic. `ext.dgrs_attrs_s(bm, cps, diag, math, red, spl)` returns one exact instance's attrs.
- Checks: folded row shares equal the runtime ones (S 1..8, red 0/1/2, every rank), and the dispatch order is checked.

sha256 of `dgrs.py` v9: c8b76b59502806b8…; v8 kept as `dgrs_v8.py`.

## v8 (Nov 5 ~00:20 UTC), after PE35 (job f1139823): the excess sits inside the CTAs

PE35 per-CTA stamps, unpooler m32 r2 m4, median / max:

| Phase | Time |
|---|---|
| Launch skew | 0.16 µs |
| Cluster launch vs plain launch | ≤ 0.34 µs |
| Main loop | 13.00 / 13.87 µs |
| Push + peer wait | 1.48 / 2.32 µs |
| Bar 2 | 0.02 µs |
| Final pass | 1.55 / 1.59 µs |
| Exit | 0.08 µs |

The final pass is the same 1.51 µs on o. That's ~2,750 cycles for ~24 shared-memory loads and 4 stores per thread, a fixed cost that doesn't depend on the shape. Meanwhile the small code regions (bar 2, exit) cost nearly nothing. This fits cold instruction-cache misses: the epilogue runs once, after a long hot loop.

The precompute's gathers stretch the loop by 0.89 µs. That matches ~4 MB of 32-byte sector traffic at 4.66 TB/s, so it's bandwidth and pacing won't help. Sourcing x from the ring's tiles would (v9 if worth it).

**What v8 adds:**
- **DIAG 256 (red 2, splits > 1; host-checked):** a dry pass of the math body before the main loop. It skips the math, executes the push with predicated-off DSMEM stores (`st_cluster_f32_if`), runs the finalize without stores, and takes no barriers. It runs while the ring's first copies are in flight, so the push and finalize code is cached when the real pass reaches it.
  - **One body copy:** the body is a lambda called once with `real = true` (the non-DIAG-256 builds fold back to v7's code), or twice inside an opaque `#pragma unroll 1` loop.
  - Values are unchanged. It's a production candidate if it removes the ~1.5 µs.
- **New stamps:** [12] dry pass done, [13] pushes issued (before the phase-B arrive), [14] finalize loads summed (before the stores).
- **Checks:**
  - New: no barrier, store or math is reachable in the dry pass (structural walk of the body).
  - Updated: barrier matching, stamp slots 0..14.
  - 137 pins, full emulation (12 cases at 1.6–1.7e-3), clang including DIAG 256/320/352.
  - `mutate_dgrs.py` 70/70 caught. v8 adds 6 dry-pass mutants: pushes, phase-A wait, bar 2, main loop, out stores, and an opaque loop.
- sha256 of `dgrs.py` v8: c80ffca870818bc9…

## v7 (Nov 4 ~23:20 UTC): diagnostics only, after PE34 (job 41316acb)

PE34: red 2 saves 1.15 µs (unpooler m32 22.16 µs, +1.6% vs SGLang 22.51), but d0 − d16 is still 4.3 µs.

**Two-point fit of PE34's d16 lines.** o and unpooler m32 run the same grid (bm32 s2 m4, 128 CTAs) with 8 vs 32 chunks per CTA (10.24 / 18.34 µs):

| | Slope (steady streaming) | Fixed per-call cost |
|---|---|---|
| DG12-RS | 0.3375 µs per chunk = 4.66 TB/s of 12-bit bytes | 7.5 µs (≈ 11.8 µs with the epilogue) |
| SGLang, same fit | 4.21 TB/s of BF16 bytes | 6.6 µs |

So streaming is not the problem; the fixed cost is.

**DIAG 64.** 16 u32 per CTA go into `part` (red 0 / 2 only; the host checks the size). Values are unchanged.

| Slot | Contents |
|---|---|
| 0, 9 | %globaltimer at entry / exit |
| 1 | %clock at entry |
| 2 | Producer's last copy issued |
| 3 | `pre_corr` done |
| 4, 11 | Math loop end for pair 0 / pair 1 |
| 5 | Reduced (after the push or pull barrier) |
| 6 | Barrier 2 passed |
| 7 | Finalize done |
| 8 | Exit |
| 10 | SM id |

Clocks are per SM: use differences within a CTA, and use globaltimer for spread across CTAs.

**Other changes.**
- `prefetch.tensormap` of the x map at entry (as CUTLASS does); DIAG 128 turns it off.
- Checks: stamp slot / guard check, 127 pins, 64/64 mutations (the "stamp unguarded" mutant was missed at first; the guard check was tightened and now catches it).
- sha256 of `dgrs.py` v7: b6a3340d71fb5290…

## v6 (Nov 4 ~22:40 UTC, after bf16_proj's PE33, job a7f74880: unpooler m32 22.54 µs = SGLang; 3.86 µs after the main loop on unpooler, 3.37 on o)

- **`red = 2`: push reduction (new; red 0 / 1 unchanged).**
  - Each slice stores its accumulators, both MATH 4 pairs, straight into the owner rank's receive area: rank = row / RS, slot = sl · NP + p, layout [slot][BM][RSP] fp32 via `mapa` + `st.shared::cluster`.
  - Then one cluster barrier, and each rank finalizes from its own shared memory, summing slots in (slice, pair) order.
  - This removes the ring-done barrier, the tile write, the MATH 4 pair-sum pass, the DSMEM pulls and the exit cluster barrier.
  - **Cluster start:** every thread does a relaxed `barrier.cluster.arrive` at kernel start and waits on it before the first push, which guarantees every peer CTA is running.
  - **RSP** = RS rounded up to 4 mod 8, so each warp's pushes hit 32 distinct banks and the finalize reads are contiguous.
- **Async gathers in `pre_corr`.** It now fetches x with 4-byte `cp.async`: the word holding element k & ~1, half-selected by k & 1, zero-filled for dead tokens. It runs G = 512 / BM list entries per batch, double-buffered, so no registers are held while loads are in flight.
  - PE33's 40 B MATH 4 stack was `pre_corr`'s: the d32 build has the same finalize code and 0 B.
- **Shared memory** adds a 12 KB gather staging area, plus the receive area for red 2 (splits · NP · BM · RSP · 4 B, e.g. 34 KB at BM32 MATH 4 S2). Ring stages at smax 8:

  | Config | Stages |
  |---|---|
  | red 2, BM32 MATH 4 | 8 |
  | red 2, BM64 MATH 4 | 6 (4 at S6) |
  | red 2, BM128 | S2 rejected (`dgrs_stages` = 0); 4 at S3–5 and S7–8, 0 at S6 |
  | red 0, BM128, S 1–4 | 5 / 6 / 6 / 6 (v5: 5 / 6 / 6 / 7) |
- **Verified locally:**
  - 123 pins.
  - Barrier matching for red 0 and red 2: one start arrive, then wait / arrive / wait per role, with the pushes between phase A's wait and phase B's arrive.
  - Push map: every (slice, pair, row, token) lands once, each rank reads exactly its S · NP partials, conflict-free, S 2–8, MATH 2 / 4.
  - Layout for every config, including red 2.
  - `pre_corr` with staged words, bit-exact vs the direct chain in 114 cases.
  - 12 real-size cases at rel err 1.6–1.7e-3.
  - clang.
  - `mutate_dgrs.py`: 62/62 caught. v6 adds gather half / align / buffer / wait depth, staging size, push padding (bank check) / slot / row + 8 rank / token stride / slots read / receive size / phase A wait / producer barrier B / exit barrier / cluster launch.
  - sha256 of `dgrs.py` v6: 09533e32b2d92d2b…; v5 kept as `dgrs_v5.py`.

## v5 (Nov 4 ~21:50 UTC, after bf16_proj's PE32, job 7ae14723: MATH 4 hides the chain, the finalize costs ~7.3 µs, MATH 4's full build spilled 336 B)

- **Math-only epilogue.** The producer warpgroup (24 registers under MATH 4) never runs epilogue code: tile writes, the
  pair sum, red 1's partials and counter, and the finalize are done by the 128 × MATH math threads (named barrier 1),
  so MATH 4 d0 should compile without stack (check `dgrs_attrs`).
- **Precompute in the producer's idle warps.** While the math runs, producer warps 1–3 (`pre_corr`) take this CTA's
  finalize rows (rank share; all 128 for red 1). For each row they write the scale, list its escapes (warp scan of the
  counts) and zero its corrections. Then, with lane = token, they apply the list U at a time (U = 4 under MATH 4, 8 under
  MATH 2), issuing the x gathers first: corr[t][row] = fma(x, true, fma(−x, wdec, c)) per escape, in record order.
  These go into shared memory: sScale [128], sEsc [3][304] uint2, sCorr [BM][RS + 1] fp32.
- **Finalize** = Σ slices (DSMEM / tile / global partials, rank order) → `fmaf(sum, scale, corr)` → BF16, with no global
  loads. FU = 8 tokens per step under MATH 2, 4 under MATH 4 (one step at BM 32, S 2).
- **Barriers.** The producer arrives once on named barrier 2 (`barrier.arrive`, non-aligned), and the math threads sync
  on it before the finalize. The producer also passes the math threads' two cluster barriers (S > 1, red 0), so no
  thread relies on exited-thread rules.
- **Shared memory.** 226 KB per CTA (112 KB at CPS 2); the ring gets what's left after the 7.8 KB scale / escape areas
  and the corrections. Ring stages at smax 8 are unchanged for BM 32/64. BM 128 gets 5 / 6 / 6 / 7 at S 1 / 2 / 3 / 4
  (v4: 7). `ext.dgrs_stages(bm, cps, math, splits, red, smax)` returns S, or 0 for a launch `dgrs_gemm` rejects (fewer
  than 4 stages, or the epilogue tiles don't fit).
- **DIAG 32** (lab): skip `pre_corr`. Values are wrong; d0 − d32 times what the precompute adds to the critical path.
- **API unchanged** apart from `dgrs_stages`.
- **Verified locally:**
  - 99 pins.
  - A barrier-matching check: producer and math barrier counts per role, bar 1 math-only.
  - Layout checks for every config × S 1..8 × red × smax.
  - `pre_corr` port vs a direct per-row chain, **bit for bit**, in 114 cases (every rank share of S 1..8, BM 32/64/128,
    random and all-7 escape rows; list ≤ 304).
  - Finalize map for 256 / 512 math threads.
  - 12 real-size cases at rel err 1.6–1.7e-3.
  - clang syntax incl. DIAG 32.
  - `mutate_dgrs.py`: 47/47 caught (v5 adds: list offset, scan lanes, escape word, row split, tag, zeroing, live token, warp 0 in pre_corr, corr stride, dropped corr, escape area, odd MATH 4 ring, a missing producer cluster barrier, a missing producer arrive).
  - sha256 of `dgrs.py` v5: a1373577236e2420…; v4 kept as `dgrs_v4.py`.

## Kernel (v3)

- CTA = 128 weight rows × BM tokens × one K slice; warpgroup 0 = producer (thread 0), 1–2 = math (64 rows each).
- Producer per 64-k chunk: one `cp.async.bulk` of the 12,288 weight bytes + one TMA 2-D load of the x tile (tensor map
  over x [M][K], box 64 × BM, SWIZZLE_128B, rows ≥ M zero-filled; encoded per call with dlopen'd
  `cuTensorMapEncodeTiled`), both on the stage's full barrier (count 1, expect_tx = both). v2's 128 cp.async threads,
  their 128 arrivals per stage and the math warps' per-chunk proxy fence are gone (TMA writes are async-proxy).
- Math warp w of warpgroup c = row tile 4c + w: 3 × `ld.shared.v4`, F12M's 7-op decode into wgmma's A registers,
  4 × `wgmma m64nBMk16` RS, `wait_group<1>`, release. Fragments triple-buffered.
- Epilogue: fp32 tile [BM][128] in the CTA's own (free) ring memory; cluster barrier (red 0) / last-arriver (red 1) /
  `__syncthreads` (S = 1); rank r finalizes rows [r·⌈128/S⌉, …) (all 128 for red 1 and S = 1): thread → (row, token
  lane), record / scale / escapes loaded once, slices summed in rank order (deterministic), F12M's scale and escapes,
  BF16 stores contiguous on rows; final cluster barrier.

## Verified locally (CPU, no GPU)

- `emu_dgrs.py` (full, ~7 s) / `emu_dgrs.py quick`: with your own encoder: the kernel's offsets + a port of f12m_decode give
  exactly W · 2^(112 − base) (escapes 2^−15) for all four layers; x tiles read back exactly through the SW128 rule;
  split-K GEMM + epilogue vs FP32 x @ Wᵀ for qkv / o / pooler / unpooler at m = 32 / 128 / 256 (12 cases, 2,000–4,600
  escapes each): rel err 1.6–1.7e-3, worst row ≤ 2.7e-3; the v3 finalize map covers every (row, token) once over cluster ranks and threads for S = 1..8 and BM 32/64/128;
  53 CUDA lines pinned (incl. red 1 and MATH 4); MATH 4 / MATH 2 chunk schedule simulated with hardware parity waits
  (S 4–8, 1–22 chunks; a planted wrong release is caught); finalize map also for 640 threads.
- `check_dgrs_syntax.py`: clang -fsyntax-only, host + sm_90a, BM 32/64/128 + CPS 2 + DIAG 1/2/4/8/15 (own stub copy in
  `stubs/`: decode_layer's + cluster attribute + driver TMA types + dlfcn).
- `mutate_dgrs.py`: 33/33 planted bugs caught (v4 adds pair chunk start / index / tile / sum) (decode, slots, row-tile map, strides, TMA coordinate and box, descriptor,
  split size, tile rows, rank row share, token lanes / step, escapes, scale, barrier count, tx bytes, release slot,
  red 1 counter / reset / partial index / single finalizer, pack).
- Not verified: anything on the GPU (timing, registers, the cp.async → wgmma visibility at scale). Expected from R14:
  ~790–930 cycles per 128 × 64 stage at 1 CTA/SM → ~3.5–4 TB/s of 12-bit bytes if that per-stage time holds here.

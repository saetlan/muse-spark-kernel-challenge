# Focus prompts for the 5 track sessions (Nov 4, ~13:00 UTC)

Paste each block into its session yourself: the sessions don't act on relayed go-aheads.
Numbers come from each track's best scored run's judge profile (see `current_state.md`).
All times are per call, in µs (microseconds) unless marked ms; speedups are × over SGLang.
If you also want them to spend checks/runs without asking, add a line like
"You may run okc check / okc submit for this track without asking until <time> UTC."

---

## muse_bf16_proj

```
Focus for the rest of the kernel track (closes Nov 5 07:59 UTC): the shapes where we still run
SGLang's own kernel or barely beat it. From the scored run 33748bdb (submission_v6.py, 1.318×):

- m = 32: qkv_gate 14.7 µs (SoL 8.0 µs, 55%), o 6.7 µs (54%), pooler 20.1 µs (71%), unpooler
  18.9 µs (75%) — all cuBLAS nvjet = SGLang, 1.00×; qkv_gate pays a separate split-K reduce kernel
  (8.6 µs GEMM + 1.9 µs reduce).
- m = 128: qkv 16.4 µs (51%), o 7.5 µs (53%), pooler 21.4 µs (69%, DeepGEMM), unpooler 20.0 µs (73%).
- m = 256: qkv 19.9 µs (49%), o 9.6 µs (47%), pooler 30.8 µs (56%), unpooler 24.4 µs (71%).
- router at every m is 20–44% of SoL (fixed-cost bound: 4.0 µs for 1.8 µs of bytes at m = 1).
9 of 30 shapes sit at 1.00×. Goal: a tensor-core kernel family for m = 32/128/256 only, added to
the per-shape dispatch where the lab proves it faster (PF32–PF34 list what failed so far: the
per-chunk handshake, wide clusters, latency-bound pipelines). Consider the 12-bit lossless weights
(F12) in a wgmma kernel (−25% bytes), and fusing the split-K reduction into the kernel.

New scored-run rule (CLAUDE.md rule 1): propose a scored run at ≥ 1.5% over the board; anything
clean and passing below 1.5% goes in TRACKER.md "Final-round candidates". Your 3.60 ms check
4b8e250d (≈1.344×, +2.0% over 1.318×) now qualifies: propose it.
You own shared block ② in TRACKER.md "Shared building blocks": keep its row current so
decode_layer and prefill can embed your latest judged file.
Always give units with every number (µs, ms, MB, TB/s, %).
```

---

## muse_attention_decode

```
Focus for the rest of the kernel track (closes Nov 5 07:59 UTC). From the scored run f9b4a77b
(submission_v6.py, 1.459×), the weak shapes:

- verify 16 full: 205.7 µs (SGLang FA3 194 µs inside), SoL 63.4 µs → 31% of SoL, 1.00×.
- verify 32 full: 390.1 µs (FA3 373 µs), SoL 126.8 µs → 33% of SoL, 1.00×.
  These two are ~60% of our total time. Our decode kernel dec4 already streams the same FP8 cache
  at ~60% of SoL, so a verify kernel at that level would make them ~2× faster: L16 is the lever.
- verify 32 sliding: 35.1 µs vs SGLang 34.5 µs → 0.985×, i.e. slower than SGLang: fall back to
  SGLang's kernel or re-tune the Triton config for that class (free ~1.5% on that load).
- verify 1 full: 1.14× (31.5 µs; ver4 24.5 µs + combine 2.8 µs + prologue 1.3 µs; SoL 4.3 µs).
- sliding layers are fixed-cost bound (0.1–3.7 µs of bytes, 9–35 µs of time, 3 kernels each):
  fewer kernels (fuse prologue/combine, PDL) is the lever there.

New scored-run rule (CLAUDE.md rule 1): propose a scored run at ≥ 1.5% over the board; anything
clean and passing below 1.5% goes in TRACKER.md "Final-round candidates" (e.g. check 2ac5385c,
2.56 ms ≈ 1.47×, +0.8%).
You own shared block ③ in TRACKER.md "Shared building blocks": keep its row current so
decode_layer and prefill can embed your latest judged file.
Always give units with every number (µs, ms, MB, TB/s, %).
```

---

## muse_moe_w4a16

```
Focus for the rest of the kernel track (closes Nov 5 07:59 UTC). From the scored run f43d8045
(1.459×; per-shape numbers below from the e14 profile 2e2f531a):

- decode 32: 195.4 µs vs SGLang 215.6 µs → only 1.10× (SoL 70.8 µs, 36%). The gated W13 kernel
  m11<2,2,2,4,true> takes 133.1 µs here vs 53.7 µs at decode 16 (2.5× for ~1.4–1.6× more bytes),
  and the same config runs in 63–75 µs inside decode_layer's decode-32 profile. Find out why
  (routing spread of the judge's inputs, config pick, wave quantization) — likely the cheapest win.
- verify 128: 331 µs (SGLang Marlin 345 µs inside + ~20 µs of glue), SoL 101 µs → 31%.
- verify 256: 470 µs (Marlin 489 µs), SoL 109 µs → 23%.
  E12 / RS-wgmma for 128/256 tokens is the lever; it also lands directly in decode_layer's
  verify 16/32 shapes (Marlin is 47–68% of those).
- decode 1 (30% of SoL) and verify 8 (40%) still leave room in the m11 path.

New scored-run rule (CLAUDE.md rule 1): propose a scored run at ≥ 1.5% over the board; anything
clean and passing below 1.5% goes in TRACKER.md "Final-round candidates".
You own shared block ① (≤ 256 tokens) in TRACKER.md "Shared building blocks": keep its row
current; the prefill session's R14 solves the same 4-bit problem at 753+ tokens — read its
findings (RF21, R47, R48: cheaper 4-bit decode bit layout, 128-wide K stages) for ideas.
Always give units with every number (µs, ms, MB, TB/s, %).
```

---

## muse_decode_layer

```
Focus for the rest of the kernel track (closes Nov 5 07:59 UTC). From the scored run 0dcf0c8e
(submission_v7.py, 1.364×):

1. Integrate the siblings' latest judged files (TRACKER.md "Shared building blocks"):
   - bf16_proj v6 (proj/scored/submission_v6.py, 33748bdb, 1.318×). v7 still runs the older
     Triton projections: at decode 1, pj__fma ×4 = 42.0 µs where bf16_proj v6's f12v kernels take
     31.8 µs for the same 4 layers (≈ −10 µs of 103 µs, ~10% on decode-1 shapes); at verify 1 the
     projections take ~53 µs (cudnn 11.8 + swap 21.0 + fma 8.8 + nvjet 7.0 + tinygemm 4.9 µs) vs
     ~38 µs in bf16_proj v6 at m = 8 (≈ −15 µs).
   - attention v6 (f9b4a77b, 1.459×) and MoE e14 (f43d8045, 1.459×).
2. Weak shapes: decode 32 sliding 1.14× (233 µs: experts 110 µs, projections ~66 µs on cuBLAS,
   attention 21 µs), verify 16 full 1.15× (613 µs), verify 32 full 1.12× (945 µs: Marlin ~440 µs,
   FA3 381 µs, projections 134 µs). Most of it is the siblings' open gaps; your own lever is the
   glue: 2 × norm_residual ~4 µs, routing ~2.5 µs, post-norm ~3 µs, attention prologue/combine,
   plus kernel boundaries (qkv epilogue → attention prologue, router + routing, post-norm +
   unpooler, PDL).

New scored-run rule (CLAUDE.md rule 1): propose a scored run at ≥ 1.5% over the board; anything
clean and passing below 1.5% goes in TRACKER.md "Final-round candidates".
You own shared block ④ (glue) in TRACKER.md "Shared building blocks".
Always give units with every number (µs, ms, MB, TB/s, %).
```

---

## muse_prefill_layer

```
Focus for the rest of the kernel track (closes Nov 5 07:59 UTC). From the scored run c8adfdf2
(v5_scored.py, 1.344×), the weak shapes:

- first turn 32K sliding (weight 138, the largest in the score): 11.1 ms vs SGLang 17.2 ms.
  Experts = NVFP4 → BF16 dequant 0.81 ms written to HBM + CUTLASS grouped GEMM 4.14 ms; a fused
  4-bit GEMM (R14 at 16K, RE9 already started picking it) removes the dequant round trip.
  Dense projections 4.23 ms are SGLang's cuBLAS (compute-bound; activations stay BF16 per rule 6).
- turn 1 full: 1.18× (1.19 ms; FA3 attention 0.42 ms = 35%).
- turn 1 sliding: 1.26× (0.84 ms; R14 experts 0.47 ms vs your ~0.25 ms target).
- first turn 64K full: 1.27× (19.2 ms; FA3 8.1 ms = 42%).

New scored-run rule (CLAUDE.md rule 1): propose a scored run at ≥ 1.5% over the board; anything
clean and passing below 1.5% goes in TRACKER.md "Final-round candidates".
Shared block ① (TRACKER.md "Shared building blocks"): you own R14 (753+ tokens); the MoE session
owns the ≤ 256-token kernels — share findings both ways (4-bit decode layout, K-stage width,
TMA multicast). You also consume blocks ② bf16_proj and ③ attention where they fit your shapes.
Always give units with every number (µs, ms, MB, TB/s, %).
```

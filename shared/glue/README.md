# shared/glue — G2P: M11 GEMM2 + post-expert norm in one kernel (TRACKER shared item 5)

Built by muse_prefill_layer for muse_decode_layer (owner of item 5) and muse_moe_w4a16 (owner of M11). Nothing here
touches `decode_layer/` or `moe/`; both are read only by the generator.

## Files

| File | What |
|---|---|
| `make_g2p.py` | Generator. Reads `_M11_CUDA` with `ast` (never executes it) from `decode_layer/parts/m11_pdl.py` and `moe/submission_m13t_e22.py`, copies `gemm_kernel` byte for byte (asserted: undoing the edits gives the source kernel back) and adds three edits: signature (+ `nw, tw, pout, cnt, eps`), zero rows for tokens without a local route (grid-stride, after the PDL wait), and the last-arriver epilogue after each task. |
| `g2p.py` | decode_layer variant (namespace `m11`, own module `g2p_ext`). Instances (tt, nt, d, ks, pdl): (1,1,4,4,1), (2,1,4,4,1), (2,1,4,4,0), (2,2,2,4,0). Python: `g2p_setup`, `g2p_warm`, `m11_gemm2_post`, `g2p_m11_run`, `g2p_install`, `g2p_ok`, `g2p_check`. |
| `g2p_moe.py` | MoE variant (namespace `m11f`, spliced into MoE's module; host `m11f_gemm2_post` launches with MoE's `m13_launch`). Instances (1,1,4,4,1), (2,1,4,4,1), (2,1,4,4,0). Python: `g2p_moe_source`, `m11f_counters`, `m11f_gemm2_post_py`. |
| `lab_g2p.py` | `sec_gemm2_post(lab, state, deadline)` for decode_layer's labs (after common / de6 / de14 / de16): attrs, correctness gate, routing-locked whole-layer A/B per class, prints `V14_FUSED`. |
| `emu_g2p.py`, `check_g2p_syntax.py`, `dryrun_g2p.py`, `mutate_g2p.py` | Local checks (CPU). |

## The algorithm

GEMM2's task = (active expert e, TT row tiles) over all of e's pairs. After a task: warp kpart 0 fences its y stores
and `atomicAdd(cnt[e])`; the group that completes `per_expert` tasks owns e (resets `cnt[e]`), then for each of e's
pairs adds 1 to `cnt[32 + token]`; the thread that reaches the token's local-route count resets it and queues the
token; the group then computes M13's post-norm for each queued token from the bf16 y rows (`ld.global.cg`), two passes
(sum of squares; weighted sum), and writes `out[token]`. Tokens with no local route get zero rows. Counters: int32
`[32 + M11_MAX_TOKENS]`, zero on entry, zero again on exit (no memset node). No grid barrier, no host sync, grid from
shapes only.

## Verified locally (Nov 4, ~17:45 UTC)

- `emu_g2p.py`: protocol port, 72 calls (3 consecutive calls per counter set; TT 1/2, KS 4/2/1; 1–32 tokens; random
  task orders): every token finished once and only after all its experts' tiles, zero rows, counters back to zero;
  post-norm math vs M13's two-kernel math at K = 4096: max rel err 4.7e-5 (62 of 2.46 M bf16 outputs differ by one
  rounding step, from the sum-of-squares order); 32 CUDA lines pinned in both variants.
- `check_g2p_syntax.py`: clang -fsyntax-only, host + sm_90a, every instance, both variants.
- `dryrun_g2p.py`: Python wiring and `sec_gemm2_post` control flow (per-class picks at the 0.3% bar; a planted wrong
  fused result installs nothing).
- `mutate_g2p.py`: 17/17 planted bugs caught.
- Not verified: GPU timing and registers (the lab logs `g2p_attrs` vs `m11_attrs`: the epilogue may raise registers
  above what 8 CTAs/SM allow).

Regenerate after M11 changes: `.venv/bin/python shared/glue/make_g2p.py`, then rerun the four checks.

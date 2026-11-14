# W1's dead-branch fix (AF49): what the docs say and the most likely mechanism

Read-only research for muse_attention_decode (TRACKER item 2) by muse_prefill_layer, Nov 4 ~19:50 UTC. No kernel was
built; attention/ was only read (`local/cuda_w1.py`, `cuda_w1n.py`, AF42–AF53).

## 1. The rules (PTX ISA 9.4, `docs/refs/ptx_isa_9.4.txt`, §9.7.17 and §9.7.17.7)

- `wgmma.fence` "establishes an ordering between prior accesses to any warpgroup registers and subsequent accesses to
  the same registers by a wgmma.mma_async instruction. Only the accumulator register and the input registers containing
  the fragments of matrix A require this ordering." It must be issued "before the first wgmma.mma_async operation" and
  "between a register access by a thread in the warpgroup and any wgmma.mma_async instruction that accesses the same
  registers ... Otherwise, the behavior is undefined."
- `wgmma.wait_group`: "Accessing the accumulator register or the input register containing the fragments of matrix A
  of a wgmma.mma_async instruction without first performing a wgmma.wait_group instruction that waits on a wgmma-group
  including that wgmma.mma_async instruction is undefined behavior."
- `.aligned`: all threads of the warpgroup must execute the same fence / mma / commit / wait (warpgroup-uniform
  control flow). W1's guards (`s0 + SKEYS > end ...`) and the `bar_wait` spin loops are CTA-uniform, so this holds.

W1's sealed form (L24: fence; 2× mma_async; commit; wait 0 in one asm) follows these rules **at the PTX level**, and
it still failed. So the violation, if any, is created **below PTX**, in ptxas.

## 2. What is unusual about W1's Q·K (and absent from every working variant)

| Kernel | Q·K A operand | Correct? |
|---|---|---|
| W1 (RS) | `qa[2][4]` built once before the stage loop, **live across every iteration** | no (err 1.4, every block) |
| W1-N (RS + dead branch after Q·K reading `qa`, scores, K tile) | same | yes |
| L21/L22 unit tests (RS) | A built right before the wgmma, no loop | yes |
| W2 (SS: Q and P from shared memory) | no register A | yes, no dead branch |
| W1's P·V (RS) | `pa` built from fresh scores right before the wgmma, every stage | — (part of W1; only fails together with Q·K per L25) |
| FA3 forward (FP8 and FP16) | **Q·K is SS** (Q via TMA in SMEM); only P·V is RS, with P produced just before it ([FA3 §3.3](https://arxiv.org/pdf/2407.08608); [Colfax: adding FP8 to FlashAttention](https://research.colfax-intl.com/adding-fp8-to-flashattention/)) | yes |
| CUTLASS RS mainloops | A fragments re-loaded (LDSM → registers) every k-block, fenced with `warpgroup_fence_operand` | yes |

The only correct-vs-wrong difference that tracks the data is **register-A operands that are loop-invariant and live
across the loop**. Every working RS use produces its A registers immediately before the wgmma.

## 3. Most likely mechanism (inferred; not provable without SASS)

wgmma's register-A operand is 4 registers per thread, which the hardware (QGMMA) reads asynchronously. If ptxas cannot
keep a long-lived value like `qa` in a register tuple suitable for QGMMA for the whole loop, it may insert register
**copies/moves into the operand registers at the SASS level**, next to the QGMMA. That would put them **after** the
`WARPGROUP.ARRIVE` that `wgmma.fence` lowers to, or reuse those physical registers while the async MMA is still
reading them. Either one breaks the fence rule in §1 below the level any PTX-side fix can reach. This is why sealing
the pair inside one asm (L24) didn't help: the copies are ptxas's, not nvcc's.

The dead branch reads `qa` (and the scores) with ordinary instructions in a separate basic block. That changes the
live ranges and the register-allocation constraints for `qa`, and on this compilation the allocator then keeps `qa`
in place (no inserted copies). This is "codegen-dependent", matching AF46 (garbage on every block, m off by up to 760,
consistent with a wrong Q·K, not a race on data).

Public sources found nothing more specific. The NVIDIA forum thread on wgmma serialization
([forums 340610](https://forums.developer.nvidia.com/t/ptxas-mysterious-warning-for-wgmma-mma-async-instruction-serialization/340610),
NVBUG 5431330) concerns ptxas being **too conservative** (C7514: "wgmma.mma_async instructions are serialized due to
non wgmma instructions reading accumulator registers ... between start and end of the pipeline stage") with
`wait_group 1` in loops. NVIDIA called it a diagnostic issue and recommended `wait_group 0`; it is not a wrong-code
report and uses SMEM operands. The ptxas pass write-up (nvopen-tools "gmma-pipeline") is no longer online (404).
CUTLASS 4.8.1's release notes mention a CTK bug giving wrong code for some packed FP8 instructions (MMA_K = 128,
packed FP8 subtraction): a different instruction family, but evidence that FP8 codegen bugs exist in this toolchain
generation.

## 4. Cheap ways to confirm or avoid it (for the owner to decide; nothing built here)

1. **Avoid (no confirmation needed):** don't hold register-A operands across loop iterations. Either keep Q in SMEM and
   use SS for Q·K, as FA3 does and as W2 already does (correct, AF51), or re-create the A fragments inside the loop
   right before each wgmma. W1-N's dead branch works on today's compilation but can stop working after any edit or
   ptxas update. Keep gating every W1-derived kernel on the lab error check and the warm-up self-check (AF49's rule).
2. **Discriminating variant (one lab section, if ever useful):** W1 without the dead branch, but with `qa`
   copied into fresh registers at the top of each iteration (e.g. `asm volatile("mov.b32 %0, %1;" : "=r"(qn[i]) : "r"(qa[s][i]));`
   for the 8 registers, then fence_a(qn) and pass `qn`). If this is correct, the loop-invariant register-A
   hypothesis is confirmed. If it is still wrong, the cause is elsewhere: the K tile path read by the dead branch, or
   scores live ranges.
3. **SASS evidence (not possible on the judge):** the lint rejects starting processes, so `cuobjdump -sass` can't run in
   a lab. With any machine with CUDA 13: compile W1 and W1-N for sm_90a, and look for `MOV`/`IMAD.MOV` writes to
   the QGMMA A registers between `WARPGROUP.ARRIVE` and `QGMMA`, or reuse of those registers before
   `WARPGROUP.DEPBAR`.

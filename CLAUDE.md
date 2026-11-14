# Open Kernel Challenge

Goal is to achieve the number 1 position for each track of the kernel challenge on https://openkernelchallenge.dev/#problems which are

- muse_bf16_proj 1.69× the #1 entry
- muse_attention_decode 3.34× the #1 entry
- muse_moe_w4a16 2.76× the #1 entry
- muse_decode_layer 2.55× the #1 entry
- muse_prefill_layer 2.25× the #1 entry

## How Kernels Are Judged

1. Submission: Write the kernel in the submission file. Start from the PyTorch starter and use whatever SGLang uses (Triton, CUTLASS, FlashInfer, DeepGEMM) or write your own CUDA.

2. Correctness: Your kernel is tested on random inputs it hasn’t seen. Its output has to match a reference implementation within the problem’s tolerance.

3. Timing: Your kernel and SGLang’s take turns on the same H200 with the same inputs, each run as a CUDA graph, the way SGLang decodes in production.

4. Ranking: The timed shapes come from SGLang serving agentic coding conversations, from one user up to 32. Your score is how much faster you are than SGLang.

## Claude Code Behaviour Guidelines

- Avoid ownership-dodging behaviour: if you encounter an issue, take responsibility for it and work towards a solution instead of passing it on to someone else. Don't say things like "not caused by my changes" or say that it's "a pre-existing issue". Instead, acknowledge the problem and take initiative to fix it. Also, don't give up with excuses like "known limitation" and don't mark it for "future work".
- Avoid premature stopping: if you encounter a problem, don't stop at the first obstacle. Instead, keep pushing forward and find a way to overcome it. Don't say things like "good stopping point" or "natural checkpoint". Instead, keep going until you have a complete solution.
- Do plan multi-step approaches before acting (plan which files to read and in what order, which tools to use, etc).
- Do recall and apply project-specific conventions from CLAUDE.md files.
- Do catch your own mistakes by applying reasoning loops and self-checks, and fix them before committing or asking for help.

### Use of tools

Adhere to the following guidelines when using tools:

- Always use a **Research-First approach**: Before using any tool, conduct thorough research to understand the context and requirements. This ensures that you use the most appropriate tool for the task at hand. Never use an Edit-First approach. You should prefer making surgical edits to the codebase instead of rewriting whole files or doing large, sweeping changes.
- Use **Reasoning Loops** very frequently. Don't be lazy and skip them. Reasoning loops are essential for ensuring the quality and accuracy of your work.

### Thinking Depth

When working on tasks that require complex problem-solving use Opus 5.5 and always apply the same **level of thinking depth** than the parent session.

When thinking is shallow, the model outputs to the cheapest action available. We don't want that. We don't mind consuming more tokens if it means a better output. So always apply the highest level of thinking depth.

Never reason from assumptions, always reason from the actual data. You need to read and understand the actual code, publication or documentation in order to make informed decisions. Don't rely on assumptions or guesses, as they can lead to mistakes and misunderstandings.

Several Claude sessions work in this folder at once, one per problem. All of them follow this file.
Read, in order: this file, `TRACKER.md` (shared), your own track file (`<session title>.md`),
`docs/SOURCES.md`, then your problem's `TASK.md` and `AGENTS.md`.

## The user's rules
1. **Plan before spending.** No `okc check`, `okc submit`, `okc serve ...` or GPU rental (Modal
   included) without the user's go-ahead. **Right before each launch, say what it uses and what's
   left** ("`okc check lab_e3b.py`: 1 check, leaving 35 of 40 today"), also for later steps of a
   sequence the user already approved.
   **Overnight standing go-ahead (user, Nov 4 ~05:40 UTC), valid until Nov 4 09:30 UTC (10:30 the
   user's local time):** each track session may run `okc check` and `okc submit` for its own problem
   without asking, and keeps iterating on its own (schedule its own wakeups; never sit idle waiting
   for the user). Limits for this window: keep 4 checks and 3 scored runs of today's quota in
   reserve; submit a scored run only when a check beat the board entry by ≥3%, from a clean file
   that passed presubmit; still state the cost in the track file before each launch. Everything
   else stays: no Modal/GPU rental, no `okc serve`, no `okc login`, no terms, rules 2–7 unchanged.
   (That window has ended.)
   **Scored-run threshold (user, Nov 4 ~13:00 UTC, replaces the 3% above):** propose a scored run
   when a check beats the board entry's speedup by **≥1.5%**. A clean candidate that beats the board
   by less than 1.5% is not submitted now: list it in `TRACKER.md` → "Final-round candidates" (file,
   check job, speedup, board speedup, gain %) so the user can run it before the track closes.
   **Checks go-ahead (user, Nov 5 ~02:35 UTC, for all sessions), valid until Nov 5 07:30 UTC:**
   "You may run okc check for this track without asking until 07:30 UTC; scored runs still need my
   yes." Rules 1 (state the cost before each launch) and 4 (local presubmit first) still apply;
   `okc submit` still needs the user's yes each time.
2. The user runs `okc login` themselves. Never accept rules or terms for anyone.
3. **Keep every option open** (CUDA, CUTLASS, warp specialization...). Record cost, risk and gain
   honestly; mark an option blocked only when it's impossible or against the rules, and say why.
4. **Check the docs and source of every tool at the judge's exact versions** before relying on an
   API (`docs/SOURCES.md`, local copies in `docs/refs/`), and run the local presubmit before any okc run.
   **The user's six reference links** (Inference Engineering, Hugging Face LLM optimization, vLLM,
   SGLang, TensorRT-LLM, GPU MODE; table at the top of `docs/SOURCES.md`) serve two purposes: before
   submitting, check the implementation follows what they document (semantics, numerics, graph
   rules, judging practice); while designing, mine them for optimization ideas. Note what you
   checked in your track file.
5. Don't submit other participants' work as is. One participant's public repository: design our own first and don't read
   its kernel files for now (user's decision, Nov 3).
6. **Spirit of the challenge:** real speedups only. NVFP4 weights stay NVFP4 (lossless repack in
   `prepare()` is fine), BF16 activations wherever SGLang keeps BF16, no work outside the timed
   call, never cache anything derived from the inputs.
7. Track every idea, finding, experiment and run in your own track file as it happens.

## Files and parallel sessions
- One session per problem, each with its own folder and tracker at the root:
  muse_moe_w4a16 → `moe/`, `muse_moe_w4a16.md` · muse_bf16_proj → `proj/`, `muse_bf16_proj.md` ·
  muse_attention_decode → `attention/`, `muse_attention_decode.md` · muse_decode_layer →
  `decode_layer/`, `muse_decode_layer.md` · muse_prefill_layer → `prefill_layer/`,
  `muse_prefill_layer.md`. The serving track's SGLang checkout is `serving/`.
- Your tracker uses prefixed IDs (MoE: M/T/E/MF; bf16_proj: P/PE/PF; prefill: R/RE/RF; ...).
- `TRACKER.md` is shared: deadlines, limits, the pre-submission checklist, cross-track findings
  (F-numbers). Edit it only with small exact-snippet edits, only your own rows, never by rewriting
  the whole file; re-read it before taking the next F number.
- Don't edit another track's folder or files. Work on a shared kernel for another track goes in
  `shared/<item>/` (see `TRACKER.md` → "Shared work queue"). Shared files (`TRACKER.md`, this file, `docs/`)
  change by small additions.
- Limits are per problem per UTC day (20 scored runs, 40 checks since Nov 3 21:00 UTC; check with
  `okc problems --json`), reset at 00:00 UTC. Kernel track closes **Nov 5, 07:59 UTC**.

## The method that works (MoE went 1.065× → 1.38× on Nov 3 with it)
1. **Recon check first.** `kernel_fn` = the baseline (or your last good version) byte for byte,
   while `prepare()` runs a lab that measures what you need on the judge's H200: environment,
   per-op times under CUDA graphs with a cold L2, bandwidth, compile times. Every passing check
   also saves a per-kernel profile in `profiles/<file>.<job>.txt`: read it.
2. **Labs: many experiments per check.** A builder (`moe/local/make_lab.py`) appends lab sections
   to the submission file and renames its `prepare` to `_submission_prepare`, so the timed
   `kernel_fn` is unchanged. Sections are time-boxed and wrapped (a failure costs one log line).
   `prepare()` gets 300 s in all, a run 15 minutes, and only the last 16,000 characters of output
   come back, so keep logs compact. A lab may pick configs from its own timings on synthetic
   inputs and print the config line to copy into the next file.
3. **Hybrid dispatch.** A new kernel serves only the shape classes where the lab proved it correct
   (against an FP32 reference, stricter than the judge's 1.5%) and at least 3% faster; everything
   else falls back to the last good version, so a check never goes backwards.
4. **Verify locally before spending** (there's no GPU on the Mac):
   - emulate the kernel's math in PyTorch with the kernel's own index formulas and compare with an
     FP32 reference at the real sizes (a layout bug can hide at small K);
   - for bit tricks, interpret the actual asm text parsed from the source, on every input value;
   - CUDA: `clang -x cuda -fsyntax-only` against stub headers (`moe/local/check_cuda_syntax.py`);
   - dry-run the lab's `prepare()` on the CPU with mocks;
   - mutation-test every check (plant a few bugs; each must be caught);
   - lint for the judge's static checks and CUDA-graph breakers (`moe/local/lint_submission.py`);
   - one command runs everything: `sh moe/local/presubmit.sh <files>` (other tracks: reuse or copy it).
5. **Scored runs:** a clean file (no lab code) with the lab's configs hardcoded, presubmit, state
   the cost, submit, then snapshot the exact file into `<track>/scored/` and record the job ID.
6. **Never risk an out-of-bounds access in a lab:** a CUDA fault is sticky and fails the whole run.

## Hard-won facts (details and sources in `TRACKER.md` findings)
- Triton 3.8: a jit function may read only globals that are modules, jit functions, `tl.*` or
  `x = tl.constexpr(...)` (F24). `inline_asm_elementwise` packs sub-32-bit values per register
  and doesn't promise which elements share one: write packed asm to be elementwise-correct (F25, F43).
- `load_inline`: pass `-gencode=arch=compute_90a,code=sm_90a` (any "arch" flag drops torch's plain
  sm_90); torch defines `__CUDA_NO_HALF_OPERATORS__` and friends, so use intrinsics; launch on
  `at::cuda::getCurrentCUDAStream()`; compile in `prepare()` (it counts toward the 300 s; 22 kernel
  instances took ~23 s) (F27, F28, F33).
- CUDA graphs: no host syncs, no allocation, branch or grid size that depends on input values;
  size grids for the worst case and exit early; launch every kernel once in `prepare()` (checklist 4).
- Numerics: PTX `mul.rn.bf16x2` keeps subnormals; plain Triton FP32 multiplies didn't flush (F30, F31).
- H200: ~4.49 TB/s streaming reads, ~0.8 µs per extra kernel in a graph (F20, F21). At full
  bandwidth there are only ~3.7 thread-instructions per 4-bit weight, so W4A16 kernels need weights
  pre-shuffled into MMA fragment order and decoded in registers; Triton can't do that, CUDA can
  (F45, F46; MoE MF1-MF2). At small batches the limit is warps and bytes in flight: split K across
  warps, more CTAs per SM, deeper prefetch (MoE E3b).
- Hopper features from NVIDIA's docs (F47): 228 KB shared memory per SM (227 per block; static
  `__shared__` capped at 48 KB, more via dynamic memory + carveout), TMA bulk copies (`cp.async.bulk`)
  tracked by `mbarrier` expect-tx so one thread keeps large copies in flight without registers
  (warp-specialized producer/consumer pipelines), clusters of up to 8 (16 opt-in) with distributed
  shared memory and TMA multicast, L2 access-policy windows (streaming vs persisting). No FP4
  tensor-core input on Hopper.

## Tools
- okc: `export PATH="$HOME/.okc/bin:$PATH"`. From the problem folder: `okc check <file> --json`,
  `okc submit <file> --json`, `okc status <job>`, `okc leaderboard <problem> --json`,
  `okc problems --json`. Run checks in the background with output to a scratch file.
- Local Python: `.venv/` (torch 2.15, CPU only) for every local check.
- Modal (free $30/month of H200 time) is parked (`TRACKER.md` G5): only with the user's own
  account and go-ahead, and a cost estimate first.

## Keep in mind

- Brainstorm: The platform to optimize for is H200, think deep about specific tricks related to this platform. For each track you are working on read the doc and also you are authorize to search more documentation online to understand specificities about the architecture.
  - https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
  - https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- I wanna achieve the number 1 position on the leaderboard so I need advanced improvements, not quick wins
- Think about best way to achieve the goal and what way is the best (Triton, CUTLASS, FlashInfer, DeepGEMM, pure CUDA or something else)
# Sources

Every session uses these. The six reference links below come first because they do two jobs:
**check that an implementation follows the documented behaviour, and give optimization ideas**.
Part 1 is what to **validate against before any `okc check` or `okc submit`**, at the judge's
exact versions. Part 2 lists further idea sources. Local copies live in `docs/refs/` (snapshots of
the files we read; re-fetch from the URL if in doubt). Add new sources here with what they're
good for, as a row, not a rewrite.

## The six reference links (the user's, Nov 3): use each one both ways

Before submitting a new kernel or patch, go through the rows that touch what you changed: does
the implementation match what the docs say (semantics, numerics, graph rules)? While designing,
read the same pages for techniques. Record what you checked in your track file. Prefer the exact
pages listed; they were read for this contest.

| Link | Check our implementation against | Ideas to take from it |
|---|---|---|
| **Inference Engineering**: https://inferenceengineering.tech (chapters `hardware/`, `techniques/`) | Roofline numbers: is the kernel memory- or compute-bound at each scored shape, and is the time we see near what bandwidth allows (compare with each shape's floor in `TASK.md`)? | Decode is bandwidth-bound: fewer bytes, more bytes in flight, fewer kernels; batching and quantized-weight techniques |
| **Hugging Face, LLM inference optimization**: https://huggingface.co/docs/transformers/main/llm_optims | Static shapes and graph capture (static KV cache, `torch.compile` modes), what each quantization scheme keeps in which precision | Static caches, compile-friendly shapes, fused attention, weight-only quantization trade-offs |
| **vLLM**: https://docs.vllm.ai (`design/cuda_graphs.html`, `features/batch_invariance/`, `api/vllm/model_executor/layers/fused_moe/fused_moe/`, `design/fused_moe_modular_kernel.html`, `design/fusions/`) | CUDA-graph rules (static shapes, no host syncs, buffers reused across replays); fused-MoE semantics (top-k weights applied once, renormalization, which slots are local under EP); batch shape changes numerics | Modular MoE kernels, the no-align path when tokens × top-k is tiny, fusions around attention and MoE, tuned MoE configs per batch size |
| **SGLang**: https://docs.sglang.ai (also docs.sglang.io: cookbook `autoregressive/Meta/MuseSparkOpen.md`, `advanced_features/{quantization,expert_parallelism,breakable_cuda_graph,deterministic_inference,speculative_decoding}.html`) and the pinned checkout `serving/sglang/` | What the baseline actually does for Muse Spark: NVFP4 via Marlin on H200, EP8, post-expert norm with (1 + w), FP32 router, CUDA-graph batch sizes; the code is the final word when docs and code differ | Where SGLang leaves time on the table (e.g. PyTorch post-norm below 32 tokens, Marlin tile sizing under EP), serving-track knobs |
| **TensorRT-LLM**: https://nvidia.github.io/TensorRT-LLM/ (`developer-guide/moe-perf-guide.html`, `developer-guide/perf-analysis.html`, tech blogs 01/04/08/14/20/24) and GitHub `tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md` | NVFP4 semantics (E2M1 values, E4M3 scale per 16, FP32 global scale); known bugs to avoid: PDL triggered too early, 4-bit interleave wrong past one tile (PRs #14537, #14775; checklist 8-9) | Hopper W4A16 NVFP4 MoE = vendored Marlin there too, so the headroom is our own pipeline; MoE as dense GEMM at large batch; PDL; CUDA-graph batch-size tuning |
| **GPU MODE**: https://github.com/gpu-mode (`reference-kernels` evals, `kernelbot`, `kernelguard`, `lectures`, `kernelboard`) | How contest judges time kernels (graphs, cold L2, fresh inputs, output fingerprints) and what they reject as reward hacking (KernelGuard): run our files past those rules | Lectures 36 (CUTLASS + FA3 on Hopper), 34 (low-bit Triton, W4A16, split-K), 7 (int4 kernels), 23 (tensor cores), 86/57 (CuTe), 14 (Triton); NVFP4 contest write-ups in `docs/refs/contest-writeups/` |

What we've already taken from them is in `TRACKER.md`: R-G (GPU MODE), R-S (vLLM + SGLang docs),
R-T (TensorRT-LLM), and the findings and checklist items that cite them.

## The judge's environment (verify before relying on any API)

From E1 (`TRACKER.md` F22): torch **2.14.1+cu130**, CUDA **13.0**, Triton **3.8.0**, flashinfer
0.7.0.post1, tilelang 0.1.12, nvidia-cutlass-dsl 4.8.0 (CuTe DSL), cuda-python 13.4.1, deep_gemm
importable, Helion **not** installed, `CUDA_HOME=/usr/local/cuda`, H200 (132 SMs, 60 MB L2, 1980 MHz,
driver 580.159.03). SGLang is the pinned checkout in `serving/sglang/` (commit in each `baseline.py`
docstring). Public image of the same stack: `lmsysorg/sglang:dev-muse-spark-open`.

## Part 1: validation

### The contest and each problem (the rules that decide pass/fail)
| Source | Use it for |
|---|---|
| https://openkernelchallenge.dev/llms.txt | The contest entry point for agents |
| https://openkernelchallenge.dev/ and https://openkernelchallenge.dev/rules | Rules, kernel and serving tracks, what gets a run disqualified |
| `<problem>/TASK.md`, `<problem>/AGENTS.md` | Shapes, scored loads, tolerances, time limits (run 15 min, `prepare()` 300 s), validation and "spirit" |
| `<problem>/reference.py`, `checks.py`, `nvfp4.py` | Exactly what's checked (never import `reference`/`checks` from a submission) |
| `<problem>/baseline.py` | SGLang's timed path, call for call |
| `okc help <command>`, `okc problems --json` | CLI options; today's limits and what's left |

### Judge-version sources (local copies in `docs/refs/`)
| Source | Local copy | What we checked in it |
|---|---|---|
| Triton v3.8.0 `python/triton/language/core.py` — https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/core.py | `docs/refs/triton-v3.8.0/python/triton/language/core.py` | `reshape`/`broadcast_to` take tuples, `inline_asm_elementwise` semantics ("which set of inputs a block receives is unspecified"), `dot` signature |
| Triton v3.8.0 `python/triton/compiler/code_generator.py` | `docs/refs/triton-v3.8.0/python/triton/compiler/code_generator.py` | Global-variable rule (F24): only modules, jit functions, `tl.*`, dtypes, `x = tl.constexpr(...)`; dynamic `range` bounds |
| Triton v3.8.0 `python/triton/runtime/jit.py` | `docs/refs/triton-v3.8.0/python/triton/runtime/jit.py` | `do_not_specialize` takes names; a launch returns the compiled kernel (`.asm`, `n_regs`, `n_spills`) |
| Triton v3.8.0 `third_party/nvidia/backend/compiler.py` | `docs/refs/triton-v3.8.0/third_party/nvidia/backend/compiler.py` | Builds for `sm_90a`; `enable_reflect_ftz=True` (F26, F31) |
| Triton v3.8.0 `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` | `docs/refs/triton-v3.8.0/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` | Inline-asm operand packing and numbering (F25, F43) |
| Triton v3.8.0 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp` (path from its includes) | same tree | NVIDIA elementwise lowering |
| PyTorch v2.14.1 `torch/utils/cpp_extension.py` — https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/cpp_extension.py | `docs/refs/pytorch-v2.14.1/torch/utils/cpp_extension.py` | `load_inline` flags: any "arch" flag disables torch's (pass `-gencode=arch=compute_90a,code=sm_90a`), the `-D__CUDA_NO_*` defines, C++20, `use_pch`, `no_implicit_headers` (F27, F33) |
| PyTorch v2.14.1 `torch/cuda/graphs.py` | `docs/refs/pytorch-v2.14.1/torch/cuda/graphs.py` | Capture mode "global", private pool (F29) |
| PTX ISA 9.4 — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | `docs/refs/ptx_isa_9.4.txt` | §9.7.16.5.8 m16n8k16 fragment layouts (A/B/C), `cvt` (e4m3x2 → f16x2 is sm_89+; → bf16x2 is PTX 9.2), `prmt`, `mul.rn.bf16x2` (sm_90, keeps subnormals), `bar.sync` |
| CUDA programming guide, PDL — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html | – | `griddepcontrol` rules (checklist 8) |
| **Hopper tuning guide** (user's pick, CLAUDE.md) — https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | – | Shared memory and carveout sizes, registers and occupancy limits, TMA, clusters, L2 (`TRACKER.md` F47) |
| **Hopper architecture in depth** (user's pick) — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | – | TMA, async transaction barriers, clusters and DSMEM, tensor-core rates; ideas for warp-specialized pipelines (F47) |
| CUDA programming guide: L2 cache control — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html; async copies (TMA, `cp.async.bulk`) — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html; advanced kernel programming (async proxy, fences) — https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html | – | Exact API and alignment rules for `accessPolicyWindow`, bulk copies with `mbarrier` expect-tx, proxy fences (F47) |
| CUDA driver API, tensor maps — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html | – | TMA alignment and box limits (checklist 7) |
| SGLang checkout `serving/sglang/` (pinned) | in repo | Baseline kernels and their exact semantics (e.g. `moe_align_block_size` puts non-local slots in `-1` blocks; the post-norm falls back to PyTorch below 32 tokens) |
| GPU MODE KernelGuard / KernelBot — https://github.com/gpu-mode/kernelguard, https://github.com/gpu-mode/kernelbot | – | What contest judges reject as reward hacking (R-G); our lint borrows from it |

## Part 2: optimization ideas

### Other links the user shared (Nov 3)
| Source | Good for |
|---|---|
| https://openkernelchallenge.dev/llms.txt and https://openkernelchallenge.dev/ | The contest's own instructions (also listed in Part 1) |
| From the Gemini brainstorm: https://another participant's public repository | A public 2.33× CUDA MoE entry. **User's decision (Nov 3): design our own first, don't read its kernel files; compare and borrow ideas later**, keeping its Apache-2.0 notice if anything is reused (M9) |
| From the Gemini brainstorm: https://x.com/shiyi_c98/status/2117240310736515510 | Berkeley's EAT (Evolving Agent Teams) reportedly reached 2.44× on the decode layer with iterative agent search over configs; supports "many variants per check" |

### Pages the research subagents read (Nov 3), by topic
- **TensorRT-LLM MoE:** `tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md` and `.../marlin/base.py` on GitHub (Hopper W4A16 NVFP4 runs a vendored Marlin); developer-guide `moe-perf-guide.html`, `perf-analysis.html`; tech blogs 01, 04, 08, 14, 20, 24; `features/torch_compile_and_piecewise_cuda_graph.html`, `features/speculative-decoding.html`; PRs #14537 and #14775 (PDL trigger bug, interleave bug; checklist 8-9).
- **FlashInfer:** https://docs.flashinfer.ai/generated/flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm.html (SM90 mixed-input weight interleave), https://docs.flashinfer.ai/api/comm.html.
- **SGLang:** cookbook `autoregressive/Meta/MuseSparkOpen.md`, `advanced_features/{quantization,expert_parallelism,speculative_decoding,adaptive_speculative_decoding,breakable_cuda_graph,hicache_design,hicache_best_practices,deterministic_inference}.html`, `references/faq.html`, `benchmark/kernels/fused_moe_triton/README.md`.
- **vLLM:** `configuration/env_vars.html`, `api/vllm/model_executor/layers/fused_moe/fused_moe/`, `design/fused_moe_modular_kernel.html`, `design/fusions/`, `design/cuda_graphs.html`, `features/batch_invariance/`, `serving/expert_parallel_deployment.html`, speculative-decoding pages; recipe https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B.
- **LMSYS blogs:** DFlash v2 speculative decoding (2026-06-15), unified radix cache (2026-08-11).

- **Hopper W4A16 GEMM design (prefill session, Nov 3 night):** Machete — https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel and vLLM PR #7174 / `csrc/quantization/machete` (swap-AB RS wgmma, weight pre-shuffle, stream-K); LiquidGEMM arXiv 2509.01229 (dequant in each compute WG's registers beats a separate dequant WG); QServe arXiv 2405.04532; FLUTE arXiv 2407.10960; CUTLASS examples 55 (mixed dtype, group-size rule in its README), 63 (L2 weight prefetch under PDL), 69 (grouped mixed), 113 (Hopper grouped GEMM with fused gated activation), `cutlass/detail/collective/mixed_input_utils.hpp` (`_e2m1_to_bf16_x8`, `dequantize_A_kblock`); FlashInfer v0.7.0.post1 `docs/design_docs/cute_dsl_moe_sm90.md` (SM90 BF16 MoE: cp.async gather producer + wgmma consumers); DeepGEMM `m_grouped_bf16_gemm_nt_contiguous` (≥ v2.1.0) and `scheduler/gemm.cuh`; GPU MODE winner's worklog https://github.com/gau-nernst/gpu-mode-kernels (grouped GEMM: persistent static LPT schedule, swapped operands).
- **Attention on Hopper (prefill session):** sgl-attn @ f89bc23 (the FA3 SGLang pins in `serving/sglang/python/sglang/kernels/aot/CMakeLists.txt:83-89`): `hopper/tile_size.h`, `heuristics.h`, `flash_api.cpp`, `paged_kv.h`; FA3 paper arXiv 2407.08608 (MUFU 3.9 TFLOPS vs 989 matmul); FA4 paper arXiv 2603.05451 and https://tridao.me/blog/2026/flash4/ (polynomial exp2, lazy rescale; SGLang vendors it: `python/sglang/kernels/ops/attention/flash_attn/cute/utils.py`).
- **Cross-op overlap:** NanoFlow arXiv 2408.12757; Hazy Research megakernel https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main; TRT-LLM tech blog 1 (PDL +3%).

### Code already in the judge image or our checkout (fastest to learn from)
| Path | What it shows |
|---|---|
| `serving/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/` (`marlin_template.h`, `moe_wna16_marlin.cuh`) and `.../gemm/marlin/dequant.h` | Marlin's W4A16 MoE: fragment-ordered weights, FP4 → BF16 bit trick, cp.async pipeline (F4, F41) |
| `serving/sglang/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py` | SGLang's Triton int4 MoE (`fused_moe_kernel_gptq_awq`); SGLang prefers Marlin when it can |
| `serving/sglang/python/sglang/srt/rocm_muse_spark/muse_moe_triton.py` | A Triton MoE for this exact model on ROCm: fused SwiGLU, per-slot output, one-program align kernel (F42) |
| `serving/sglang/python/sglang/kernels/ops/muse_spark_v12/` | SGLang's Muse Spark fused kernels (post-expert norm, etc.) |
| CUTLASS/CuTe headers in the image: `flashinfer/data/cutlass/include`, `deep_gemm/include`, `tilelang/3rdparty/cutlass/include`, `nvidia/mathdx/external/cutlass/include` under `/opt/sglang/lib/python3.12/site-packages/` | For `load_inline` CUDA (F39) |
| `docs/refs/contest-writeups/` | GPU MODE NVFP4 contest write-ups (GEMV, GEMM, gated dual GEMM, grouped GEMM; Blackwell, techniques only) and an AMD FP8 GEMM report |
| https://github.com/gau-nernst/gpu-mode-kernels | Worklogs of NVFP4 contest kernels (Blackwell; techniques only) |

### Our own measurements
`TRACKER.md` findings F1-F46 (cross-track facts: judge env, bandwidth 4.49 TB/s, graph-kernel cost
~0.8 µs, Triton/torch/PTX rules, the W4A16 instruction budget) and each session's tracker
(`muse_*.md`, e.g. MF1-MF2 for MoE).

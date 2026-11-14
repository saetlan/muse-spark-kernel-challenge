# Muse Spark Open Kernel Challenge: a 48-hour AI-assisted experiment

This repository holds my work on the kernel track of the
[Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/): five GPU kernel problems for
Muse Spark Open on one H200, each scored by speedup over SGLang's own kernel.

## What this was

**An experiment.** I wanted to see how far I could get using only public documentation (NVIDIA's
Hopper and PTX docs, the Triton, PyTorch, SGLang, FlashInfer and CUTLASS sources and docs) and AI
assistance (Claude Code, one session per problem), **without using any public repository from the
competition**. The kernels were designed in this project; no code was taken from other participants.
Open-source library code and some outside ideas were used, and they are listed in "Third-party code
and ideas" below.

**I joined 2 days before the end.** The kernel track closed on Nov 5, 2026 at 07:59 UTC and I started
on Nov 3, so every result below was obtained in about 48 hours, within the daily check and scored-run
limits.

Two precisions on the "no public repository" claim:

- At my request, one session (MoE) read the public README and metadata of one other participant's
  repository on Nov 3. No kernel file from it was read or used, and the project rules (`CLAUDE.md`,
  rule 5) forbade reading competitors' kernels.
- On Nov 4 I pasted a top entry's public write-up into the MoE session. Two ideas from it were
  recorded (T28, T29). T29 produced two experimental kernels that lost and were not scored; T28 was
  never built. Also, a post about another team's decode-layer result was read as brainstorm input by
  the decode_layer session; nothing from it was built.

Each of the five sessions was asked afterwards, from its own notes, whether it read, copied or adapted
another participant's code. All answered no. Three of them (attention, decode_layer, bf16_proj) could
not vouch for the earliest part of their work beyond their notes.

## Results

Final standings I recorded (Nov 5, 2026, 13:10 UTC, after the track closed). Speedup is over SGLang;
rank is among all ranked entries on that problem's board.

| Problem | Our speedup | Our rank | #1 speedup |
|---|---|---|---|
| `muse_bf16_proj` | 1.3771× | 556 of 1746 | 1.7277× |
| `muse_attention_decode` | 1.8314× | 741 of 1379 | 3.3389× |
| `muse_moe_w4a16` | 1.7027× | 887 of 1425 | 2.8438× |
| `muse_decode_layer` | 1.6199× | 772 of 1299 | 2.6144× |
| `muse_prefill_layer` | 1.3783× | 616 of 898 | 2.1471× |

None of the five reached first place.

## Layout

- `proj/`, `attention/`, `moe/`, `decode_layer/`, `prefill_layer/`: one folder per problem, with the
  submissions (`scored/` holds the exact files that were judged), lab files that ran experiments on
  the judge's H200, and local verification scripts.
- `muse_<problem>.md`: the notes of each problem's session: every idea, experiment and run, including
  the failures.
- `TRACKER.md`: findings shared across problems (H200 measurements, judge behaviour).
- `CLAUDE.md`: the working rules the AI sessions followed.
- `.claude/skills/okc-dashboard-refresh/`: the script that built a standings dashboard from `okc`.

## Method

Recon checks first (a lab that measures the judge's H200), then many experiments per check, a hybrid
dispatch so a check never goes backwards, local verification before every spend, and scored runs only
from clean files. `CLAUDE.md` has the details.

## Third-party code and ideas

All of this is public documentation or open-source code, recorded in each `muse_<problem>.md`:

- **Code reused or adapted:** SGLang's own kernels (the baseline; for example its cluster
  norm-residual kernel is called as is in the decode-layer submissions). The bf16_proj session forked
  DeepGEMM's sm90 BF16 GEMM (MIT licence, DeepSeek) as DG12, in `proj/dg/dg12.py` and the lab files
  that embed it. A search of the scored files found no DeepGEMM kernel text in them.
- **Documentation and papers:** NVIDIA's Hopper tuning guide and architecture blog, the CUDA guide,
  the PTX ISA, the CUTLASS, FlashInfer, vLLM, TensorRT-LLM and SGLang sources and docs, the
  FlashAttention-3 paper, and several arXiv papers on FP8 and lossless compression.
- **Ideas from another contest:** GPU MODE's NVFP4 contest write-ups (Blackwell kernels, techniques
  only) and a winner's worklog (scheduling ideas), plus a Hazy Research blog post.
- **AI brainstorms:** a few ideas came from a Gemini brainstorm (MoE T1, T3 and T7).
- **Between our own sessions:** MoE kernels reused in the decode-layer submissions, and a W4A16
  design shared between the MoE and prefill sessions.

## Licence

My submissions are licensed under Apache 2.0, as the challenge rules require. The third-party code
listed above keeps its own licences. SGLang, PyTorch, Triton and NVIDIA's documents are not copied
into this repository.

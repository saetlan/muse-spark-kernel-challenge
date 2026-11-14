"""Build submission.py (v2) from baseline.py: SGLang's layer + v2/gmm.py, with exactly two edits.

  prepare():   `_gmm_prepare(state, weights)` right before `del freqs`
  kernel_fn(): the experts call goes through _moe(..., tokens), which picks the grouped path from
               GMM_MIN_TOKENS tokens when prepare()'s GPU check passed, else SGLang's _experts.
v2/gmm.py goes right before `def kernel_fn`. check_v2_vs_baseline.py rebuilds this and compares ASTs.

Run: .venv/bin/python prefill_layer/local/make_v2.py
"""

from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
HEADER = '''# okc: problem=muse_prefill_layer
"""v2: SGLang's decoder layer (baseline.py, commit a55ab7d5) with the routed experts as BF16 grouped
GEMMs from 4096 tokens up.

* From 4096 tokens: every local expert with a route has its NVFP4 weights decoded into a BF16 scratch
  each call (exact: E2M1 x E4M3 fits BF16; the FP32 global scale is applied to the GEMM outputs),
  the local routes are sorted by expert on the GPU, and torch._grouped_mm runs both expert GEMMs.
  SiLU-and-mul and the post-expert norm weighted sum are Triton kernels over the sorted rows.
  At 16K tokens Marlin ran ~230 TFLOP/s and the grouped GEMM ~600 (RE1).
* Below 4096 tokens, and if prepare()'s GPU check against Marlin fails, SGLang's Marlin path.

Everything else is SGLang's code path, call for call.

'''
PREP_OLD = "    del freqs\n"
PREP_NEW = "    _gmm_prepare(state, weights)\n    del freqs\n"
CALL_OLD = '    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n'
CALL_NEW = '    expert_output = _moe(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state, tokens)\n'
ANCHOR = "\ndef kernel_fn(data, state):"


def build_text():
    src = (DIR / "baseline.py").read_text()
    assert src.startswith('"""SGLang')
    src = HEADER + src[3:]
    for old, new in ((PREP_OLD, PREP_NEW), (CALL_OLD, CALL_NEW)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    assert src.count(ANCHOR) == 1
    frag = (DIR / "v2" / "gmm.py").read_text().strip("\n")
    return src.replace(ANCHOR, "\n" + frag + "\n\n" + ANCHOR, 1)


if __name__ == "__main__":
    (DIR / "submission.py").write_text(build_text())
    print("wrote prefill_layer/submission.py (v2)")

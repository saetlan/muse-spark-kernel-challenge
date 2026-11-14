"""Build submission.py (v5) from baseline.py: SGLang's layer + v2/gmm.py + v5/r14b.py, with exactly two edits.

  prepare():   `_gmm_prepare(state, weights)` and `_r14_prepare(state, weights)` right before `del freqs`
  kernel_fn(): the experts call goes through _moe_v4(..., tokens): R14 for the token class if configured and
               checked on the GPU, else v2's grouped path from 8192 tokens, else SGLang's Marlin.
Both fragments go right before `def kernel_fn`. check_v5_vs_baseline.py rebuilds this and compares ASTs.

Run: .venv/bin/python prefill_layer/local/make_v5.py [r14_configs_text]
"""

import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
HEADER = '''# okc: problem=muse_prefill_layer
"""v5: SGLang's decoder layer (baseline.py, commit a55ab7d5) with our own routed-expert paths.

* R14 (per token class, where a lab measured it correct and faster): Hopper W4A16 NVFP4 grouped GEMMs in
  CUDA. Weights are wgmma's register A operand, decoded from NVFP4 in registers (weights repacked
  losslessly in prepare into fragment order, scales as exact bf16); tokens are the shared-memory B operand
  (128B swizzle, written pre-swizzled by the gather); a producer warp streams each stage with two bulk
  async copies into an mbarrier ring; two consumer warpgroups decode and run wgmma; SwiGLU on the
  accumulators; SGLang's post-expert norm weighted sum over the per-route rows.
* v2 from 8192 tokens otherwise: touched experts decoded exactly into a BF16 scratch each call,
  torch._grouped_mm for both expert GEMMs.
* Otherwise, and whenever prepare()'s GPU check against Marlin fails, SGLang's Marlin path.

Everything else is SGLang's code path, call for call.

'''
PREP_OLD = "    del freqs\n"
PREP_NEW = "    _gmm_prepare(state, weights)\n    _r14_prepare(state, weights)\n    del freqs\n"
CALL_OLD = '    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n'
CALL_NEW = '    expert_output = _moe_v4(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state, tokens)\n'
ANCHOR = "\ndef kernel_fn(data, state):"
CONFIG_LINE = "R14_CONFIGS = {}\n"


def build_text(configs=None):
    src = (DIR / "baseline.py").read_text()
    assert src.startswith('"""SGLang')
    src = HEADER + src[3:]
    for old, new in ((PREP_OLD, PREP_NEW), (CALL_OLD, CALL_NEW)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    assert src.count(ANCHOR) == 1
    gmm = (DIR / "v2" / "gmm.py").read_text().strip("\n")
    r14 = (DIR / "v5" / "r14b.py").read_text().strip("\n")
    if configs is not None:
        assert r14.count(CONFIG_LINE) == 1
        r14 = r14.replace(CONFIG_LINE, f"R14_CONFIGS = {configs}\n")
    return src.replace(ANCHOR, "\n" + gmm + "\n\n" + r14 + "\n\n" + ANCHOR, 1)


if __name__ == "__main__":
    configs = Path(sys.argv[1]).read_text().strip() if len(sys.argv) > 1 else None
    (DIR / "submission.py").write_text(build_text(configs))
    print("wrote prefill_layer/submission.py (v5)")

"""Build submission.py (v3) from baseline.py: SGLang's layer + v2/gmm.py + v3/r3.py, with exactly two edits.

  prepare():   `_gmm_prepare(state, weights)` and `_r3_prepare(state, weights)` right before `del freqs`
  kernel_fn(): the experts call goes through _moe_v3(..., tokens): an R3 config for the token class if
               one passed prepare()'s check, else v2's grouped path from GMM_FROM_TOKENS, else Marlin.
Both fragments go right before `def kernel_fn`. check_v3_vs_baseline.py rebuilds this and compares ASTs.

Run: .venv/bin/python prefill_layer/local/make_v3.py [r3_configs_file]
"""

import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
HEADER = '''# okc: problem=muse_prefill_layer
"""v3: SGLang's decoder layer (baseline.py, commit a55ab7d5) with our own routed-expert paths.

* R3 (per token class, where a lab measured it correct and faster): fused NVFP4 grouped GEMMs in Triton.
  Local routes sorted by expert on the GPU, tiles of BT rows per expert, NVFP4 decoded in registers
  per BK step (Marlin's bit trick, weights nibble-permuted losslessly in prepare), SwiGLU in GEMM1's
  epilogue, SGLang's post-expert norm weighted sum over the sorted rows.
* v2 from 8192 tokens otherwise: touched experts decoded exactly into a BF16 scratch each call,
  torch._grouped_mm for both expert GEMMs.
* Otherwise, and whenever prepare()'s GPU check against Marlin fails, SGLang's Marlin path.

Everything else is SGLang's code path, call for call.

'''
PREP_OLD = "    del freqs\n"
PREP_NEW = "    _gmm_prepare(state, weights)\n    _r3_prepare(state, weights)\n    del freqs\n"
CALL_OLD = '    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n'
CALL_NEW = '    expert_output = _moe_v3(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state, tokens)\n'
ANCHOR = "\ndef kernel_fn(data, state):"
CONFIG_LINE = "R3_CONFIGS = {}\n"


def build_text(configs=None):
    src = (DIR / "baseline.py").read_text()
    assert src.startswith('"""SGLang')
    src = HEADER + src[3:]
    for old, new in ((PREP_OLD, PREP_NEW), (CALL_OLD, CALL_NEW)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    assert src.count(ANCHOR) == 1
    gmm = (DIR / "v2" / "gmm.py").read_text().strip("\n")
    r3 = (DIR / "v3" / "r3.py").read_text().strip("\n")
    if configs is not None:
        assert r3.count(CONFIG_LINE) == 1
        r3 = r3.replace(CONFIG_LINE, f"R3_CONFIGS = {configs}\n")
    return src.replace(ANCHOR, "\n" + gmm + "\n\n" + r3 + "\n\n" + ANCHOR, 1)


if __name__ == "__main__":
    configs = Path(sys.argv[1]).read_text().strip() if len(sys.argv) > 1 else None
    (DIR / "submission.py").write_text(build_text(configs))
    print("wrote prefill_layer/submission.py (v3)")

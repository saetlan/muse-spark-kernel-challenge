# v11 = v10 with the attention session's v8 (parts/attn8.py, from attention/scored/v8_submission.py: v6 + UD3 (8,4,3)
# on sliding decode 16, V5 (1,4,1) on full verify 1, FA3 with num_splits 4 + pack_gqa behind a Triton FP8 prologue
# on full verify 16; checks 1.471x / 1.472x / 1.481x on muse_attention_decode) and the bf16_proj session's v7a
# router configs at m = 32 / 128 / 256 (proj/scored/submission_v7a.py lines 65-67, PE10: wider Triton tiles, +3-9% per
# shape on the judge; v7a's only change to v6's timed path). Both are module-level swaps: the v9 glue's prepare
# self-checks every pj_CHOICE entry against SGLang (< 1e-2) and the attention segment on V6_ATT_CASES (< 0.2 vs the FA3
# path, cache bytes), now including full verify 16 (v8's fa3t route).

V9_ATT_NAME = "attention v8"
V9_PJ_NAME = "bf16_proj v6 + v7a router"
V6_ATT_CASES = V6_ATT_CASES + ((16, 8, False),)
V11_ROUTER = {
    ('router', 32): {'kind': 'gemm', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 4},
    ('router', 128): {'kind': 'gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'mode': 1, 'warps': 8, 'stages': 3},
    ('router', 256): {'kind': 'gemm', 'BM': 256, 'BN': 64, 'BK': 64, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 3},
}
pj_CHOICE.update(V11_ROUTER)


def v11_setup(state, weights):
    """Nothing to build: report what v11 swapped in (the checks ran in the glue's prepare)."""
    st = state.get("pj_state")
    print(f"[v11] attention v8 {'on' if state.get('v4_att') else 'OFF'}; v7a router configs at m "
          f"{'/'.join(str(m) for _l, m in sorted(V11_ROUTER))} "
          f"{'checked with the projections' if st is not None else 'unused (projections OFF)'}")

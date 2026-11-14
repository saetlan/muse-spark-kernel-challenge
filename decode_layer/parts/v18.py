# v18 = v17 with the attention session's scored v9 (parts/attn9.py, from attention/scored/v9_submission.py, md5
# ab2ce732, scored 1.5655x on muse_attention_decode, job 2e994b0659da4d84): v8 + W1-N (wgmma FP8 verify, 1 CTA per SM,
# 4 stages) on sliding verify 16 / 32 and full verify 1 / 16 / 32, built in its main CUDA module; FA3T and V5 routes
# gone; decode routes unchanged. The attention self-check in prepare (< 0.2 vs the FA3 path, cache bytes) gains full
# verify 32 (W1-N's Fv32 route) so every W1-N class is exercised before any capture.

V9_ATT_NAME = "attention v9"
V6_ATT_CASES = V6_ATT_CASES + ((32, 8, False),)


def v18_setup(state, weights):
    print(f"[v18] attention v9 {'on' if state.get('v4_att') else 'OFF'} (W1-N verify; {len(V6_ATT_CASES)} self-check cases)")

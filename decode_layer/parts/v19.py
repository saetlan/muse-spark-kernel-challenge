# v19 = v18 with the attention session's scored v11 (parts/attn11.py, from attention/scored/v11_submission.py, md5
# 0ad5e5e2, scored 1.6102x on muse_attention_decode, job c98e26a4329c4fab) in place of v9: v9's W1-N stays on full
# verify 16 / 32 and sliding verify 32; v10's W2 serves full verify 1; v11's UW2 (one launch per sliding verify layer:
# prologue, W2 attend and cluster merge) serves sliding verify 1 / 16. v18's self-check cases (15, incl. full verify 32)
# already cover every one of these classes.

V9_ATT_NAME = "attention v11"


def v19_setup(state, weights):
    print(f"[v19] attention v11 {'on' if state.get('v4_att') else 'OFF'} (W1-N / W2 / UW2 verify; "
          f"{len(V6_ATT_CASES)} self-check cases)")

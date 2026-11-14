# v20 = v18 with the attention session's scored v14 (parts/attn14.py, from attention/scored/v14_submission.py, md5
# 2d153ba3, scored 1.722x on muse_attention_decode, job 01bc09ba1a3e437b) in place of v9. v14's final routes: D3
# (dec4) on full decode 1 / 16 / 32 as before; UD4 (UD3 with the row-owner finalize and the new token's K / V loaded at
# entry) on sliding decode 1 / 16 / 32; UW2 on sliding verify 1 / 16; W1-N on sliding verify 32; W2 on full verify 1;
# W3b (4 consumer warpgroups, P.V into O in place, whole-row copier) on full verify 16 / 32. Same entry points and data
# keys as v9 / v11; a route that fails v14's own warm-up check falls back to its Triton / PDL chain. The embed empties
# the CUDA sources of the kinds v14 no longer routes (make_attn.py TRIM: 220 KB, F61's size limit); the assert below
# stops the import if a routed kind lost its source. v18's 15 self-check cases cover every routed class (W3b at batch
# 2 and 32, UD4 at 1 / 16 / 32) against SGLang's attention before any capture.

V9_ATT_NAME = "attention v14"
V20_ROUTED_KINDS = sorted({route[0] for route in att_CUDA_ROUTE.values()})
assert all(att__KINDS[k][0] for k in V20_ROUTED_KINDS), [k for k in V20_ROUTED_KINDS if not att__KINDS[k][0]]


def v18_setup(state, weights):  # noqa: F811 (v18's line named attention v9; v20_setup reports instead)
    return None


def v20_setup(state, weights):
    routes = {f"{'F' if full else 'S'}{'v' if verify else 'd'}{b}": r[0]
              for (full, verify, b), r in sorted(att_CUDA_ROUTE.items())}
    print(f"[v20] attention v14 {'on' if state.get('v4_att') else 'OFF'} ({len(V6_ATT_CASES)} self-check cases); "
          f"CUDA module {'built' if 'mod' in att__CUDA else 'MISSING'}, routes after warm-up {routes}")

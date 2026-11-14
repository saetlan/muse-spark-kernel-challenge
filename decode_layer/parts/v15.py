# v15 = v13 with the expert stage's two small glue kernels moved off the main stream's critical path. Today, after the
# pooler, the main stream waits for the side stream (router + routing), then runs the local-ids kernel, then the
# pre-expert RMSNorm, then GEMM1. With state["v15_side"] the local ids run on the side stream right after the routing
# (they only need topk_ids and local_start), and the pre-expert norm runs on the main stream before the join (it only
# needs the pooler's output), so both overlap the other stream's work. Same kernels, same inputs: bit-identical.
# On only where a lab measured it faster (V15_SIDE; the parts/v14.py slot is the prefill session's GEMM2 + post-norm).

V15_SIDE = False  # set from a lab's printed V15_SIDE


def v15_side_ids(topk_ids, local_start, state):
    """Called inside the side-stream block: the local ids there when v15 is on (else None)."""
    return v3_local_ids(topk_ids, local_start, state) if state.get("v15_side") else None


def v15_pre_norm(expert_input, state):
    """Called on the main stream before the join: the pre-expert norm there when v15 is on (else None)."""
    return v3_pre_expert_norm(expert_input, state) if state.get("v15_side") else None


def v15_experts(expert_input, normed, topk_weights, local_ids, state):
    """v3_experts, without its pre-expert norm when v15 already ran it (normed is not None)."""
    if normed is None:
        return v3_experts(expert_input, topk_weights, local_ids, state)
    cfg = m11_config_for(state, normed.shape[0]) if "m11" in state else None
    if cfg is None:
        return _experts(normed, topk_weights, local_ids, state)
    return m11_run(normed, local_ids.to(torch.int32), topk_weights.to(torch.float32), state, cfg)


def v15_setup(state, weights):
    state["v15_side"] = bool(V15_SIDE)
    print(f"[v15] local ids on the side stream, pre-expert norm before the join: {'on' if V15_SIDE else 'off'}")

# The PDL GEMM2 + CUDA post-norm option (parts/v10.py) also at 9-16 tokens: that class's GEMM2 config (2,1,4,4) is
# one of the M11P instances already built (M11P_INSTANCES), so v10_setup checks it and a lab can pick it.
V10_PDL_CLASSES = (5, 9, 17)

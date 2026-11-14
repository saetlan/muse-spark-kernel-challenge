"""CPU emulation of attention/local/cuda_unified.py's ud_kernel (one launch per decode layer):
- queries folded in FP32 and rounded to FP16, the new token's K/V computed and written
  (emulate_cuda_fused.fold_queries / new_rows: the kernel patches them into its ring slot and
  stores them to the cache);
- the attention core = D2's per-warp ring at register/shared-memory level
  (emulate_cuda_dec3.attend_dec3: CTA partials after the warp merge, per split);
- the hierarchical merge exactly as the kernel indexes it: split -> cluster ci = split // CS,
  rank = split % CS; each cluster's leader merges its CS ranks (log2 max, weights, sum), then,
  with several clusters, the last leader merges the cluster partials; finalize = (o / l) x
  v_scale, RMSNorm over the 64 dims, x sigmoid(gate), BF16.
Run: .venv/bin/python attention/local/emulate_cuda_ud.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import load_submission  # noqa: E402
from emulate_cuda_dec import tail_key_case  # noqa: E402
from emulate_cuda_dec3 import attend_dec3  # noqa: E402
from emulate_cuda_fused import fold_queries, new_rows  # noqa: E402


def merge(ms, ls, os_):
    """Merge partials (log2-domain max, sum, output) along dim 0."""
    mm = ms.max(0).values
    wt = torch.exp2(ms - mm)
    return mm, (wt * ls).sum(0), (wt[..., None] * os_).sum(0)


def run_ud(data, sub, cs, nc, warps=4, stages=3):
    cfg = data["config"]
    hq, hkv, B = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["batch"]
    nsplit = cs * nc
    q32 = fold_queries(data, sub)
    new_rows(data, sub)
    o, m, l_ = attend_dec3(data, q32, nsplit, warps, stages)
    o = o.view(B, hq, nsplit, 64)
    m = m.view(B, hq, nsplit)
    l_ = l_.view(B, hq, nsplit)
    out = torch.empty(B, 1, hq * 64)
    gate = data["qkv"].reshape(B, -1)[:, (hq + 2 * hkv) * 64:].float().view(B, hq, 64)
    vsc = float(data["v_scale"])
    for b in range(B):
        for head in range(hq):
            cm, cl, co = [], [], []
            for ci in range(nc):                                   # leader of cluster ci
                sl = slice(ci * cs, ci * cs + cs)                  # its ranks 0 .. CS-1
                mm, ll, oo = merge(m[b, head, sl], l_[b, head, sl], o[b, head, sl])
                cm.append(mm), cl.append(ll), co.append(oo)
            if nc == 1:
                mm, ll, oo = cm[0], cl[0], co[0]
            else:                                                  # the last leader (ticket)
                mm, ll, oo = merge(torch.stack(cm), torch.stack(cl), torch.stack(co))
            a = oo * (vsc / ll)
            a = a * torch.rsqrt((a * a).sum() / 64.0 + 1e-5)
            out[b, 0, head * 64:(head + 1) * 64] = a / (1.0 + torch.exp(-gate[b, head]))
    return out.to(torch.bfloat16)


def main():
    import reference as ref
    sub = load_submission()
    quick = "--quick" in sys.argv
    cases = [
        (dict(batch=1, ctx=10, heads="tp8", layer="sliding"), (8, 1)),
        (dict(batch=1, ctx=3000, heads="tp8", layer="sliding", page=1, dist="soft"), (16, 1)),
        (dict(batch=3, ctx=2500, heads="tp8", layer="sliding", page=1, dist="edges"), (8, 1)),
        (dict(batch=2, ctx=6000, heads="tp8", layer="full", page=1, dist="coherent"), (8, 3)),
        (dict(batch=1, ctx=9000, heads="tp8", layer="full", page=1, dist="soft"), (16, 2)),
        (dict(batch=2, ctx=2600, heads="tp8", layer="full", page=64), (8, 2)),
        (dict(batch=2, ctx=2500, heads="tp8", layer="sliding", kv_scale=0.015625, page=1), (8, 1)),
        (dict(batch=1, ctx=2200, heads="dp", layer="sliding", page=1), (8, 1)),
        (dict(batch=1, ctx=1800, heads="dp", layer="full", dist="coherent"), (8, 2)),
    ]
    if quick:
        cases = [cases[1], cases[3], cases[5], cases[7]]
    bad = 0
    for i, (case, (cs, nc)) in enumerate(cases):
        data = ref.generate_input(seed=1100 + i, device="cpu", **case)
        out = run_ud(data, sub, cs, nc)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"{case} CS {cs} clusters {nc} -> {msg}")
        bad += not ok
    data = tail_key_case(ref)
    out = run_ud(data, sub, 8, 2)
    ok, msg = ref.check_implementation(data, ref.collect(data, out))
    print(("PASS " if ok else "FAIL ") + f"newest key weighted, partial last chunk, CS 8 x 2 -> {msg}")
    bad += not ok
    print("emulate_cuda_ud:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

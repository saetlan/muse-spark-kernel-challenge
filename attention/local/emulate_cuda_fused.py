"""CPU emulation of attention/local/cuda_fused.py's fused_dec_kernel (S1b): the A5b attention core at
register level (emulate_cuda_dec2.attend_dec2, the same loop the kernel carries), with the parts
the fused kernel adds:
- queries folded from qkv in FP32 (RMSNorm x qk factor, interleaved RoPE with the given inv_freq
  tensor) and rounded straight to FP16 (no BF16 step);
- the pair's new key/value computed in FP32 (RMSNorm + RoPE for the key, / scales), rounded to
  E4M3 and placed at cache row cache_loc[b] (the kernel patches them into its fragments and
  stores them; only the CTA holding position n-1 reads that row);
- the merge of all slices + RMSNorm x sigmoid(gate) (the ticket's last CTA).
Run: .venv/bin/python attention/local/emulate_cuda_fused.py [--quick]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from emulate import combine, load_submission  # noqa: E402
from emulate_cuda_dec import tail_key_case  # noqa: E402
from emulate_cuda_dec2 import attend_dec2  # noqa: E402

EPS = 1e-5


def fold_queries(data, sub):
    cfg = data["config"]
    hq, hkv, B = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["batch"]
    qkv = data["qkv"].reshape(B, -1).float()
    q = qkv[:, :hq * 64].view(B, hq, 64)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) / 64.0 + EPS) * cfg["qk_scale_factor"]
    if cfg["rope_theta"]:
        inv = sub._inv_freq(cfg["rope_theta"], "cpu")
        ang = data["positions"].view(B).float()[:, None] * inv[None, :]
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        e, o = q[..., 0::2], q[..., 1::2]
        q = torch.stack([e * c - o * s, e * s + o * c], -1).flatten(-2)
    return q                                            # FP32; attend_dec2 rounds to FP16


def new_rows(data, sub):
    cfg = data["config"]
    hq, hkv, B = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["batch"]
    qkv = data["qkv"].reshape(B, -1).float()
    k = qkv[:, hq * 64:(hq + hkv) * 64].view(B, hkv, 64)
    v = qkv[:, (hq + hkv) * 64:(hq + 2 * hkv) * 64].view(B, hkv, 64)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) / 64.0 + EPS)
    if cfg["rope_theta"]:
        inv = sub._inv_freq(cfg["rope_theta"], "cpu")
        ang = data["positions"].view(B).float()[:, None] * inv[None, :]
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        e, o = k[..., 0::2], k[..., 1::2]
        k = torch.stack([e * c - o * s, e * s + o * c], -1).flatten(-2)
    kc = data["k_cache"].view(-1, hkv, 64)
    vc = data["v_cache"].view(-1, hkv, 64)
    kc[data["cache_loc"]] = (k / data["k_scale"]).to(torch.float8_e4m3fn)
    vc[data["cache_loc"]] = (v / data["v_scale"]).to(torch.float8_e4m3fn)


def run_fused(data, sub, nsplit, warps=4):
    q32 = fold_queries(data, sub)
    new_rows(data, sub)
    o, m, l_ = attend_dec2(data, q32, nsplit, warps)
    return combine(data, o, m, l_, nsplit, 1 << max(0, (nsplit - 1).bit_length()), sub)


def main():
    import reference as ref
    sub = load_submission()
    quick = "--quick" in sys.argv
    cases = [
        (dict(batch=1, ctx=10, heads="tp8", layer="sliding"), [1, 3]),
        (dict(batch=3, ctx=2047, heads="tp8", layer="sliding", page=1, dist="soft"), [1, 16]),
        (dict(batch=2, ctx=3000, heads="tp8", layer="full", page=1, dist="coherent"), [7, 40]),
        (dict(batch=2, ctx=2600, heads="tp8", layer="full", page=64), [3]),
        (dict(batch=2, ctx=4000, heads="tp8", layer="sliding", dist="edges", page=1), [4]),
        (dict(batch=2, ctx=2500, heads="tp8", layer="full", kv_scale=0.015625, page=1), [3]),
        (dict(batch=3, ctx=2500, heads="tp8", layer="sliding", kv_scale=0.015625, page=1), [5]),
        (dict(batch=1, ctx=1500, heads="dp", layer="full", dist="coherent"), [2]),
        (dict(batch=2, ctx=2100, heads="dp", layer="sliding", page=1), [2]),
    ]
    if quick:
        cases = [cases[1], cases[2], cases[3], cases[4], cases[6], cases[7]]
    bad = 0
    for i, (case, splits) in enumerate(cases):
        for nsplit in splits[-1:] if quick else splits:
            data = ref.generate_input(seed=500 + i, device="cpu", **case)
            out = run_fused(data, sub, nsplit)
            ok, msg = ref.check_implementation(data, ref.collect(data, out))
            print(("PASS " if ok else "FAIL ") + f"{case} nsplit {nsplit} -> {msg}")
            bad += not ok
    for nsplit in (1, 2):
        data = tail_key_case(ref)
        out = run_fused(data, sub, nsplit)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        print(("PASS " if ok else "FAIL ") + f"newest key weighted, partial last chunk, nsplit {nsplit} -> {msg}")
        bad += not ok
    print("emulate_cuda_fused:", "all passed" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

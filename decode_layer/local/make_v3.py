"""Build submission_v3.py from submission_d1.py + parts/m11_e3c.py + parts/v3.py.

D1's prepare and kernel_fn become _d1_prepare and _d1_kernel_fn (kept for labs). v3's kernel_fn is D1's
kernel_fn with exact call-site swaps (each must match once): L2 prefetch fork after the qkv GEMM, the FP32
router, fused local ids, the expert path (pre-expert norm + M11 / Marlin), the unpooler slice, and the
prefetch join before the return. Everything else stays SGLang's.

Run: .venv/bin/python decode_layer/local/make_v3.py
"""

from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
SWAPS = (
    ('    qkv = _gemm(data["attn_input"], dense["qkv_gate"])\n',
     '    qkv = _gemm(data["attn_input"], dense["qkv_gate"])\n    forked = v3_prefetch_fork(state, cfg)\n'),
    ('            logits = F.linear(router_input, state["router"])\n',
     '            logits = v3_router_logits(router_input, state)\n'),
    ('    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS\n',
     '    local_ids = v3_local_ids(topk_ids, data["local_start"], state)\n'),
    ('    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n',
     '    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n'),
    ('        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])\n',
     '        unpooled = v3_unpool_slice(expert_output, state)\n'),
    ('    return next_attn_input, stream, topk_ids\n',
     '    v3_prefetch_join(state, forked)\n    return next_attn_input, stream, topk_ids\n'),
)
SEP = "\n\n# " + "=" * 70 + " {name}\n\n"


def build(out=DECODE / "submission_v3.py"):
    d1 = (DECODE / "submission_d1.py").read_text()
    first = '"""D1: SGLang\'s decoder layer'
    assert d1.count(first) == 1
    d1 = d1.replace(first, '"""v3 = D1 + E3c M11 experts, L2 prefetch during attention, an FP32 router, fused local ids and a\nTriton pre-expert norm (see the v3 section at the end).\n\nD1: SGLang\'s decoder layer', 1)
    kernel = d1[d1.index("\ndef kernel_fn(data, state):"):]
    for old, new in SWAPS:
        if kernel.count(old) != 1:
            raise SystemExit(f"expected exactly one {old.strip()!r} in D1's kernel_fn")
        kernel = kernel.replace(old, new)
    for name in ("prepare(weights", "kernel_fn(data, state"):
        if d1.count(f"\ndef {name}") != 1:
            raise SystemExit(f"expected one def {name} in submission_d1.py")
        d1 = d1.replace(f"\ndef {name}", f"\ndef _d1_{name}", 1)
    parts = [d1.rstrip() + "\n"]
    for name in ("m11_e3c", "v3"):
        text = (DECODE / "parts" / f"{name}.py").read_text()
        assert not text.startswith("# okc:")
        parts.append(SEP.format(name=f"parts/{name}.py") + text.strip() + "\n")
    parts.append(SEP.format(name="v3 kernel_fn: D1's with call sites swapped (local/make_v3.py)") + kernel.strip() + "\n")
    out.write_text("".join(parts))
    print(f"wrote {out} ({sum(map(len, parts))} chars)")


if __name__ == "__main__":
    build()

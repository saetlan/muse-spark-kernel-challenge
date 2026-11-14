"""Build submission_v2.py from submission_d1.py + parts/m11.py + parts/v2.py.

D1's prepare and kernel_fn become _d1_prepare and _d1_kernel_fn (kept for labs). v2's kernel_fn is
D1's kernel_fn with exactly three call sites swapped (router, experts, unpooler slice), so every
other line stays SGLang's. parts/v2.py defines the new prepare.

Run: .venv/bin/python decode_layer/local/make_v2.py
"""

from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
SWAPS = (
    ('            logits = F.linear(router_input, state["router"])\n',
     '            logits = v2_router_logits(router_input, state)\n'),
    ('    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n',
     '    expert_output = v2_experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n'),
    ('        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])\n',
     '        unpooled = v2_unpool_slice(expert_output, state)\n'),
)
SEP = "\n\n# " + "=" * 70 + " {name}\n\n"


def build(out=DECODE / "submission_v2.py"):
    d1 = (DECODE / "submission_d1.py").read_text()
    first = '"""D1: SGLang\'s decoder layer'
    assert d1.count(first) == 1
    d1 = d1.replace(first, '"""v2 = D1 + M11 experts, an FP32 router kernel and tiny_gemm (see the v2 section at the end).\n\nD1: SGLang\'s decoder layer', 1)
    start = d1.index("\ndef kernel_fn(data, state):")
    kernel = d1[start:]
    for old, new in SWAPS:
        if kernel.count(old) != 1:
            raise SystemExit(f"expected exactly one {old.strip()!r} in D1's kernel_fn")
        kernel = kernel.replace(old, new)
    for name in ("prepare(weights", "kernel_fn(data, state"):
        if d1.count(f"\ndef {name}") != 1:
            raise SystemExit(f"expected one def {name} in submission_d1.py")
        d1 = d1.replace(f"\ndef {name}", f"\ndef _d1_{name}", 1)
    parts = [d1.rstrip() + "\n"]
    for name in ("m11", "v2"):
        text = (DECODE / "parts" / f"{name}.py").read_text()
        assert not text.startswith("# okc:")
        parts.append(SEP.format(name=f"parts/{name}.py") + text.strip() + "\n")
    parts.append(SEP.format(name="v2 kernel_fn: D1's with the router, experts and unpooler-slice calls swapped")
                 + kernel.strip() + "\n")
    out.write_text("".join(parts))
    print(f"wrote {out} ({sum(map(len, parts))} chars)")


if __name__ == "__main__":
    build()

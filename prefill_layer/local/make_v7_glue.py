"""Apply the v7 grouped-path glue edits (first-turn class, >= 8192 tokens) to a v5-based submission file:
- inv[perm[i]] = i by one Triton kernel instead of torch.empty_like + scatter_ (96 us in v5's scored profile at 16K);
- counts by a block-local Triton histogram instead of scatter_add_ (95.6 us at 16K in RE24's profile);
- the gather walks routes in token order (a[inv[r]] = x[r // TOPK] for local routes) instead of sorted rows, so x is
  read sequentially (R14's gather order: 110 us vs v5's 180 us at 16K).
Same rows, same inv, same offsets: the GEMMs, act and combine see bit-identical inputs.

Usage: .venv/bin/python prefill_layer/local/make_v7_glue.py <in.py> <out.py>"""

import sys
from pathlib import Path

KERNELS_ANCHOR = "def _dequant(q, sf, counts, out):\n"
KERNELS = '''INV_BLOCK = 1024
HIST_BLOCK = 4096
HIST_BINS = 64  # >= LOCAL_EXPERTS + 2 (keys are 0..LOCAL_EXPERTS; masked lanes go to bin LOCAL_EXPERTS + 1)


@triton.jit
def _key_hist_kernel(keys_ptr, counts_ptr, n, NE: tl.constexpr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    """counts[e] += #(keys == e) for e <= NE, one block-local histogram then one atomic per bin (v7: replaces
    scatter_add_'s 131K contended atomics on 33 counters)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(keys_ptr + i, mask=i < n, other=NE + 1)
    h = tl.histogram(k, BINS)
    b = tl.arange(0, BINS)
    tl.atomic_add(counts_ptr + b, h, mask=b <= NE)


@triton.jit
def _inv_perm_kernel(perm_ptr, inv_ptr, n, BLOCK: tl.constexpr):
    """inv[perm[i]] = i for i < n (v7: replaces empty_like + scatter_)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    p = tl.load(perm_ptr + i, mask=m, other=0)
    tl.store(inv_ptr + p, i.to(tl.int64), mask=m)


@triton.jit
def _gather_routes_kernel(x_ptr, keys_ptr, inv_ptr, out_ptr, H: tl.constexpr, TOPK: tl.constexpr,
                          NE: tl.constexpr):
    """out[inv[r]] = x[r // TOPK] for local routes (keys[r] < NE), routes in token order (v7)."""
    r = tl.program_id(0)
    if tl.load(keys_ptr + r) >= NE:
        return
    row = tl.load(inv_ptr + r).to(tl.int64)
    tok = (r // TOPK).to(tl.int64)
    cols = tl.arange(0, H)
    tl.store(out_ptr + row * H + cols, tl.load(x_ptr + tok * H + cols))


'''
OLD = '''    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,
                                  num_warps=4)
'''
NEW = '''    _key_hist_kernel[(triton.cdiv(slots, HIST_BLOCK),)](keys.contiguous(), counts, slots, NE=LOCAL_EXPERTS,
                                                        BLOCK=HIST_BLOCK, BINS=HIST_BINS, num_warps=4)
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    _inv_perm_kernel[(triton.cdiv(slots, INV_BLOCK),)](perm, inv, slots, BLOCK=INV_BLOCK, num_warps=4)
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_routes_kernel[(slots,)](x.contiguous(), keys.contiguous(), inv, a, H=D_EXPERT, TOPK=TOP_K,
                                    NE=LOCAL_EXPERTS, num_warps=4)
'''


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    s = src.read_text()
    assert s.count(KERNELS_ANCHOR) == 1 and s.count(OLD) == 1, "base file does not have v5's grouped glue"
    assert "import triton\n" in s or "\nimport triton" in s
    s = s.replace(KERNELS_ANCHOR, KERNELS + KERNELS_ANCHOR).replace(OLD, NEW)
    dst.write_text(s)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()

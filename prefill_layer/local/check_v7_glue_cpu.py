"""CPU check of the v7 grouped-path glue (make_v7_glue.py): ports of _inv_perm_kernel and _gather_routes_kernel
(same index formulas, grid and masks; the kernel lines are pinned) against v5's own orchestration (torch scatter_
for inv, the sorted-row gather) on full-size random routing (16,384 tokens x top 8 = 131,072 routes, ~1/4 local,
plus all-local / none-local / one-expert cases). inv must be identical and the gathered rows identical.

Run: .venv/bin/python prefill_layer/local/check_v7_glue_cpu.py prefill_layer/v7e_glue.py [more files]"""

import sys
from pathlib import Path

import torch

PINS = [
    "    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
    "    m = i < n",
    "    p = tl.load(perm_ptr + i, mask=m, other=0)",
    "    tl.store(inv_ptr + p, i.to(tl.int64), mask=m)",
    "    if tl.load(keys_ptr + r) >= NE:",
    "    row = tl.load(inv_ptr + r).to(tl.int64)",
    "    tok = (r // TOPK).to(tl.int64)",
    "    tl.store(out_ptr + row * H + cols, tl.load(x_ptr + tok * H + cols))",
    "    _inv_perm_kernel[(triton.cdiv(slots, INV_BLOCK),)](perm, inv, slots, BLOCK=INV_BLOCK, num_warps=4)",
    "    _gather_routes_kernel[(slots,)](x.contiguous(), keys.contiguous(), inv, a, H=D_EXPERT, TOPK=TOP_K,",
    "INV_BLOCK = 1024",
    "    k = tl.load(keys_ptr + i, mask=i < n, other=NE + 1)",
    "    h = tl.histogram(k, BINS)",
    "    tl.atomic_add(counts_ptr + b, h, mask=b <= NE)",
    "    _key_hist_kernel[(triton.cdiv(slots, HIST_BLOCK),)](keys.contiguous(), counts, slots, NE=LOCAL_EXPERTS,",
    "HIST_BLOCK = 4096",
    "HIST_BINS = 64  # >= LOCAL_EXPERTS + 2 (keys are 0..LOCAL_EXPERTS; masked lanes go to bin LOCAL_EXPERTS + 1)",
]
HIST_FILES_ONLY = PINS[-6:]


def port_inv(perm, slots, block=1024):
    inv = torch.full_like(perm, -7)
    for pid in range(-(-slots // block)):          # grid (cdiv(slots, BLOCK),)
        i = pid * block + torch.arange(block)
        m = i < slots
        p = perm[i[m]]
        inv[p] = i[m].to(torch.int64)
    return inv


def port_hist(keys, ne, block=4096, bins=64):
    counts = torch.zeros(ne + 1, dtype=torch.int32)
    n = keys.numel()
    for pid in range(-(-n // block)):              # grid (cdiv(slots, HIST_BLOCK),)
        i = pid * block + torch.arange(block)
        k = torch.where(i < n, keys[i.clamp(max=n - 1)].long(), torch.full_like(i, ne + 1))
        h = torch.bincount(k, minlength=bins)[:bins].to(torch.int32)   # tl.histogram: unit bins from 0
        b = torch.arange(bins)
        counts += h[b <= ne]                       # atomic_add masked to b <= NE
    return counts


def port_gather(x, keys, inv, out, topk, ne):
    for r in range(keys.numel()):                  # grid (slots,): one route per program
        if int(keys[r]) >= ne:
            continue
        out[int(inv[r])] = x[r // topk]


def v5_orchestration(x, keys, topk, ne):
    slots = keys.numel()
    sorted_keys, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(ne + 1, dtype=torch.int32)
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:ne].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, dtype=perm.dtype))
    n = int(offs[ne - 1])
    a = torch.full((slots, x.shape[1]), -1.0)
    a[:n] = x[perm[:n] // topk]                    # v5's _gather_rows_kernel
    return perm, inv, a, n, counts


def main():
    files = [Path(f) for f in sys.argv[1:]] or [Path(__file__).resolve().parents[1] / "v7e_glue.py"]
    hist = True
    for f in files:
        src = f.read_text()
        pins = PINS if "_key_hist_kernel" in src else [p for p in PINS if p not in HIST_FILES_ONLY]
        hist = hist and "_key_hist_kernel" in src
        missing = [p for p in pins if src.count(p + "\n") < 1]
        assert not missing, f"{f.name}: kernel lines changed, update the ports: {missing[:2]}"
    gen = torch.Generator().manual_seed(7)
    topk, ne, h = 8, 32, 16
    cases = []
    tokens = 16384
    ids = torch.randint(0, 128, (tokens, topk), generator=gen)
    cases.append(("16K, 1/4 local", torch.where(ids < ne, ids, torch.full_like(ids, ne)).view(-1).to(torch.int32)))
    cases.append(("all local", torch.randint(0, ne, (2048 * topk,), generator=gen).to(torch.int32)))
    cases.append(("none local", torch.full((1000 * topk,), ne, dtype=torch.int32)))
    cases.append(("one expert", torch.full((777 * topk,), 5, dtype=torch.int32)))
    for name, keys in cases:
        t = keys.numel() // topk
        x = torch.randn(t, h, generator=gen)
        perm, inv_ref, a_ref, n, counts_ref = v5_orchestration(x, keys, topk, ne)
        if hist:
            assert torch.equal(port_hist(keys, ne), counts_ref), name
        inv = port_inv(perm, keys.numel())
        assert torch.equal(inv, inv_ref), name
        a = torch.full((keys.numel(), h), -1.0)
        port_gather(x, keys, inv, a, topk, ne)
        assert torch.equal(a[:n], a_ref[:n]) and torch.equal(a[n:], a_ref[n:]), name
        print(f"ok: {name}: {keys.numel()} routes, {n} local rows; inv, gathered rows" + (", counts" if hist else "")
              + " identical to v5's")
    print(f"ok: {len(PINS)} kernel lines pinned in {', '.join(f.name for f in files)}")


if __name__ == "__main__":
    main()

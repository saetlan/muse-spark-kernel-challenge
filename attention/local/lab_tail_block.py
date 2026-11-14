


# ---------------------------------------------------------------------------------------------
# A22 tail test (B6): does the longest request set the full-layer kernels' time? v14's path (D3 on Fd16 / Fd32, W3b on
# Fv16 / Fv32) timed on the same requests and pages with (a) the lab's agentic lengths (80,000 x (1 +- 0.075)), (b) every
# request at their mean length (same total keys), (c) every request at their maximum. If (a) ~ (b), equal splits per
# request leave no tail and A22 (length-aware split) is dropped; if (a) ~ (c), the longest request sets the time and a
# balanced split could save up to (a - b) / a.
# ---------------------------------------------------------------------------------------------
def _tail_inputs(batch, length, k_cache, v_cache, gen, seq):
    data = _lab_inputs(batch, length, 0, k_cache, v_cache, gen)
    dev = k_cache.device
    table = data["page_table"]
    seq = seq.to(dev).long()
    pos = (seq[:, None] + torch.arange(length, device=dev)).reshape(-1)
    data["positions"] = pos
    data["cache_loc"] = torch.gather(table.long(), 1, pos.view(batch, length)).reshape(-1)
    data["cache_seqlens"] = (seq + length).to(torch.int32)
    data["seq_lens"] = seq.to(torch.int32)
    return data


def _lab_tail(lab, k_cache, v_cache, gen):
    for name, batch, length in (("Fd16", 16, 1), ("Fd32", 32, 1), ("Fv16", 16, 8), ("Fv32", 32, 8)):
        if lab.left() < 20:
            _log("tail: out of budget")
            break
        try:
            u = torch.rand((batch,), generator=gen, device=k_cache.device)
            seq = (80000 * (1 + 0.075 * (2 * u - 1))).long()
            res = []
            for tag, s in (("agentic", seq), ("mean", torch.full_like(seq, int(seq.float().mean()))),
                           ("max", torch.full_like(seq, int(seq.max())))):
                data = _tail_inputs(batch, length, k_cache, v_cache, gen, s)
                _kernel_fn_submitted(data)
                res.append((tag, lab.time_graph(lambda: _kernel_fn_submitted(data))))
            a, m, x = (r[1] for r in res)
            _log(f"tail {name} ({CUDA_ROUTE.get((True, length == 8, batch))}): agentic {a:.1f}us, mean-length {m:.1f}us, "
                 f"max-length {x:.1f}us; agentic / mean {a / m:.3f}, max / mean {x / m:.3f}; longest / mean length "
                 f"{float(seq.max()) / float(seq.float().mean()):.3f}")
        except Exception as exc:
            _log(f"tail {name}: FAILED {_why(exc)}")

# L10: more L2 staging on L9's kernels (parts/l9.py; plain prefetches only, no eviction-priority policy, so
# nothing is meant to outlive the call), read from the same per-case dict state["l9_mode"]:
#   pool  the first p/16 of the pooler's 67 MB (its first rows: bf16_proj's FMA / swap kernels take rows in
#         program order and each row from K = 0) during the attention window, after o;
#   unp   the first p/16 of the full unpooler (> 8 tokens) during the expert stage (M11 / Marlin leave HBM
#         headroom), joined right after the experts.


def l10_fork_attention(cfg, state):
    m = l9_mode(cfg, state)
    if not m.get("pool"):
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["pooler"]
        nbytes = (w.numel() * w.element_size() * m["pool"] // 16) & ~15
        state["l9"].l9_ranges(w, nbytes, w, 0, m.get("pm", 0), state["l9_blocks"])


def l10_fork_unpool(cfg, state):
    m = l9_mode(cfg, state)
    if not m.get("unp") or cfg["tokens"] <= SLICED_UNPOOL_MAX_TOKENS:
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["unpooler"]
        nbytes = (w.numel() * w.element_size() * m["unp"] // 16) & ~15
        state["l9"].l9_ranges(w, nbytes, w, 0, m.get("pm", 0), state["l9_blocks"])


def l10_join_unpool(cfg, state):
    m = l9_mode(cfg, state)
    if m.get("unp") and cfg["tokens"] > SLICED_UNPOOL_MAX_TOKENS:
        torch.cuda.current_stream().wait_stream(state["side_stream"])


def l10_setup(state, weights):
    print(f"[l10] pooler / unpooler staging: {'on' if 'l9' in state else 'OFF (needs l9)'}")

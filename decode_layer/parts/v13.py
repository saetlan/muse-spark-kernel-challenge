# v13 = v12 + one kernel fewer at <= 8 tokens: SGLang pads this GPU's 1024 unpooler columns to the full 8192 with
# F.pad (ATen constant_pad_nd: an empty output, a fill kernel, then a copy kernel into the narrowed view). v13 builds the
# same tensor with one torch.cat of the slice and a constant zero block allocated in prepare (zeros, not derived from
# any input; the values are bit-identical to F.pad's). On only where a lab measured it faster (V13_CAT).

V13_CAT = False  # set from a lab's printed V13_CAT


def v13_pad(unpooled, state):
    zeros = state.get("v13_zeros")
    if zeros is None or unpooled.shape[0] > zeros.shape[0] or unpooled.dtype != zeros.dtype:
        return F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    return torch.cat((unpooled, zeros[: unpooled.shape[0]]), dim=1)


def v13_install(state, on):
    if on:
        device = state["dense"]["unpooler"].device
        state["v13_zeros"] = torch.zeros((SLICED_UNPOOL_MAX_TOKENS, HIDDEN - UNPOOL_SLICE), dtype=torch.bfloat16,
                                         device=device)
    else:
        state.pop("v13_zeros", None)


def v13_setup(state, weights):
    """Install the concat path if V13_CAT, after checking it equals F.pad bit for bit at 1..8 tokens."""
    v13_install(state, V13_CAT)
    if not V13_CAT:
        print("[v13] unpooler pad: F.pad (SGLang's)")
        return
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(61)
    ok = True
    for tokens in range(1, SLICED_UNPOOL_MAX_TOKENS + 1):
        x = torch.randn((tokens, UNPOOL_SLICE), generator=gen, device=device).to(torch.bfloat16)
        got, ref = v13_pad(x, state), F.pad(x, (0, HIDDEN - UNPOOL_SLICE))
        ok &= got.shape == ref.shape and got.dtype == ref.dtype and got.is_contiguous() and bool(torch.equal(got, ref))
    torch.cuda.synchronize()
    if not ok:
        v13_install(state, False)
    print(f"[v13] unpooler pad: {'one concat kernel (bit-identical to F.pad at 1-8 tokens)' if ok else 'F.pad (concat check FAILED)'}")

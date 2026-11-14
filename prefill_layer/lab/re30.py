"""RE30 lab (check runs only): the dense projections at the turn-1 / turn-4 token counts. SGLang's baseline and our
kernel_fn both call _bf16_gemm_dispatch_impl (same cuBLAS kernel), so a gain needs a better kernel choice: time each
dense weight (qkv_gate, o, pooler, unpooler) at M = 753 / 3,012 as called today, and with the rows padded to
768 / 3,072 (copy into a zero-padded buffer included, output sliced). CUDA-graph replays, cold L2. kernel_fn = v7g's.
Appended after v7g_glue2.py by local/make_lab.py --base v7g_glue2.py."""

RE30_LIMIT_S = 250.0


def re30_sec(lab, weights, state):
    dense = state["dense"]
    gen = torch.Generator(device="cuda").manual_seed(30)
    for m, mp in ((753, 768), (3012, 3072)):
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            w = dense[name]
            k = w.shape[1]
            x = (torch.randn((m, k), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            xp = torch.zeros((mp, k), device="cuda", dtype=torch.bfloat16)
            base = lab.time_graph(lambda: _gemm(x, w), reps=15)

            def padded():
                xp[:m].copy_(x)
                return _gemm(xp, w)[:m]

            pad = lab.time_graph(padded, reps=15)
            ref = _gemm(x, w)
            same = torch.equal(padded(), ref)
            note(f"RE30 {name} M {m}: dispatch {base:.1f} us, padded to {mp} {pad:.1f} us "
                 f"({100.0 * (pad - base) / base:+.1f}%), bit-equal {same}")


RE7_SUMMARY = []


def note(msg):
    log(msg)
    RE7_SUMMARY.append(msg)


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    note(f"submission prepare {time.monotonic() - t0:.1f}s")
    lab = Lab(RE30_LIMIT_S - (time.monotonic() - t0))
    lab.section("re30 dense", re30_sec, weights, state, min_left=20.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log("===== summary =====")
    for line in RE7_SUMMARY:
        log(line)
    return state

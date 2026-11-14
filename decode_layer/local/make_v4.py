"""Build submission_v4.py: submission_d1.py + parts/m11_e3c.py + parts/v3.py + parts/attn.py + parts/proj.py +
parts/v4.py, then _v4_attention_d1 (D1's attention block, extracted verbatim) and v4's kernel_fn = D1's with
exact call-site swaps. D1's prepare/kernel_fn become _d1_*, v3's prepare becomes _v3_prepare.

Run: .venv/bin/python decode_layer/local/make_v4.py            (submission_v4.py)
     .venv/bin/python decode_layer/local/make_v4.py --pf       (submission_v5.py: the same with parts/m11_pf.py,
                                                              M11 + L8 next-task L2 prefetch, pf from the configs)
     .venv/bin/python decode_layer/local/make_v4.py --v6       (submission_v6.py: v5 with attention v5 (parts/attn5.py),
     ... [--pf | --v6] --table <profiles log>                  (append the M11 table a lab printed, V5_M11_CONFIGS)
     ... [--v6 | --v7] --modes <log> / --stage <log>            (append a lab's printed V4_MODE / L9_MODE)
                                                              bf16_proj v4 (parts/proj4.py) and parts/v6.py)
     .venv/bin/python decode_layer/local/make_v4.py --v9 [--modes <log>]
     .venv/bin/python decode_layer/local/make_v4.py --v11 [--modes <log>] [--pdl <log>]   (v10 with attention v8,
                                                              parts/attn8.py, and parts/v11.py: v7a router configs)
     .venv/bin/python decode_layer/local/make_v4.py --v10 [--modes <log>] [--pdl <log>]   (v9 + parts/m11_pdl.py and
                                                              parts/v10.py: MoE E15 / E16's PDL path, per class)
                                                             (submission_v9.py: v7 without L9, with attention v6
                                                              (parts/attn6.py), bf16_proj v6 (parts/proj6.py) and
                                                              parts/v9.py)
"""

import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
ATT_START = '    if cfg["rope_theta"]:\n        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])\n'
ATT_END_LINE = '    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)\n'
SWAPS = (
    ('    qkv = _gemm(data["attn_input"], dense["qkv_gate"])\n',
     '    qkv = v4_project(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], cfg, state)\n'),
    ('    attn_output = _gemm(o, dense["o"])  # all-reduce: identity\n',
     '    attn_output = v4_project(o, "o", dense["o"].shape[0], cfg, state)  # all-reduce: identity\n'),
    ('    expert_input = _gemm(ffn_input, dense["pooler"])\n',
     '    expert_input = v4_project(ffn_input, "pooler", dense["pooler"].shape[0], cfg, state)\n'),
    ('''        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
''', '        logits = v4_router(ffn_input, router_input, cfg, state)\n'),
    ('    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS\n',
     '    local_ids = v3_local_ids(topk_ids, data["local_start"], state)\n'),
    ('    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)\n',
     '    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n'),
    ('        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])\n',
     '        unpooled = v3_unpool_slice(expert_output, state)\n'),
    ('        ffn_output = _gemm(expert_output, dense["unpooler"])\n',
     '        ffn_output = v4_project(expert_output, "unpooler", dense["unpooler"].shape[0], cfg, state)\n'),
)
SEP = "\n\n# " + "=" * 70 + " {name}\n\n"


TABLE_SECTION = '''# The M11 table a lab chose (copied from its log by local/make_v4.py --table): it overrides parts/v3.py's
# V3_M11_CONFIGS, which _v3_prepare installs. m11_table_check (run by prepare after the glue's) checks each
# class above 32 tokens (new in DE7) against Marlin at its token count and drops (Marlin) any that misses 2e-2.
V3_M11_CONFIGS = @@TABLE@@
M11_TABLE_SOURCE = "@@SOURCE@@"


def m11_table_check(state, weights):
    if "m11" not in state:
        return
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(43)
    table, msgs = list(state["m11_configs"]), []
    for i, (below, cfg) in enumerate(table):
        if cfg is None or below <= 33:
            continue
        tokens = {192: 128, 321: 256}.get(below, below - 1)
        try:
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            sc[:, :LOCAL_EXPERTS] += 0.5
            ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            err = _v3_rel(m11_run(x, ids, tw, state, cfg), _experts(x, tw, ids, state))
        except Exception as exc:
            err = float("inf")
            msgs.append(f"T{tokens} {type(exc).__name__}")
        if not err < 2e-2:
            table[i] = (below, None)
        msgs.append(f"T{tokens} {err:.1e} -> {\'M11\' if err < 2e-2 else \'Marlin\'}")
    state["m11_configs"] = tuple(table)
    torch.cuda.synchronize()
    print(f"[table] M11 classes above 32 tokens vs Marlin ({M11_TABLE_SOURCE}): " + ("; ".join(msgs) or "none"))
'''


def table_section(log_name):
    """V5_M11_CONFIGS as a lab printed it (its last such line), verified as a literal."""
    import ast
    lines = [ln for ln in (DECODE / "profiles" / log_name).read_text().splitlines() if "V5_M11_CONFIGS = " in ln]
    text = lines[-1].split("V5_M11_CONFIGS = ", 1)[1].strip()
    table = ast.literal_eval(text)
    assert isinstance(table, tuple) and all(isinstance(b, int) for b, _ in table)
    return TABLE_SECTION.replace("@@TABLE@@", repr(table)).replace("@@SOURCE@@", log_name)


# Extra parts after the glue: each defines <name>_setup(state, weights), run by the generated prepare after
# the glue's, and may swap call sites in kernel_fn (exact, counted).
EXTRA_SWAPS = {
    "v15": (
        ('        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)\n'
         '    main_stream.wait_stream(side_stream)\n'
         '    if not torch.cuda.is_current_stream_capturing():\n'
         '        for tensor in (topk_weights, topk_ids):\n'
         '            tensor.record_stream(main_stream)\n'
         '    # This GPU\'s experts start at local_start: as SGLang\'s expert_map on that EP rank.\n'
         '    local_ids = v3_local_ids(topk_ids, data["local_start"], state)\n'
         '    # all-reduce: identity\n'
         '    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n',
         '        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)\n'
         '        side_ids = v15_side_ids(topk_ids, data["local_start"], state)  # v15: on the side stream when on\n'
         '    expert_normed = v15_pre_norm(expert_input, state)  # v15: before the join when on\n'
         '    main_stream.wait_stream(side_stream)\n'
         '    if not torch.cuda.is_current_stream_capturing():\n'
         '        for tensor in (topk_weights, topk_ids) + (() if side_ids is None else (side_ids,)):\n'
         '            tensor.record_stream(main_stream)\n'
         '    # This GPU\'s experts start at local_start: as SGLang\'s expert_map on that EP rank.\n'
         '    local_ids = v3_local_ids(topk_ids, data["local_start"], state) if side_ids is None else side_ids\n'
         '    # all-reduce: identity\n'
         '    expert_output = v15_experts(expert_input, expert_normed, topk_weights, local_ids, state)\n'),
    ),
    "v13": (
        ('        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity\n',
         '        ffn_output = v13_pad(unpooled, state)  # all-gather: identity (F.pad unless V13_CAT)\n'),
    ),
    "l9": (
        ('    qkv = v4_project(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], cfg, state)\n',
         '    qkv = v4_project(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], cfg, state)\n'
         '    l9_fork_attention(cfg, state)\n'),
        ('        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)\n',
         '        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)\n'
         '        l9_stage_experts(topk_ids, data["local_start"], cfg, state)\n'),
        ('    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n',
         '    l9_fork_slice(cfg, state)\n'
         '    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n'
         '    l9_join_slice(cfg, state)\n'),
    ),
    "l10": (  # applied after l9's swaps
        ("    l9_fork_attention(cfg, state)\n", "    l9_fork_attention(cfg, state)\n    l10_fork_attention(cfg, state)\n"),
        ("    l9_fork_slice(cfg, state)\n", "    l9_fork_slice(cfg, state)\n    l10_fork_unpool(cfg, state)\n"),
        ("    l9_join_slice(cfg, state)\n", "    l9_join_slice(cfg, state)\n    l10_join_unpool(cfg, state)\n"),
    ),
}


def printed_literal(log_name, key):
    """The literal a lab printed as `key = ...` (its last such line), verified with ast.literal_eval."""
    import ast
    lines = [ln for ln in (DECODE / "profiles" / log_name).read_text().splitlines() if f"{key} = " in ln]
    value = ast.literal_eval(lines[-1].split(f"{key} = ", 1)[1].strip())
    assert isinstance(value, (dict, bool))
    return value


def build(out=DECODE / "submission_v4.py", m11="m11_e3c", attn="attn", proj="proj", glue="v4", table=None, extras=(),
          picks=(), embeds=(), override=None):
    d1 = (DECODE / "submission_d1.py").read_text()
    first = '"""D1: SGLang\'s decoder layer'
    assert d1.count(first) == 1
    lead = ('"""v11 candidate = v10 (v9 + MoE E15 / E16\'s PDL path, per class) with the attention session\'s v8 and\n'
            'bf16_proj v7a\'s router configs (parts/v11.py)' + (', + SGLang\'s cluster norm-residual kernel at the '
            'layer boundaries,\nper token class (parts/v12.py).\n' if "v12" in extras else '.\n') +
            'v9 = v7 (scored 1.364x) without L9, with the attention session\'s v6 and the bf16_proj\n'
            'session\'s v6 (12-bit lossless dense weights) (parts/v9.py).\n'
            'v5 = v4 with parts/m11_pf.py (M11 + L8: next-task L2 prefetch, pf per config).\n\n' if "v11" in extras else
            '"""v9 candidate = v7 (scored 1.364x) without L9, with the attention session\'s v6 and the bf16_proj\n'
            'session\'s v6 (12-bit lossless dense weights) (parts/v9.py).\n'
            'v5 = v4 with parts/m11_pf.py (M11 + L8: next-task L2 prefetch, pf per config).\n\n' if glue == "v9" else
            '"""v6 candidate = v5 with the attention session\'s v5 and the bf16_proj session\'s v4 (parts/v6.py).\n'
            'v5 = v4 with parts/m11_pf.py (M11 + L8: next-task L2 prefetch, pf per config).\n\n' if glue == "v6" else
            '"""v5 candidate = v4 with parts/m11_pf.py (M11 + L8: next-task L2 prefetch, pf per config).\n\n'
            if m11 == "m11_pf" else '"""')
    d1 = d1.replace(first, lead + 'v4 = v3 + the attention session\'s kernels and the bf16_proj session\'s projections, per case\n(see the v4 section at the end).\n\nD1: SGLang\'s decoder layer', 1)
    kernel = d1[d1.index("\ndef kernel_fn(data, state):"):]
    # D1's attention block, verbatim, as a function
    a = kernel.index(ATT_START)
    b = kernel.index(ATT_END_LINE) + len(ATT_END_LINE)
    block = kernel[a:b]
    att_fn = ("def _v4_attention_d1(data, qkv, state):\n"
              "    \"\"\"D1's (SGLang's) attention block, extracted verbatim by local/make_v4.py.\"\"\"\n"
              "    cfg = data[\"config\"]\n"
              "    batch, length, tokens = cfg[\"batch\"], cfg[\"q_len\"], cfg[\"tokens\"]\n"
              "    hq, hkv, d = cfg[\"num_q_heads\"], cfg[\"num_kv_heads\"], cfg[\"head_dim\"]\n"
              + block + "    return o\n")
    kernel = kernel[:a] + "    o = v4_attention(data, qkv, cfg, state)\n" + kernel[b:]
    swaps = list(SWAPS)
    if glue in ("v6", "v9"):  # the unpooler slice through bf16_proj's CHOICE when pj is on (needs cfg)
        swaps = [(o, n.replace("v3_unpool_slice(expert_output, state)", "v6_unpool_slice(expert_output, cfg, state)"))
                 for o, n in swaps]
        assert sum("v6_unpool_slice" in n for _, n in swaps) == 1
    for name in extras:
        swaps += list(EXTRA_SWAPS.get(name, ()))
    for old, new in swaps:
        if kernel.count(old) != 1:
            raise SystemExit(f"expected exactly one {old.strip()[:60]!r} in D1's kernel_fn")
        kernel = kernel.replace(old, new)
    for name in ("prepare(weights", "kernel_fn(data, state"):
        assert d1.count(f"\ndef {name}") == 1
        d1 = d1.replace(f"\ndef {name}", f"\ndef _d1_{name}", 1)
    parts = [d1.rstrip() + "\n"]
    for name in (m11, "v3", attn, proj, *embeds, glue):
        text = (DECODE / "parts" / f"{name}.py").read_text()
        assert not text.startswith("# okc:")
        if name == "v3":
            assert text.count("\ndef prepare(weights") == 1
            text = text.replace("\ndef prepare(weights", "\ndef _v3_prepare(weights", 1)
        if name == glue and (table or extras):
            assert text.count("\ndef prepare(weights") == 1
            text = text.replace("\ndef prepare(weights", "\ndef _glue_prepare(weights", 1)
        parts.append(SEP.format(name=f"parts/{name}.py") + text.strip() + "\n")
    for name in extras:
        text = (DECODE / "parts" / f"{name}.py").read_text()
        assert f"\ndef {name}_setup(state, weights)" in text and "\ndef prepare(" not in text
        parts.append(SEP.format(name=f"parts/{name}.py") + text.strip() + "\n")
    for key, log_name in picks:  # a lab's printed per-case picks, as literals (they override the parts' defaults)
        parts.append(SEP.format(name=f"{key} from profiles/{log_name} (local/make_v4.py)")
                     + f"# The per-case picks the lab printed in profiles/{log_name}.\n{key} = {printed_literal(log_name, key)!r}\n")
    if override:  # (section name, text): applied after the picks (local/make_v4.py)
        parts.append(SEP.format(name=override[0]) + override[1])
    if table:
        parts.append(SEP.format(name=f"M11 table from profiles/{table} (local/make_v4.py --table)") + table_section(table))
    if table or extras:
        hooks = [f"    {name}_setup(state, weights)\n" for name in extras] + (["    m11_table_check(state, weights)\n"] if table else [])
        early = [f"    {name}_early(weights)\n" for name in extras
                 if f"\ndef {name}_early(weights)" in (DECODE / "parts" / f"{name}.py").read_text()]
        parts.append(SEP.format(name="prepare: extras' early steps, the glue's, then each extra section's setup "
                                     "(local/make_v4.py)")
                     + "def prepare(weights):\n" + "".join(early) + "    state = _glue_prepare(weights)\n"
                     + "".join(hooks) + "    return state\n")
    parts.append(SEP.format(name="D1's attention block (local/make_v4.py)") + att_fn)
    parts.append(SEP.format(name="v4 kernel_fn: D1's with call sites swapped (local/make_v4.py)") + kernel.strip() + "\n")
    out.write_text("".join(parts))
    print(f"wrote {out} ({sum(map(len, parts))} chars)")


V20B_M19 = (
    "# M19 (E25's two instances, the 65-191 class's path) also at 192-320 tokens: whole-layer routing-locked A/Bs at verify\n"
    "# 32 (sliding / full, weights 46 / 16) measured -2.3% (DE18), -2.0% (DE19), -2.1% (DE23: 633.0 -> 619.5 us, err 4e-3),\n"
    "# under the labs' 3% bar; included on the user's call. prepare's v17_setup checks M19 vs Marlin at 256 tokens.\n"
    "V17_CHOICE = {**V17_CHOICE, 321: 'm19'}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    tbl = args[args.index("--table") + 1] if "--table" in args else None
    picks = tuple((key, args[args.index(flag) + 1]) for flag, key in (("--modes", "V4_MODE"), ("--stage", "L9_MODE"),
                                                                      ("--pdl", "V10_PDL"), ("--boundary", "V12_BOUNDARY"),
                                                                      ("--cat", "V13_CAT"), ("--side", "V15_SIDE"),
                                                                      ("--m19", "V16_M19"), ("--fused", "V14_FUSED"),
                                                                      ("--choice", "V17_CHOICE"), ("--dgrs", "V21_DGRS"))
                  if flag in args)
    if "--v23" in args:  # v22 with MoE E49 (parts/moe49.py + parts/v23.py) in place of E36
        build(DECODE / "submission_v23.py", "m11_pdl", "attn18", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v20", "v23"), picks=picks, embeds=("moe49",))
    elif "--v22" in args:  # v20b with MoE E36 (parts/moe36.py subset + parts/v22.py: + split kinds s128 / s256) in place of
        # E25; --m19-256 re-applies v20b's M19 at 192-320 tokens over the --choice log's V17_CHOICE (the lab's base)
        build(DECODE / "submission_v22.py", "m11_pdl", "attn18", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v20", "v22"), picks=picks, embeds=("moe36",),
              override=("V17_CHOICE + M19 at 192-320 tokens (local/make_v4.py --v22 --m19-256)", V20B_M19)
              if "--m19-256" in args else None)
    elif "--v21" in args:  # v20 + bf16_proj v11b's DG12-RS for the pooler / unpooler at 32 tokens (parts/dgrs.py subset
        # + parts/v21.py: per (layer, tokens) from a lab's V21_DGRS)
        build(DECODE / "submission_v21.py", "m11_pdl", "attn14", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v20", "v21"), picks=picks,
              embeds=("moe25", "dgrs"))
    elif "--v20b" in args:  # v20 + M19 at 192-320 tokens (V17_CHOICE[321]), on the user's call on the 3% bar
        build(DECODE / "submission_v20b.py", "m11_pdl", "attn14", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v20"), picks=picks, embeds=("moe25",),
              override=("V17_CHOICE + M19 at 192-320 tokens (local/make_v4.py --v20b)", V20B_M19))
    elif "--v20" in args:  # v18 with attention v14 (parts/attn14.py, trimmed: + W3b full verify 16 / 32, UD4 sliding
        # decode) + parts/v20.py; G2P (parts/v14.py) dropped as in v19
        build(DECODE / "submission_v20.py", "m11_pdl", "attn14", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v20"), picks=picks, embeds=("moe25",))
    elif "--v19" in args:  # v18 with attention v11 (parts/attn11.py: + W2 / UW2 verify) + parts/v19.py
        # G2P (parts/v14.py) is off everywhere (DE18) and dropped: the judge rejects files over 1,000,000 bytes (DE22)
        build(DECODE / "submission_v19.py", "m11_pdl", "attn11", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15", "v17", "v18", "v19"), picks=picks, embeds=("moe25",))
    elif "--v18" in args:  # v17 with attention v9 (parts/attn9.py: W1-N verify) + parts/v18.py
        build(DECODE / "submission_v18.py", "m11_pdl", "attn9", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v14", "v15", "v17", "v18"), picks=picks, embeds=("moe25",))
    elif "--v17" in args:  # v15 + MoE E25's path (parts/moe25.py subset + parts/v17.py: M11b at 1-32, M19 at 65-191; P9
        # dropped) + G2P (parts/v14.py, off unless picked); replaces v16's E19 subset
        build(DECODE / "submission_v17.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v14", "v15", "v17"), picks=picks, embeds=("moe25",))
    elif "--v16" in args:  # v15 + MoE E19's M19 path at 65-320 tokens (parts/moe19.py subset + parts/v16.py; P9 dropped)
        # + G2P (parts/v14.py: the prefill session's fused GEMM2 + post-norm, shared/glue/g2p.py)
        build(DECODE / "submission_v16.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v14", "v15", "v16"), picks=picks, embeds=("moe19",))
    elif "--v15" in args:  # v13 + local ids on the side stream, pre-expert norm before the join (parts/v15.py)
        build(DECODE / "submission_v15.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13", "v15"), picks=picks)
    elif "--v13" in args:  # v12 + one concat instead of F.pad's fill + copy at <= 8 tokens (parts/v13.py)
        build(DECODE / "submission_v13.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl,
              extras=("v10", "v11", "v12", "v13"), picks=picks)
    elif "--v12" in args:  # v11 + SGLang's cluster norm-residual kernel at the boundaries, per class (parts/v12.py)
        build(DECODE / "submission_v12.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl, extras=("v10", "v11", "v12"),
              picks=picks)
    elif "--v11" in args:  # v10 + attention v8 (parts/attn8.py) + bf16_proj v7a's router configs (parts/v11.py)
        build(DECODE / "submission_v11.py", "m11_pdl", "attn8", "proj6", "v9", table=tbl, extras=("v10", "v11"),
              picks=picks)
    elif "--v10" in args:  # v9 + the MoE session's E15 / E16 PDL path at the smallest M11 classes (parts/v10.py)
        build(DECODE / "submission_v10.py", "m11_pdl", "attn6", "proj6", "v9", table=tbl, extras=("v10",), picks=picks)
    elif "--v9" in args:  # v7 without L9, with attention v6 and bf16_proj v6 (parts/v9.py)
        build(DECODE / "submission_v9.py", "m11_pf", "attn6", "proj6", "v9", table=tbl, picks=picks)
    elif "--v8" in args:  # v7 + L10 (pooler / unpooler staging, parts/l10.py)
        build(DECODE / "submission_v8.py", "m11_pf", "attn5", "proj4", "v6", table=tbl, extras=("l9", "l10"), picks=picks)
    elif "--v7" in args:  # v6 + L9 (L2 staging of upcoming weights in HBM-idle windows, parts/l9.py)
        build(DECODE / "submission_v7.py", "m11_pf", "attn5", "proj4", "v6", table=tbl, extras=("l9",), picks=picks)
    elif "--v6" in args:
        build(DECODE / "submission_v6.py", "m11_pf", "attn5", "proj4", "v6", table=tbl, picks=picks)
    elif "--pf" in args:
        build(DECODE / "submission_v5.py", "m11_pf", table=tbl)
    else:
        build()

"""Mutation test of the v6 checks (check_v6_cpu.py + check_cuda_syntax.py): plants bugs in v4/r14.py one at a
time, rebuilds submission.py, expects each to fail a check, then restores the clean files.
Run from the project root: .venv/bin/python prefill_layer/local/mutate_v6_checks.py"""
import subprocess

G = "prefill_layer/v6/r14c.py"
clean = open(G).read()
MUTS = {
    "pack nibble positions": ("[12, 28, 8, 24, 4, 20, 0, 16]", "[28, 12, 8, 24, 4, 20, 0, 16]"),
    "pack scale order": (".permute(0, 5, 1, 2, 4, 6, 3).contiguous()  # [nb, kc, c, w, g, ks, half]",
                         ".permute(0, 5, 1, 2, 4, 3, 6).contiguous()  # [nb, kc, c, w, g, ks, half]"),
    "gate/up halves swapped": ("(r >= 8).long() * D_EXPERT", "(r < 8).long() * D_EXPERT"),
    "decode row scale": ("  r1 = hmul2(dec<1>(wd), sg8);", "  r1 = hmul2(dec<1>(wd), sg);"),
    "descriptor SBO": ("(static_cast<uint64_t>(1024 >> 4) << 32)", "(static_cast<uint64_t>(512 >> 4) << 32)"),
    "gather swizzle off": ("((u ^ (p & 7)) * 8 + offs % 8)", "((u) * 8 + offs % 8)"),
    "route seg align": ("start[e + 1] = start[e] + ((cnt[e] + SEG - 1) / SEG) * SEG;",
                        "start[e + 1] = start[e] + cnt[e];"),
    "GEMM1 epilogue column": ("    const int col = c * 32 + w * 8 + g;", "    const int col = c * 32 + w * 8 + q;"),
    "GEMM2 epilogue row": ("    const int r = c * 64 + w * 16 + g;", "    const int r = c * 64 + w * 8 + g;"),
    "epilogue scale uses w2": ('return (gmm["w13_gs"] * R14_OUT_SHIFT).contiguous()', 'return (gmm["w2_gs"] * R14_OUT_SHIFT).contiguous()'),
    "too few tiles": ("max_pieces = -(-slots // tile) + 4 * LOCAL_EXPERTS", "max_pieces = -(-slots // tile) + LOCAL_EXPERTS"),
    "C++ typo": ("  const int4 piece = pieces[pc];", "  const int4 piece = piece[pc];"),
    "ring too shallow": ("  S = S > DEPTH + 1 ? S : DEPTH + 2;", "  S = S > 1 ? S : 2;"),
    "B stage stride": ("if (!no_mma(DIAG)) mma_k64<N>(acc, cur, make_desc(sB0 + rg.si * (N * 128))", "if (!no_mma(DIAG)) mma_k64<N>(acc, cur, make_desc(sB0 + rg.si * B_BYTES)"),
    "load parity flip": ("    if (rg.sl == 0) rg.pl ^= 1u;", "    if (rg.sl == 1) rg.pl ^= 1u;"),
    "tile mode rounding": ("mode ? ((rem + SEG - 1) / SEG) * SEG", "mode ? ((rem + 7) / 8) * 8"),
    # v6 cluster protocol (each line is pinned; its semantics are simulated in sim_cluster_ring.py)
    "empty count not per cluster": ("mbar_init(empty0 + 8 * s, 2 * CL);", "mbar_init(empty0 + 8 * s, 2);"),
    "release own CTA only": ("      else mbar_arrive_cluster<CL>(empty0 + 8 * rg.sr);", "      else mbar_arrive(empty0 + 8 * rg.sr);"),
    "multicast to self only": ("static_cast<uint16_t>((1u << CL) - 1u)", "static_cast<uint16_t>(1u << cluster_rank())"),
    "slice offset ignores rank": ("const uint32_t boff = CL > 1 ? cluster_rank() * slice : 0u;", "const uint32_t boff = 0u;"),
    "no cluster sync after init": ("  if (CL > 1) cluster_sync();  // every CTA's barriers", "  if (CL > 9) cluster_sync();  // every CTA's barriers"),
    "no cluster sync at exit": ("  if (CL > 1) cluster_sync();\n}", "  if (CL > 9) cluster_sync();\n}"),
    "expect only own slice": ("mbar_expect_tx(full0 + 8 * s, WS_BYTES + bbytes);", "mbar_expect_tx(full0 + 8 * s, WS_BYTES + slice);"),
    "mma skipped in production": ("{ return d == 2 || d == 5; }", "{ return d == 2 || d == 6; }"),
    "early order waits too little": ("    wg_wait<DEPTH - 1>();", "    wg_wait<DEPTH>();"),
    "early order reads wrong slot": ("      wv = *reinterpret_cast<const uint4*>(sW + rg.sl * WS_BYTES + woff);", "      wv = *reinterpret_cast<const uint4*>(sW + rg.si * WS_BYTES + woff);"),
    "ping-pong barrier ids swapped": ("named_sync(wgi == 0 ? 3 : 2, 256);", "named_sync(wgi == 0 ? 2 : 3, 256);"),
    "R47 rotate amount": ("rotr(wd, 6) & 0x01C001C0u", "rotr(wd, 5) & 0x01C001C0u"),
    "R47 layout bit": ("(1, 6), (2, 13)", "(1, 6), (2, 12)"),
    "pairs release too early": ("  if ((kc & 1) && kc >= 3) {", "  if ((kc & 1) && kc >= 1) {"),
    "pairs ring 2": ("    S = S > 3 ? S : 3;", "    S = S > 2 ? S : 2;"),
    "pairs second B tile offset": ("smem_u32(sB + s * (2 * bbytes) + bbytes)", "smem_u32(sB + s * (2 * bbytes))"),
    "R53 A swizzle off": ("((unit ^ g) << 4) + 4 * q) = r[j];", "((unit) << 4) + 4 * q) = r[j];"),
    "R53 A row half": ("          const int row = 16 * w + g + 8 * (j & 1);", "          const int row = 16 * w + g + 8 * (j >> 1);"),
    "R53 empty count 5": ("mbar_init(empty0 + 8 * i, 6);", "mbar_init(empty0 + 8 * i, 5);"),
    "R53 odd ring": ("  S &= ~1;            // even (see above)", "  S |= 0;            // even (see above)"),
    "R53 A-empty parity": ("static_cast<uint32_t>(ra - 1) & 1u, err);", "static_cast<uint32_t>(ra) & 1u, err);"),
    "lane-0 wait without syncwarp": ("  ok = __shfl_sync(0xffffffffu, ok, 0);\n  __syncwarp();", "  ok = __shfl_sync(0xffffffffu, ok, 0);"),
    # R54 wide_kernel (pinned lines + gemm_wide port + ring/staging checks + mode-2 tile count)
    "R54 second pair's block": ("2 * nb2 + h", "nb2 + h"),
    "R54 second blob offset": ("wsrc0 + static_cast<size_t>(KC) * WS_BYTES;", "wsrc0;"),
    "R54 empty count 2": ("mbar_init(empty0 + 8 * s, 4);", "mbar_init(empty0 + 8 * s, 2);"),
    "R54 staging overlap": ("sB + (tid >> 8) * (N * 256)", "sB + (tid >> 8) * (N * 128)"),
    "R54 one barrier for both pairs": ("wide(DIAG) ? 2 + (tid >> 8) : 1);", "wide(DIAG) ? 2 : 1);"),
    "R54 half index": ("wide(DIAG) ? ((tid >> 7) & 1)", "wide(DIAG) ? ((tid >> 8) & 1)"),
    "R54 second pair reads block 0": ("    const uint8_t* sWh = h ? sW1 : sW0;", "    const uint8_t* sWh = sW0;"),
    "mode 2 tile cap": ("const int cap = mode == 2 ? 64 : 128;", "const int cap = mode == 2 ? 128 : 128;"),
    "mode 2 too few tiles": ("tile = 64 if mode == 2 else R14_BT", "tile = R14_BT"),
    # R55 (swapped operands; pinned lines + gemm_swap port + 4b 128-row B read + fit)
    "R55 token half offset": ("smem_u32(sB) + m * 8192, sA0", "smem_u32(sB) + m * 4096, sA0"),
    "R55 operands not swapped": ("mma_k64_ss<128>(acc, make_desc(sB0 + s * tstride), make_desc(sA0 + a * DQ_A_BYTES)", "mma_k64_ss<128>(acc, make_desc(sA0 + a * DQ_A_BYTES), make_desc(sB0 + s * tstride)"),
    "R55 slot not 64-row": ("((nsel + 63) & ~63) * 128 : bbytes;", "nsel * 128 : bbytes;"),
    "R55 second WG always idle": ("  const bool active = m == 0 || nsel > 64;", "  const bool active = m == 0;"),
    "R55 up register": ("acc[8 * i + 4 + 2 * r + h] * cs;", "acc[8 * i + 2 + 2 * r + h] * cs;"),
    "R55 token row": ("            const int t = 64 * m + 16 * w + g + 8 * r;", "            const int t = 64 * m + 16 * w + 2 * g + r;"),
    "R55 plain column": ("          const int n = 8 * j + 2 * q;", "          const int n = 8 * j + q;"),
    # R11 (check 6 with fake streams; check_v6_vs_baseline for the twin)
    "R11 no join": ("        torch.cuda.current_stream().wait_stream(state[\"gmm\"][\"deq_stream\"])", "        pass"),
    "R11 fork without wait": ("    side.wait_stream(main)\n", "    pass\n"),
    "R11 decode on main": ("    with torch.cuda.stream(side):", "    with torch.cuda.stream(main):"),
    "R11 also on R14 classes": ("\n            and r14.get(\"cfg\", {}).get(_r14_class(tokens)) is None):", "):"),
    "R11 twin keeps a dequant": ("    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,\n                                  num_warps=4)\n    h1 = ", "    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,\n                                  num_warps=4)\n    _dequant(gmm[\"w2_q\"], gmm[\"w2_sf\"], counts, gmm[\"w2\"])\n    h1 = "),
    "variant cluster 3": ("R14_VARIANTS = ((2, 8, 1, 1), ", "R14_VARIANTS = ((2, 8, 1, 3), "),
}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) >= 1, name  # every copy (base, timed and early paths share lines)
        open(G, "w").write(clean.replace(a, b))
        subprocess.run([".venv/bin/python", "prefill_layer/local/make_v6.py"], check=True, capture_output=True)
        caught = False
        for chk in ("prefill_layer/local/check_v6_cpu.py", "prefill_layer/local/check_cuda_syntax.py",
                    "prefill_layer/local/check_v6_vs_baseline.py"):
            r = subprocess.run([".venv/bin/python", chk], capture_output=True, text=True)
            caught |= r.returncode != 0
        results[name] = "CAUGHT" if caught else "MISSED"
finally:
    open(G, "w").write(clean)
    subprocess.run([".venv/bin/python", "prefill_layer/local/make_v6.py"], check=True, capture_output=True)
for k, v in results.items():
    print(f"{v:7s} {k}")

"""Mutation test of the v5 checks (check_v5_cpu.py + check_cuda_syntax.py): plants bugs in v4/r14.py one at a
time, rebuilds submission.py, expects each to fail a check, then restores the clean files.
Run from the project root: .venv/bin/python prefill_layer/local/mutate_v4_checks.py"""
import subprocess

G = "prefill_layer/v5/r14b.py"
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
    "too few tiles": ("max_pieces = -(-slots // R14_BT) + 4 * LOCAL_EXPERTS", "max_pieces = -(-slots // R14_BT) + LOCAL_EXPERTS"),
    "C++ typo": ("  const int4 piece = pieces[pc];", "  const int4 piece = piece[pc];"),
    "ring too shallow": ("  S = S > DEPTH + 1 ? S : DEPTH + 2;", "  S = S > 1 ? S : 2;"),
    "B stage stride": ("make_desc(sB0 + rg.si * (N * 128))", "make_desc(sB0 + rg.si * B_BYTES)"),
    "load parity flip": ("    if (rg.sl == 0) rg.pl ^= 1u;", "    if (rg.sl == 1) rg.pl ^= 1u;"),
    "tile mode rounding": ("mode ? ((rem + SEG - 1) / SEG) * SEG", "mode ? ((rem + 7) / 8) * 8"),
}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) == 1, name
        open(G, "w").write(clean.replace(a, b))
        subprocess.run([".venv/bin/python", "prefill_layer/local/make_v5.py"], check=True, capture_output=True)
        caught = False
        for chk in ("prefill_layer/local/check_v5_cpu.py", "prefill_layer/local/check_cuda_syntax.py"):
            r = subprocess.run([".venv/bin/python", chk], capture_output=True, text=True)
            caught |= r.returncode != 0
        results[name] = "CAUGHT" if caught else "MISSED"
finally:
    open(G, "w").write(clean)
    subprocess.run([".venv/bin/python", "prefill_layer/local/make_v5.py"], check=True, capture_output=True)
for k, v in results.items():
    print(f"{v:7s} {k}")

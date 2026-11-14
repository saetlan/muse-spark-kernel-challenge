"""Mutation test of the DG12-RS checks: plants one bug at a time in dgrs.py, runs emu_dgrs.py (quick) and
check_dgrs_syntax.py, expects a failure, restores the clean file.
Run: .venv/bin/python shared/gemm/dg12rs/mutate_dgrs.py"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parents[2] / ".venv" / "bin" / "python")
F = HERE / "dgrs.py"
clean = F.read_text()
MUTS = {
    "decode rotate": ("__funnelshift_l(q, q, 4)", "__funnelshift_l(q, q, 3)"),
    "decode H bias": ("(q & 0x87878787u) | 0x38383838u", "(q & 0x87878787u) | 0x30303030u"),
    "decode selector": ("__byte_perm(l0, ha, 0x7362u)", "__byte_perm(l0, ha, 0x5140u)"),
    "L slot for steps 2-3": ("const uint4& l = j < 2 ? l01 : l23;", "const uint4& l = j < 2 ? l23 : l01;"),
    "L word pair": ("word(l, 2 * (j & 1) + 1)", "word(l, 2 * (j & 1))"),
    "slot stride": ("sw + woff + 1024);", "sw + woff + 1536);"),
    "row tile of warp": ("((4 * c + w) * 3 * 32 + lane) * 16;", "((4 * w + c) * 3 * 32 + lane) * 16;"),
    "block offset": ("static_cast<size_t>(rb) * 8) * 1536;", "static_cast<size_t>(rb) * 16) * 1536;"),
    "chunk stride": ("static_cast<size_t>(N / 16) * 1536;  // one chunk", "static_cast<size_t>(N / 8) * 1536;  // one chunk"),
    "TMA x coordinate": ("&xmap, (kc0 + i) * 64, tt * BM,", "&xmap, (kc0 + i) * 32, tt * BM,"),
    "TMA box": ("const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(bm)};", "const cuuint32_t box[2] = {32, static_cast<cuuint32_t>(bm)};"),
    "descriptor SBO": ("(static_cast<uint64_t>(1024 >> 4) << 32)", "(static_cast<uint64_t>(512 >> 4) << 32)"),
    "split size": ("const int per = (KC + splits - 1) / splits;", "const int per = KC / splits;"),
    "tile row +8": ("mtile[t * 128 + r0 + 8] = acc[4 * j + 2 + h];", "mtile[t * 128 + r0 + 8] = acc[4 * j + h];"),
    "tile row base": ("const int r0 = 64 * c + 16 * w + g;  // rows", "const int r0 = 64 * c + 8 * w + g;  // rows"),
    "rank row share": ("  const int rlo = rank * L.rs;", "  const int rlo = rank * (L.rs - 1);"),
    "token lanes (math threads only)": ("      const int TL = rn > 0 ? TM / rn : 0;", "      const int TL = rn > 0 ? THREADS / rn : 0;"),
    "token step": ("          const int t = t0 + TL * u;", "          const int t = t0 + u;"),
    "escape sign": ("c = fmaf(-xv[u], f12m_wdec(d.y >> 8), c);", "c = fmaf(xv[u], f12m_wdec(d.y >> 8), c);"),
    "escape x token": ("static_cast<size_t>(live ? tok0 + t : 0) * K;", "static_cast<size_t>(live ? tok0 : 0) * K;"),
    "scale exponent": ("return __uint_as_float((base + 15u) << 23); }", "return __uint_as_float((base + 16u) << 23); }"),
    "full count": ("mbar_init(full0 + 8 * s, 1);  // thread 0", "mbar_init(full0 + 8 * s, 2);  // thread 0"),
    "tx bytes": ("((DIAG & 4) ? 0u : X_BYTES);", "((DIAG & 4) ? 0u : X_BYTES / 2);"),
    "release wrong chunk": ("mbar_arrive(empty0 + 8 * ((i - NP) % S));", "mbar_arrive(empty0 + 8 * ((i - 1) % S));"),
    "pair chunk start": ("      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == p);", "      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == 0);"),
    "pair index": ("const int mw = (tid >> 7) - 1, c = mw & 1, p = mw >> 1;", "const int mw = (tid >> 7) - 1, c = mw & 1, p = mw & 1;"),
    "pair tile": ("      float* mtile = tile + p * (BM * 128);", "      float* mtile = tile;"),
    "pairs not summed": ("tile[idx] += tile[BM * 128 + idx];", "tile[idx] += 0.f;"),
    "red 1 last arriver off by one": ("1) == splits - 1;", "1) == splits;"),
    "red 1 no reset": ("          atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset", "          (void)0;  // self-reset"),
    "red 1 partial index": ("(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N", "(static_cast<size_t>(sl) * Mp + (idx >> 7)) * N"),
    "red 1 everyone finalizes": ("(!glob || s_last)) {", "(!glob || true)) {"),
    "pack order": ("return p.permute(1, 0, 2, 3, 4).contiguous()", "return p.contiguous()"),
    # v5: producer precompute, finalize, layout, barriers
    "pre list offset": ("const int off = ne + inc - static_cast<int>(cnt);", "const int off = ne + inc;"),
    "pre scan lanes": ("      if (lane >= o) inc += nb;", "      if (lane > o) inc += nb;"),
    "pre escape word": ("word(q0, e + 1) : word(q1, e - 3)", "word(q0, e) : word(q1, e - 3)"),
    "pre row split": ("  const int wlo = pw * RW;", "  const int wlo = pw * (RW - 1);"),
    "pre tag base": ("static_cast<uint32_t>(ri) | (base << 8);", "static_cast<uint32_t>(ri) | (base << 7);"),
    "pre zeroing": ("ri += 32) sCorr[t * cs + ri] = 0.f;", "ri += 32) sCorr[t * cs + ri + 1] = 0.f;"),
    "pre live token": ("    const bool live = tok0 + t < M;", "    const bool live = t < M;"),
    "pre in warp 0": ("    } else if (tid >= 32 && (DIAG & 49) == 0) {", "    } else if (tid >= 1 && (DIAG & 49) == 0) {"),
    "finalize corr stride": ("sCorr[(t0 + TL * u) * L.cs + ri]", "sCorr[(t0 + TL * u) * L.rs + ri]"),
    "finalize drops corr": ("__float2bfloat16_rn(fmaf(sum[u], scale, sCorr[", "__float2bfloat16_rn(fmaf(sum[u], scale, 0.f * sCorr["),
    "layout escape area": ("  L.corr = L.esc + PRE_WARPS * ESC_CAP * 8;", "  L.corr = L.esc + PRE_WARPS * ESC_CAP * 4;"),
    "layout ring not even": ("  if (math == 4) S &= ~1;  // even", "  if (math == 8) S &= ~1;  // even"),
    "producer misses a cluster barrier": ("        cluster_sync();\n        cluster_sync();\n      }", "        cluster_sync();\n      }"),
    "producer never arrives": ("      bar_arrive_na(2, THREADS);  // this CTA", "      (void)0;  // this CTA"),
}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) >= 1, name
        F.write_text(clean.replace(a, b))
        caught = False
        for args in ((str(HERE / "emu_dgrs.py"), "quick"), (str(HERE / "check_dgrs_syntax.py"),)):
            r = subprocess.run([PY, *args], capture_output=True, text=True)
            if r.returncode != 0:
                caught = True
                break
        results[name] = "CAUGHT" if caught else "MISSED"
finally:
    F.write_text(clean)
for k, v in results.items():
    print(f"{v:7s} {k}")
sys.exit(1 if "MISSED" in results.values() else 0)

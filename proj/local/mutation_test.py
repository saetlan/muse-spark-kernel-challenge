"""Mutation test of proj's local checks: plant one bug at a time in a temporary copy of proj/
(host launch code, dispatch, lab gate, or a kernel's emulated math) and confirm that the checks
(make_v1 + check_v1, dryrun_lab_e1, make_lab + dryrun_lab_e2) fail on every one.

Limit: Triton kernel *bodies* can't run on the Mac (no Triton wheels); they are checked only on
the GPU, by the lab's FP64 gate before any config is used. Bugs in them are simulated here as
bugs in the emulation (last group below).

Run: .venv/bin/python proj/local/mutation_test.py
"""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
PY = sys.executable

MUTATIONS = [
    ("fma: K slice too short", "submission.py", "BK=k // split, ATOMIC", "BK=k // split // 2, ATOMIC"),
    ("fma: split 2 stores instead of adding", "submission.py", "    atomic = split > 1\n    if atomic:\n        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)\n    else:\n        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)\n    _fma_kernel",
     "    atomic = split > 2\n    if atomic:\n        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)\n    else:\n        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)\n    _fma_kernel"),
    ("fma: BF16 output left in FP32", "submission.py", "        M=m, BN=c[\"BN\"], BK=k // split, ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c[\"warps\"])\n    if atomic and not fp32_out:\n        out = out.to(torch.bfloat16)",
     "        M=m, BN=c[\"BN\"], BK=k // split, ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c[\"warps\"])\n    if False:\n        out = out.to(torch.bfloat16)"),
    ("triton: wrong split-K length", "submission.py", "    kps = k // split\n", "    kps = k // split - 64\n"),
    ("triton: grid too small", "submission.py", "_swap_kernel[(triton.cdiv(n, bn), split)]", "_swap_kernel[(n // bn - 1, split)]"),
    ("dispatch: router FP32 threshold", "submission.py", "ROUTER_FP32_MAX_TOKENS = 8", "ROUTER_FP32_MAX_TOKENS = 9"),
    ("dispatch: wrong CHOICE key", "submission.py", "CHOICE.get((layer, x.shape[0]))", "CHOICE.get((layer, 1))"),
    ("fma: one token short", "submission.py", "        M=m, BN=c[\"BN\"], BK=k // split,", "        M=m - 1, BN=c[\"BN\"], BK=k // split,"),
    ("lab gate: accepts anything", "lab/sweep.py", "    return rel <= 0.5 * tol and row <= 2.5 * tol and pr <= 0.5e-3", "    return True"),
    # Two independent probe guards (the probe-row term and the all-probe input): remove both.
    ("lab gate: probe checks removed", "lab/sweep.py", [
        ("    probe = ((x != 0).sum(-1) == 1).nonzero().flatten()", "    probe = ((x != 0).sum(-1) == 0).nonzero().flatten()"),
        ("    if probe_all:\n        probes[:] = True", "    if False:\n        probes[:] = True")]),
    ("kernel math: fma drops the last K split", "local/dryrun_lab_e1.py", "    for s in range(splits):\n        part = (x[:, s * BK:(s + 1) * BK].double()",
     "    for s in range(max(1, splits - 1)):\n        part = (x[:, s * BK:(s + 1) * BK].double()"),
    ("kernel math: fma router in BF16 (precision loss)", "local/dryrun_lab_e1.py",
     "        part = (x[:, s * BK:(s + 1) * BK].double() @ w[:N, s * BK:(s + 1) * BK].double().T).float()",
     "        part = (x[:, s * BK:(s + 1) * BK].double() @ w[:N, s * BK:(s + 1) * BK].to(torch.bfloat16).double().T).float()"),
    ("library: mm_bf16 given W instead of W.T", "submission.py", "flashinfer.mm_bf16(x, w.T, out_dtype", "flashinfer.mm_bf16(x, w, out_dtype"),
    ("library: DeepGEMM native flag inverted", "submission.py", 'deep_gemm.use_deterministic_algorithms(c["native"])', 'deep_gemm.use_deterministic_algorithms(not c["native"])'),
    ("p9 layout: fragment halves swapped", "p9/p9_layout.py", "v = v.permute(0, 3, 2, 5, 4, 1, 6)", "v = v.permute(0, 3, 2, 5, 1, 4, 6)"),
    ("p9 dispatch: router not hi/lo", "local/make_p9.py", '    hilo = layer == "router"\n    f12 = bool', '    hilo = False\n    f12 = bool'),
    ("p22 CUDA decode: wrong byte interleave", "p9/p9.py", "__byte_perm(loa, hia, 0x5140u)", "__byte_perm(loa, hia, 0x4150u)"),
    ("p22 CUDA decode: exponent bit dropped", "p9/p9.py", "const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x7F7F7F7Fu);", "const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x3F7F7F7Fu);"),
    ("p22 encode: escape raw byte kept", "p9/p22_layout.py", "    r = torch.where(esc, torch.zeros_like(r), r)\n", "\n"),
    ("kernel math: hi + lo without lo", "local/dryrun_lab_e1.py", "        return xs.float() @ hi.float().T + xs.float() @ lo.float().T", "        return xs.float() @ hi.float().T"),
    # F12V (12-bit m = 1 GEMV): CUDA text (parsed by check_f12v), packer, records, host dispatch.
    ("f12v CUDA decode: wrong byte selector", "f12v/f12v.py", "f[1] = __uint_as_float(__byte_perm(l0, h1, 0x5144u));", "f[1] = __uint_as_float(__byte_perm(l0, h1, 0x5044u));"),
    ("f12v CUDA decode: sign bit masked off", "f12v/f12v.py", "const uint32_t h2 = (r4 & 0x00870087u) | 0x00380038u;", "const uint32_t h2 = (r4 & 0x00070087u) | 0x00380038u;"),
    ("f12v CUDA decode: wrong rotate", "f12v/f12v.py", "__funnelshift_l(q, q, 4)", "__funnelshift_l(q, q, 3)"),
    ("f12v CUDA index: lanes swapped", "f12v/f12v.py", "{ return ((row * kbs + kb) * 3 + slot) * 32 + lane; }", "{ return ((row * kbs + kb) * 3 + slot) * 32 + (lane ^ 1); }"),
    ("f12v CUDA index: x groups swapped", "f12v/f12v.py", "{ return kb * 128 + lane * 4 + g; }", "{ return kb * 128 + lane * 4 + (g ^ 1); }"),
    ("f12v CUDA index: L words of the wrong group", "f12v/f12v.py", "{ return 2 * (g & 1) + h; }", "{ return 2 * (g & 1) + (1 - h); }"),
    ("f12v CUDA scale off by 2x", "f12v/f12v.py", "__uint_as_float((base + 15u) << 23)", "__uint_as_float((base + 16u) << 23)"),
    ("f12v CUDA x conversion", "f12v/f12v.py", "  lo = __uint_as_float(w << 16);", "  lo = __uint_as_float(w << 15);"),
    ("f12v CUDA ring reuses a slot early", "f12v/f12v.py", "    if (s + D < NS) load(slot, s + D);", "    if (s + D < NS) load(slot, s + D - 1);"),
    ("f12v CUDA escape w_dec not removed", "f12v/f12v.py", "        sum = fmaf(-xk, wdec, sum);\n", "\n"),
    ("f12v layout: sign of weights 4-7 misplaced", "f12v/f12v_layout.py", "(s[..., 4 + j] << (8 * j + 3))", "(s[..., 4 + j] << (8 * j + 4))"),
    ("f12v layout: escape mantissa kept", "f12v/f12v_layout.py", "torch.where(esc, zero, m), esc, b, base)", "m, esc, b, base)"),
    ("f12v records: count in the wrong bits", "f12v/f12v_layout.py", "    rec[:, 0] = enc[\"base\"] | (count << 8)", "    rec[:, 0] = enc[\"base\"] | (count << 9)"),
    ("f12v host: wrong instance (NS)", "local/make_f12v.py", 'inst = (c["R"], c["RG"], c["KSW"], k // 1024 // c["KSW"], c["D"])', 'inst = (c["R"], c["RG"], c["KSW"], k // 2048 // c["KSW"], c["D"])'),
    ("f12v host: no fallback without planes", "local/make_f12v.py", "    if pl is None or x.shape[0] != 1:", "    if x.shape[0] != 1:"),
    # F12M (12-bit weights in mma fragment order, m = 8..32): CUDA text (parsed by check_f12m), layout, host.
    ("f12m CUDA decode: wrong byte selector", "f12m/f12m.py", "return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),", "return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x6372u),"),
    ("f12m CUDA decode: sign bit masked off", "f12m/f12m.py", "const uint32_t ha = (q & 0x87878787u) | 0x38383838u;", "const uint32_t ha = (q & 0x07878787u) | 0x38383838u;"),
    ("f12m CUDA index: x halves swapped", "f12m/f12m.py", "{ return tok * k8 + 8 * c + 2 * t + h; }", "{ return tok * k8 + 8 * c + 2 * t + (1 - h); }"),
    ("f12m CUDA index: C rows of the wrong half", "f12m/f12m.py", "{ return 16 * (rg * rtw + i) + g + 8 * (r >> 1); }", "{ return 16 * (rg * rtw + i) + g + 8 * (r & 1); }"),
    ("f12m CUDA index: BF16 steps swapped", "f12m/f12m.py", "{ return (tile * kss + 4 * c + j) * 32 + lane; }", "{ return (tile * kss + 4 * c + (j ^ 1)) * 32 + lane; }"),
    ("f12m layout: escape rows of the wrong half", "f12m/f12m_layout.py", "+ g + 8 * (j & 1)      # [RT, 1, 32, 8]", "+ g + 8 * (j >> 1)      # [RT, 1, 32, 8]"),
    ("f12m host: wrong instance (NS)", "local/make_f12m.py", 'inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"], c["D"], c["F12"])', 'inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 128 // c["KSW"], c["D"], c["F12"])'),
    # WG (wgmma GEMM with in-cluster split-K, m = 128/256): CUDA text (check_wg), host dispatch.
    ("wg CUDA: k16 step offset in A", "wg/wg.py", "{ return 8192 * w + 32 * j; }", "{ return 8192 * w + 16 * j; }"),
    ("wg CUDA: accumulator rows", "wg/wg.py", "{ return 64 * w + 16 * q + g + 8 * ((e >> 1) & 1); }", "{ return 64 * w + 16 * q + g + 8 * (e & 1); }"),
    ("wg CUDA: uneven split-K ranges", "wg/wg.py", "{ return r * chunks / ks; }", "{ return r * (chunks / ks); }"),
    ("wg CUDA: producer parity", "wg/wg.py", "mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);\n        mbar_expect_tx(&full[s], static_cast<uint32_t>(STAGE_BYTES));", "mbar_wait(&empty[s], static_cast<uint32_t>((i / ST) & 1), failed);\n        mbar_expect_tx(&full[s], static_cast<uint32_t>(STAGE_BYTES));"),
    ("wgf CUDA: escape x from the wrong swizzle chunk", "wg/wg.py", "{ return tok * 128 + ((((2 * kk) >> 4) ^ (tok & 7)) << 4) + ((2 * kk) & 15); }", "{ return tok * 128 + ((((2 * kk) >> 4) ^ (tok & 3)) << 4) + ((2 * kk) & 15); }"),
    ("wgf CUDA: decode sign bit masked", "wg/wg.py", "  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;\n  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),\n                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));\n}\n\n__device__ __forceinline__ void wgmma_rs_128", "  const uint32_t hb = (r4 & 0x07878787u) | 0x38383838u;\n  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),\n                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));\n}\n\n__device__ __forceinline__ void wgmma_rs_128"),
    ("wgf CUDA: warp tiles overlap", "wg/wg.py", "{ return 8 * rt + 4 * w + q; }", "{ return 8 * rt + 2 * w + q; }"),
    ("wgf host: wrong instance (D)", "local/make_wg.py", 'getattr(WG_MOD, wgf_name((c["KS"], c["ST"], c["TT"], c["D"])))', 'getattr(WG_MOD, wgf_name((c["KS"], c["ST"], c["TT"], 1)))'),
    # RG (router hi + lo mma.sync kernel): CUDA text (check_rg), layout, host.
    ("rg CUDA: ldmatrix k chunk", "rg/rg.py", "{ return 2 * j + ((lane >> 3) & 1); }", "{ return 2 * j + ((lane >> 4) & 1); }"),
    ("rg CUDA: ldmatrix tokens", "rg/rg.py", "{ return tb + 16 * p + (lane & 7) + 8 * ((lane >> 3) >> 1); }", "{ return tb + 16 * p + (lane & 7) + 8 * ((lane >> 3) & 1); }"),
    ("rg CUDA: C rows", "rg/rg.py", "{ return 16 * rt + g + 8 * (r >> 1); }", "{ return 16 * rt + g + 8 * (r & 1); }"),
    ("rg CUDA: weight step", "rg/rg.py", "{ return ((tile * kss + s) * 2 + h) * 32 + lane; }", "{ return ((tile * kss + s + 1) * 2 + h) * 32 + lane; }"),
    ("rg layout: lo half lost", "rg/rg_layout.py", "    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)\n", "    lo = (0 * w).to(torch.bfloat16)\n"),
    ("rg host: no fallback without a module", "local/make_rg.py", "    if RG_MOD is None or layer != \"router\" or m > rg_mt(inst):", "    if layer != \"router\" or m > rg_mt(inst):"),
    # SX (m = 32/128/256 12-bit mma.sync GEMM): CUDA text (check_sx), layout, host.
    ("sx CUDA: ldmatrix k chunk", "sx/sx.py", "{ return 2 * j + ((lane >> 3) & 1); }", "{ return 2 * j + ((lane >> 4) & 1); }"),
    ("sx CUDA: C rows", "sx/sx.py", "{ return 16 * (rg * rtw + r) + g + 8 * (q >> 1); }", "{ return 16 * (rg * rtw + r) + g + 8 * (q & 1); }"),
    ("sx CUDA: escape flag index", "sx/sx.py", "{ return tile * chs + c; }", "{ return tile * chs + (c ^ 1); }"),
    ("sx CUDA: weight slot", "sx/sx.py", "{ return ((tile * chs + c) * 3 + slot) * 32 + lane; }", "{ return ((tile * chs + c) * 3 + (2 - slot)) * 32 + lane; }"),
    ("sx CUDA: epilogue rows", "sx/sx.py", "{ return 8 * (idx % groups); }", "{ return 8 * ((idx / 2) % groups); }"),
    ("sx CUDA: decode high bytes", "sx/sx.py", "const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;", "const uint32_t hb = (r4 & 0x87878787u) | 0x30303030u;"),
    ("sx CUDA: overlapping chunks", "sx/sx.py", "{ return r * (ns * ksw) + ks * ns + i; }", "{ return r * ns + ks * ns + i; }"),
    ("sx CUDA: token groups", "sx/sx.py", "{ return (warp / ksw) % tg; }", "{ return warp % tg; }"),
    ("sx CUDA: resident wait", "sx/sx.py", "            cp_wait<1>();\n            __syncthreads();                               // chunk 0 landed",
     "            cp_wait<2>();\n            __syncthreads();                               // chunk 0 landed"),
    ("sx layout: flags miss the 7th escape", "sx/sx_layout.py", "    for j in range(7):\n        sel = count > j", "    for j in range(6):\n        sel = count > j"),
    ("sx layout: bf16 fragment order", "sx/sx_layout.py", ".permute(0, 3, 2, 5, 4, 1, 6)", ".permute(0, 3, 2, 5, 1, 4, 6)"),
    ("sxw CUDA: accumulator rows", "sx/sx.py", "{ return 64 * wrg + 16 * q + g + 8 * ((e >> 1) & 1); }", "{ return 64 * wrg + 16 * q + g + 8 * ((e >> 2) & 1); }"),
    ("sxw CUDA: accumulator tokens", "sx/sx.py", "{ return 128 * nb + 8 * (e >> 2) + 2 * t + (e & 1); }", "{ return 128 * nb + 8 * (e >> 2) + 2 * t; }"),
    ("sxw CUDA: B offset of the 2nd token half", "sx/sx.py", "{ return ks * n * 128 + nb * 16384 + 32 * j; }", "{ return ks * n * 128 + nb * 8192 + 32 * j; }"),
    ("sxw CUDA: weight ring slots overlap", "sx/sx.py", "{ return (warp * wsl + slot) * 1536; }", "{ return (warp * wsl + slot) * 1024; }"),
    ("sxw CUDA: tile of a warp", "sx/sx.py", "{ return (cl * wgs + wrg) * 4 + q; }", "{ return (cl * wgs + wrg) * 4 + (q ^ 1); }"),
    ("sxw CUDA: fewer x slots", "sx/sx.py", "constexpr int XSL = P + 2, WSL = P + 1;", "constexpr int XSL = P + 1, WSL = P + 1;"),
    ("sxw CUDA: escape range", "sx/sx.py", "if (kk >= k0 && kk < k0 + 64 * NS) {", "if (kk >= k0 && kk <= k0 + 64 * NS) {"),
    ("sx host: planes of another layer", "local/make_sx.py", '    pl = state.get("sx", {}).get(layer)\n', '    pl = state.get("sx", {}).get("pooler")\n'),
    ("wg host: no fallback without a module", "local/make_wg.py", "    if WG_MOD is None or m > 128 * c[\"TT\"]", "    if m > 128 * c[\"TT\"]"),
]

CHECKS = [
    [PY, "local/make_lab.py", "lab/e1.py", "lab_e1.py"],
    [PY, "local/dryrun_lab_e1.py"],
    [PY, "local/make_v1.py"],
    [PY, "local/check_v1.py"],
    [PY, "local/make_v2.py"],
    [PY, "local/check_v1.py", "submission_v2.py"],
    [PY, "local/make_lab.py", "--base", "submission_v1.py", "lab/sweep.py", "lab/e2.py", "lab_e2.py"],
    [PY, "local/dryrun_lab_e2.py"],
    [PY, "local/make_lab.py", "--base", "submission_v2.py", "lab/sweep.py", "lab/e3.py", "lab_e3.py"],
    [PY, "local/dryrun_lab_e3.py"],
    [PY, "local/check_p9.py"],
    [PY, "local/make_v3.py"],
    [PY, "local/check_v1.py", "submission_v3.py"],
    [PY, "local/make_p9.py", "--base", "submission_v3.py", "--lab", "submission_p9lab.py"],
    [PY, "local/make_lab.py", "--base", "submission_p9lab.py", "lab/sweep.py", "lab/e4.py", "lab_e4.py"],
    [PY, "local/dryrun_lab_e4.py"],
    [PY, "local/check_p22.py"],
    [PY, "local/check_f12v.py"],
    [PY, "local/make_v4.py"],
    [PY, "local/make_f12v.py", "--base", "submission_v4.py", "--lab", "submission_f12vlab.py"],
    [PY, "local/make_lab.py", "--base", "submission_f12vlab.py", "lab/sweep.py", "lab/e8.py", "lab_e8.py"],
    [PY, "local/dryrun_lab_e8.py"],
    [PY, "local/check_f12m.py"],
    [PY, "local/make_f12m.py", "--base", "submission_f12vlab.py", "--lab", "submission_f12mlab.py"],
    [PY, "local/make_lab.py", "--base", "submission_f12mlab.py", "lab/sweep.py", "lab/e8.py", "lab/e9.py", "lab_e9.py"],
    [PY, "local/dryrun_lab_e9.py"],
    [PY, "local/make_v5.py"],
    [PY, "local/check_v1.py", "submission_v5.py"],
    [PY, "local/make_lab.py", "--base", "submission_v5.py", "lab/sweep.py", "lab/e10.py", "lab_e10.py"],
    [PY, "local/dryrun_lab_e10.py"],
    [PY, "local/check_wg.py"],
    [PY, "local/make_wg.py", "--base", "submission_v5.py", "--lab", "submission_wglab.py"],
    [PY, "local/make_lab.py", "--base", "submission_wglab.py", "lab/sweep.py", "lab/e10.py", "lab/e11.py", "lab_e11.py"],
    [PY, "local/dryrun_lab_e11.py"],
    [PY, "local/make_v7a.py"],
    [PY, "local/check_v1.py", "submission_v7a.py"],
    [PY, "local/make_wg.py", "--base", "submission_v7a.py", "--instances", "LAB", "--wgf", "WGF_LAB", "--lab", "submission_wgflab.py"],
    [PY, "local/make_lab.py", "--base", "submission_wgflab.py", "lab/sweep.py", "lab/e10.py", "lab/e11.py", "lab/e12.py", "lab_e12.py"],
    [PY, "local/dryrun_lab_e12.py"],
    [PY, "local/make_wg.py", "--base", "submission_v7a.py", "--instances", "LAB_DEEP", "--lab", "submission_wgdeep.py"],
    [PY, "local/make_lab.py", "--base", "submission_wgdeep.py", "lab/sweep.py", "lab/e10.py", "lab/e11.py", "lab/e13.py", "lab_e13.py"],
    [PY, "local/dryrun_lab_e13.py"],
    [PY, "local/check_rg.py"],
    [PY, "local/make_rg.py", "--base", "submission_v7a.py", "--lab", "submission_rglab.py"],
    [PY, "local/make_lab.py", "--base", "submission_rglab.py", "lab/sweep.py", "lab/e14.py", "lab_e14.py"],
    [PY, "local/dryrun_lab_e14.py"],
    [PY, "local/check_sx.py"],
    [PY, "local/make_sx.py", "--base", "submission_v7a.py", "--lab", "submission_sxlab.py"],
    [PY, "local/make_lab.py", "--base", "submission_sxlab.py", "lab/sweep.py", "lab/e15.py", "lab_e15.py"],
    [PY, "local/dryrun_lab_e15.py"],
    [PY, "local/make_sx.py", "--base", "submission_v7a.py", "--instances", "NONE", "--w-instances", "W_LAB", "--lab", "submission_sxwlab.py"],
    [PY, "local/make_lab.py", "--base", "submission_sxwlab.py", "lab/sweep.py", "lab/e16.py", "lab_e16.py"],
    [PY, "local/dryrun_lab_e16.py"],
]


WORKERS = 6
ENV = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")   # child processes of this local test only


def run_checks(root):
    for cmd in CHECKS:
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=ENV)
        if r.returncode != 0:
            return Path(cmd[1]).name
    return None


def copy_proj(dst):
    for name in ("submission.py", "baseline.py", "lab", "local", "p9", "f12v", "f12m", "wg", "rg", "sx"):
        src = PROJ / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dst / name)


def mutate(tmp, idx, mutation):
    name, rel, *edit = mutation
    root = Path(tmp) / f"m{idx}"
    root.mkdir()
    copy_proj(root)
    path = root / rel
    text = path.read_text()
    pairs = edit[0] if isinstance(edit[0], list) else [tuple(edit)]
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text)
    return name, run_checks(root)


def main():
    for name, rel, *edit in MUTATIONS:   # every pattern must match exactly once before anything runs
        text = (PROJ / rel).read_text()
        for old, new in (edit[0] if isinstance(edit[0], list) else [tuple(edit)]):
            if text.count(old) != 1:
                raise SystemExit(f"mutation {name!r}: pattern found {text.count(old)} times in {rel}")
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean"
        clean.mkdir()
        copy_proj(clean)
        assert run_checks(clean) is None, "the checks fail on the unmutated code"
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda a: mutate(tmp, a[0], a[1]), enumerate(MUTATIONS)))
        missed = []
        for name, caught in results:
            print(f"{'caught by ' + caught if caught else 'MISSED':28s} {name}", flush=True)
            if not caught:
                missed.append(name)
        if missed:
            raise SystemExit(f"{len(missed)} mutation(s) missed: {missed}")
        print(f"ok: all {len(MUTATIONS)} planted bugs caught")


if __name__ == "__main__":
    main()

"""Static checks of cuda_w1.py against the PTX ISA (9.7.17) / CUTLASS rules and its emulator:
1. tile constants equal emulate_cuda_w1.py's; slots 16 B aligned; dynamic + static SMEM <= 227 KB per variant;
2. gdesc packs start >> 4 into bits 0-13, LBO >> 4 into 16-29, SBO >> 4 into 32-45, nothing else (base
   offset 0, swizzle 0 = none), identical in the emulator;
3. the wgmma asm is CUTLASS's MMA_64x64x32_F32E4M3E4M3_RS_TN form: 32 f32 accumulators, A {%32..%35},
   descriptor %36, scale-d predicate from %37, imm-scale-a / -b 1;
4. consumer loop order per stage: full wait -> proxy fence -> (operand fences on the accumulator and the A
   fragments, wgmma.fence, 2 wgmma, commit, wait 0, the same operand fences) for Q.K, then for P.V, then the empty arrive; masks / softmax only
   between the two waits;
5. transposer: proxy fence after its Vt stores, before its full arrive; barrier counts match arrivals:
   land 32 (copier lanes, cp.async noinc), full 2 (warps 9, 10, lane 0), empty 8 (consumer warps, lane 0);
6. 352 threads = 11 warps: consumer warpgroups are warps 0-3 and 4-7 (warp rank multiple of 4).
Run: .venv/bin/python attention/local/check_w1_static.py"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ns = {}
exec(open(os.path.join(HERE, "cuda_w1.py")).read(), ns)
SRC = ns["CUDA_W1_SRC"]
import emulate_cuda_w1 as emu  # noqa: E402

bad = 0


def check(ok, what):
    global bad
    print(("ok   " if ok else "FAIL ") + what)
    bad += not ok


const = {}
for decl in re.findall(r"constexpr int ([^;]+);", SRC):
    for part in decl.split(","):
        m = re.fullmatch(r"\s*(\w+) = (\d+)\s*", part)
        if m:
            const[m.group(1)] = int(m.group(2))
for k in ("KC_HEAD", "SBO_K", "LBO_K", "VS_PITCH", "SBO_V", "LBO_V"):
    check(const.get(k) == getattr(emu, k), f"{k} = {const.get(k)} (emulator {getattr(emu, k)})")
slot = 2 * const["KC_HEAD"] + 64 * const["VS_PITCH"] + 2 * 8 * const["SBO_V"]
check(slot == emu.SLOT and slot % 16 == 0 and (2 * const["KC_HEAD"]) % 16 == 0 and (2 * const["KC_HEAD"] + 64 * const["VS_PITCH"]) % 16 == 0,
      f"slot {slot} B, tiles 16 B aligned")
variants = [tuple(int(x) for x in v.split(",")) for v in re.findall(r"X\(([^)]*)\)", re.search(r"#define OKC_W1_VARIANTS\(X\) (.*)", SRC).group(1))]
for st_, sbov, wf in variants:
    total = st_ * (2 * const["KC_HEAD"] + 64 * const["VS_PITCH"] + 2 * 8 * sbov) + 2 * st_ * 64 * 4 + 3 * st_ * 8
    check(total <= 227 * 1024 and sbov % 16 == 0 and wf in (0, 1), f"S{st_} SBOV {sbov} WF {wf}: {total} B shared memory")
check("const size_t smem = static_cast<size_t>(stages) * (okc_w1::OFF_VT + 2 * 8 * sbov) + static_cast<size_t>(2 * stages) * 64 * 4;" in SRC
      and "S * (okc_w1::OFF_VT + 2 * 8 * V) + 2 * S * 64 * 4" in SRC and "int* ptr_ring = reinterpret_cast<int*>(sm + STAGES * SLOTB);" in SRC
      and "constexpr int SLOTB = OFF_VT + 2 * VTH;" in SRC and "constexpr int VTH = 8 * SBOV;" in SRC,
      "host dynamic size = slots + page-entry ring, ring after the slots")
gd = re.search(r"uint64_t gdesc\(uint32_t saddr, uint32_t lbo, uint32_t sbo\) \{\s*return (.*?);\s*\}", SRC, re.S).group(1)
gd = " ".join(gd.split())
want = ("static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>((lbo & 0x3FFFF) >> 4) << 16) | "
        "(static_cast<uint64_t>((sbo & 0x3FFFF) >> 4) << 32)")
check(gd == want, "descriptor: start bits 0-13, LBO 16-29, SBO 32-45, base offset 0, swizzle none")
check(all(emu.gdesc(a, l, s) == ((a & 0x3FFFF) >> 4) | (((l & 0x3FFFF) >> 4) << 16) | (((s & 0x3FFFF) >> 4) << 32)
          for a in (0, 16, 4096, 200000) for l in (128,) for s in (512, 528)), "emulator descriptor = kernel's")
asm = re.search(r'"wgmma\.mma_async\.sync\.aligned\.m64n64k32\.f32\.e4m3\.e4m3 "(.*?)\);', SRC, re.S).group(1)
acc = re.findall(r"%(\d+)", asm.split("{%32")[0])
check([int(x) for x in acc] == list(range(32)) and '"{%32, %33, %34, %35},"' in asm and '" %36,"' in asm
      and '" p,   1, 1;\\n"' in asm and 'setp.ne.b32 p, %37, 0;' in SRC and asm.count('"+f"(d[') == 32
      and ': "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));' in SRC,
      "wgmma asm = CUTLASS MMA_64x64x32_F32E4M3E4M3_RS_TN (A regs, B desc, scale-d, scale-a/b 1)")
cons = SRC[SRC.index("  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n    bar_wait(&full_bar[st]"):SRC.index("  // partials:")]
seq = re.findall(r"bar_wait\(&full_bar|proxy_fence\(\)|fence_regs\((sc|opv)\)|wg_fence\(\)|wgmma_e4m3\((sc|opv)|wg_commit\(\)|wg_wait0\(\)|bar_arrive\(&empty_bar|ex2f\(fmaf", cons)
toks = []
for m in re.finditer(r"bar_wait\(&full_bar|proxy_fence\(\)|fence_regs\((?:sc|opv)\)|fence_a\((?:qa|pa)\[\d\]\)|wgmma2_e4m3_sync\((?:sc|opv)|bar_arrive\(&empty_bar|ex2f\(fmaf", cons):
    toks.append(m.group(0))
want_seq = (["bar_wait(&full_bar", "proxy_fence()", "fence_regs(sc)", "fence_a(qa[0])", "fence_a(qa[1])",
             "wgmma2_e4m3_sync(sc", "fence_regs(sc)", "fence_a(qa[0])", "fence_a(qa[1])"] + ["ex2f(fmaf"] * 4 +
            ["fence_regs(opv)", "fence_a(pa[0])", "fence_a(pa[1])", "wgmma2_e4m3_sync(opv", "fence_regs(opv)",
             "fence_a(pa[0])", "fence_a(pa[1])", "bar_arrive(&empty_bar"])
check(toks == want_seq, f"consumer stage order: {toks}")
# L23 / AF46: each wgmma pair is one asm statement: fence, wgmma (A a0, desc0, scale-d false), wgmma (A a1, desc1,
# scale-d true), commit, wait_group 0, in that order, accumulators %0..%31 in both
h = SRC[SRC.index("void wgmma2_e4m3_sync("):]
h = h[:h.index("\n}\n")]
text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', h.split(": \"+f\"")[0])).replace("\\n", "\n")
stmts = [x.strip() for x in text.split("\n") if x.strip() and x.strip() not in ("{", "}")]
order = [x.split()[0] for x in stmts]
w = [x for x in stmts if x.startswith("wgmma.mma_async")]
check(order == [".reg", "setp.ne.b32", "setp.ne.b32", "wgmma.fence.sync.aligned;", "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3",
                "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3", "wgmma.commit_group.sync.aligned;",
                "wgmma.wait_group.sync.aligned"] and stmts[-1] == "wgmma.wait_group.sync.aligned 0;"
      and len(w) == 2 and all(x.count("%") == 32 + 4 + 1 for x in w) and h.count('"+f"(d[') == 32
      and ': "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]),' in h
      and '"l"(desc0), "l"(desc1), "r"(0), "r"(1)' in h and "setp.ne.b32 p0, %42, 0;" in h and "setp.ne.b32 p1, %43, 0;" in h
      and h.count("{%0, %1, %2,") == 2 and "{%32, %33, %34, %35}, %40, p0, 1, 1;" in h and "{%36, %37, %38, %39}, %41, p1, 1, 1;" in h,
      "wgmma pair: one asm statement: fence -> wgmma(a0, desc0, p0 = 0) -> wgmma(a1, desc1, p1 = 1) -> commit -> wait 0")
tr = SRC[SRC.index("  if (warp >= 9) {"):SRC.index("  // ---- consumer warpgroup h")]
check(tr.index("make_uint4(t0[jd]") < tr.index("proxy_fence();") < tr.index("bar_arrive(&full_bar[st]);") and
      "bar_wait(&land_bar[st]" in tr, "transposer: land wait -> Vt stores -> proxy fence -> full arrive")
check('mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&land_bar[s]' in SRC and
      'mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(smem_u32(&full_bar[s]' in SRC and
      'mbarrier.init.shared::cta.b64 [%0], 8;" ::"r"(smem_u32(&empty_bar[s]' in SRC and
      SRC.count("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\" ::\"r\"(smem_u32(&land_bar[st]))") == 1 and
      tr.count("if (lane == 0) bar_arrive(&full_bar[st]);") == 1 and cons.count("if (lane == 0) bar_arrive(&empty_bar[st]);") == 1
      and SRC.count("        bar_arrive(&land_bar[st]);") == 1,
      "barrier counts: land 32 (copier lanes: noinc, or plain arrives after wait + fence with WF), full 2, empty 8")
check("__launch_bounds__(352, 1)" in SRC and "<<<grid, 352, smem, stream>>>" in SRC and "if (warp == 8) {" in SRC
      and "if (warp >= 9) {" in SRC and "const int h = warp >> 2, w = warp & 3;" in SRC,
      "352 threads: consumer warpgroups = warps 0-3 / 4-7, copier 8, transposers 9-10")
print("check_w1_static:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

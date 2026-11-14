"""Static protocol check of cuda_uv4.py's kernel (the parts the CPU emulator cannot see):
1. cluster barriers per thread: arrive.relaxed at entry, then on each path (producer branch /
   consumer branch) exactly one barrier.cluster.wait before any st.shared::cluster, then one
   cluster_sync() after the branches, then the producer warp leaves; no remote store after it;
2. consumer-only named barrier 1 (bar.sync 1, WC*32) twice in the consumer branch, never in the
   producer branch; __syncthreads only before the branch (mbarrier init);
3. shared memory: dynamic (stages x STAGE_BYTES + ROWS x LBP x 4) + static (bars, new rows, gate
   rows, RoPE table) within 227 KB for every variant; the host launch and setup use the same
   dynamic size; 16-CTA variants get the non-portable cluster attribute;
4. every compiled variant (stages, mt, cs) matches make_v7.py's UV4_VARIANTS.
Run: .venv/bin/python attention/local/check_uv4_protocol.py"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "cuda_uv4.py")).read(), ns)
SRC = ns["CUDA_UV4_SRC"]
bad = 0


def check(ok, what):
    global bad
    print(("ok   " if ok else "FAIL ") + what)
    bad += not ok


kern = SRC.split("template <int STAGES, int MT, int CS>")[1].split("}  // namespace okc_uv4")[0]
prod = kern[kern.index("  if (warp == WC) {"):kern.index("  } else {")]
cons = kern[kern.index("  } else {"):kern.index("  cluster_sync();                                     // every row")]
tail = kern[kern.index("  cluster_sync();                                     // every row"):]
pre = kern[:kern.index("  if (warp == WC) {")]
check(pre.count("barrier.cluster.arrive.relaxed.aligned") == 1 and "barrier.cluster.wait" not in pre,
      "entry: one relaxed cluster arrive, no wait")
check(prod.count("barrier.cluster.wait.aligned") == 1 and "st_cluster" not in prod and "bar.sync 1" not in prod,
      "producer branch: one cluster wait, no remote store, no consumer barrier")
w = cons.index("barrier.cluster.wait.aligned")
check(cons.count("barrier.cluster.wait.aligned") == 1 and "st_cluster" not in cons[:w] and "st_cluster" in cons[w:],
      "consumer branch: one cluster wait, every remote store after it")
check(cons.count('asm volatile("bar.sync 1, %0;" ::"n"(WC * 32)') == 2, "consumer branch: named barrier 1 twice")
check(kern.count("__syncthreads()") == 1 and kern.index("__syncthreads()") < kern.index("  if (warp == WC) {"),
      "__syncthreads only before the branch")
check(tail.count("cluster_sync();") == 1 and "if (warp == WC) return;" in tail and "st_cluster" not in tail and
      "barrier.cluster" not in tail.replace("cluster_sync();", "", 1), "after the branches: one cluster_sync, producer leaves, no remote access")
check(SRC.count("cp.async.wait_all") >= 2 and cons.index('cp.async.wait_all') < w,
      "consumers wait for their gate pieces before the cluster barriers")
# second level: writes, fence, consumer barrier, one ticket per (request pair, rank), barrier, the
# last CTA fences, reads through L2 (__ldcg) and resets the ticket after its reads
l2 = tail[tail.index("if (ncl > 1) {"):tail.index("// ---- 9.")]
f, b1 = l2.index("__threadfence();"), l2.index('asm volatile("bar.sync 1')
at = l2.index("atomicAdd(&tickets[bp * CS + static_cast<int>(rank)], 1) == ncl - 1")
b2 = l2.index('asm volatile("bar.sync 1', at)
ret = l2.index("if (!s_last) return;")
f2 = l2.index("__threadfence();", ret)
rd = [m.start() for m in re.finditer(r"__ldcg\(", l2)]
rs = l2.index("tickets[bp * CS + static_cast<int>(rank)] = 0;")
check(f < b1 < at < b2 < ret < f2 < min(rd) and max(rd) < rs and "g0[" not in l2.replace("__ldcg(g0", ""),
      "second level: write, fence, barrier, ticket, barrier, last CTA fences, L2 reads, ticket reset after them")
check("1 << 0" not in l2 and l2.count("atomicAdd") == 1, "one ticket per CTA")
# shared memory
STAGE_BYTES = 64 * (192 + 144)
LBP, ROWS = 68, 128
variants = sorted(tuple(int(x) for x in t.split(",")) for t in
                  re.findall(r"X\(([^)]*)\)", re.search(r"#define OKC_UV4_VARIANTS\(X\) ((?:.*\\\n)*.*)", SRC).group(1)))
for s, mt, cs in variants:
    rpc = ROWS // cs
    static = 2 * s * 8 + 2 * 8 * 128 + rpc * 64 * 2 + 2 * 8 * 32 * 4
    dyn = s * STAGE_BYTES + ROWS * LBP * 4
    check(dyn + static <= 227 * 1024, f"S{s} MT{mt} CS{cs}: dynamic {dyn} + static {static} B <= 227 KB")
check("static_cast<size_t>(stages) * okc_uv4::STAGE_BYTES + okc_uv4::ROWS * okc_uv4::LBP * 4" in SRC and
      "S * okc_uv4::STAGE_BYTES + okc_uv4::ROWS * okc_uv4::LBP * 4" in SRC and
      "float* rbuf = reinterpret_cast<float*>(ring + STAGES * STAGE_BYTES);" in SRC,
      "host launch / setup dynamic size = ring + receive buffer, rbuf after the ring")
check("if (C > 8) err |= cudaFuncSetAttribute(okc_uv4::uv4_kernel<S, M, C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);" in SRC,
      "16-CTA variants get the non-portable cluster attribute")
mv = {"__file__": os.path.join(HERE, "make_v7.py")}
exec(compile(open(os.path.join(HERE, "make_v7.py")).read().split("out_name = sys.argv[1]")[0], "make_v7", "exec"), mv)
check(set(variants) == mv["UV4_VARIANTS"], f"compiled variants {variants} == make_v7.UV4_VARIANTS")
print("check_uv4_protocol:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

"""Generates attention/local/cuda_w3d.py: W3d = W3b (cuda_w3b.py) with four copying producer warps (AF58 / AF60).
W3b moves ~2.2-2.3 TB/s on full verify with 2 copier warps (on 2 of the SM's 4 sub-partitions) while D3 reads the same
rows at ~3.1 TB/s with 12 issuing warps; FA3's paged-KV producer issues its cp.async from a whole 4-warp warpgroup.
W3d: producer warp q (q = warp - PW, 0..3) owns keys 16q .. 16q+15 of every 64-key stage. It copies their K and V rows
(whole 128 B rows, 4 keys per instruction; page entries for its 16 keys ride its slice of the page-entry ring S stages
ahead, as W2c's copier), and once its own copies of a stage have landed (cp.async.wait_group S-1: stage i - S + 1)
it re-lays V chunk q (exactly its 16 keys; W2c's transposer body for p = q), fences generic -> async proxy (covers the
cp.async-written K rows, FA3's writer-side pattern, and the Vt stores) and arrives on full (count 4). No land barrier.
Deadlock-free (AF61, sim_w3d_pipe.py): before waiting empty(i - S) warp q relays every stage <= i - S + NP, since FA3's
consumer order releases stage j only after full(j + NP); L35's first W3d relayed only <= i - S and hung.
Consumers, partials, host: W3b's (names w3b -> w3d; variants: 4-warpgroup ones).
Run: .venv/bin/python attention/local/make_w3d.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w3d.py")
VARIANTS = [(6, 1, 4, 104, 56), (6, 0, 4, 104, 56), (7, 1, 4, 104, 56), (4, 1, 4, 104, 56)]

PRODUCER = r'''  if (warp >= PW) {
    // ---- W3d producers: warp q copies keys 16q .. 16q+15 of every stage (K into the K-major tile, V into staging,
    // whole 128 B rows: lane -> key 16q + 4*i2 + (lane >> 3), chunk lane & 7), page entries for its 16 keys in its
    // slice of the ring S stages ahead (lanes 0-15), and re-lays V chunk q once its own copies of the stage landed.
    const int q4 = warp - PW;
    auto tok = [&](int j, int e) -> int { return min(start + j * SKEYS + e, end - 1); };
    auto issue_pt = [&](int j) {
      if (j < nst && lane < 16) cp4(ptr_ring + (j % PT_SLOTS) * 64 + 16 * q4 + lane, pt + tok(j, 16 * q4 + lane) / page);
    };
    const int hh = lane >> 4, qd = lane & 15;
    auto relay = [&](int j) {                          // W2c's transposer body for chunk p = q4, then arrive on full(j)
      const int st = j % STAGES;
      const uint8_t* vs = sm + st * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = sm + st * SLOTB + OFF_VT + hh * VTH;
      const int p = q4;
      uint32_t w[16];
#pragma unroll
      for (int kp = 0; kp < 16; ++kp) {
        const int key = 16 * p + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1);
        w[kp] = *reinterpret_cast<const uint32_t*>(vs + key * VS_PITCH);
      }
      uint32_t t0[4], t1[4], t2[4], t3[4];
      tr4(w[0], w[1], w[2], w[3], t0);       // t0[jd]: dim 4qd + jd, k-positions 0..3
      tr4(w[4], w[5], w[6], w[7], t1);
      tr4(w[8], w[9], w[10], w[11], t2);
      tr4(w[12], w[13], w[14], w[15], t3);
#pragma unroll
      for (int jd = 0; jd < 4; ++jd) {
        const int dn = 4 * qd + jd;
        *reinterpret_cast<uint4*>(vt + (dn & 7) * 16 + (dn >> 3) * SBOV + p * LBO_V) = make_uint4(t0[jd], t1[jd], t2[jd], t3[jd]);
      }
      proxy_fence();                                   // cp.async K rows + generic Vt stores -> wgmma (async proxy) reads
      __syncwarp();
      if (lane == 0) bar_arrive(&full_bar[st]);
    };
    for (int j = 0; j < STAGES; ++j) issue_pt(j);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();
    int nrel = 0;                                       // next stage this warp relays
    for (int i = 0; i < nst; ++i) {
      const int st = i % STAGES;
      if (i >= STAGES) {
        // AF61: consumers release stage i-S only after full(i-S+NP) (FA3 order: release(gst(k-1)) after full(gst(k))),
        // so relay every stage <= i-S+NP before blocking on empty(i-S): groups <= i-S+NP complete
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1 - NP));
        __syncwarp();
        for (; nrel <= i - STAGES + NP; ++nrel) relay(nrel);
        bar_wait(&empty_bar[st], static_cast<uint32_t>((i / STAGES - 1) & 1));
      }
      const int* pe = ptr_ring + (i % PT_SLOTS) * 64 + 16 * q4;
      const long r0 = lane < 16 ? static_cast<long>(pe[lane]) * page + tok(i, 16 * q4 + lane) % page : 0L;
      uint8_t* slot = sm + st * SLOTB;
#pragma unroll
      for (int i2 = 0; i2 < 4; ++i2) {
        const int key = 16 * q4 + 4 * i2 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, r0, key & 15);
        cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
             kc + row * row_bytes + poff + 16 * ch);
        cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
      issue_pt(i + STAGES);
      asm volatile("cp.async.commit_group;");          // group i: stage i's rows + stage i+S's entries
      asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));   // groups <= i-S+1: stage i-S+1 landed, entries of i+1
      __syncwarp();
      for (; nrel <= i - STAGES + 1; ++nrel) relay(nrel);
    }
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();
    for (; nrel < nst; ++nrel) relay(nrel);
    return;
  }
'''


def render():
    text = open(os.path.join(HERE, "cuda_w3b.py")).read()
    for pat, rep in ((r"\bokc_w3b", "okc_w3d"), (r"\bOKC_W3B_", "OKC_W3D_"), (r"\battend_w3b", "attend_w3d"),
                     (r"\bCUDA_W3B_", "CUDA_W3D_"), (r"\bW3B_VARIANTS", "W3D_VARIANTS")):
        text = re.sub(pat, rep, text)
    a = text.index("  if (warp == PW || warp == PW + 1) {\n")
    b = text.index("  // ---- W2 consumer warpgroup h")
    assert text.count("  if (warp == PW || warp == PW + 1) {\n") == 1 and text.count("  // ---- W2 consumer warpgroup h") == 1
    old = text[a:b]
    assert "if (warp >= PW + 2) {" in old and old.rstrip().endswith("return;\n  }") and "cp.async.wait_all" in old
    text = text[:a] + PRODUCER + "\n" + text[b:]
    old_init = 'asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(smem_u32(&full_bar[s])));    // transposer warps'
    assert text.count(old_init) == 1
    text = text.replace(old_init, 'asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(smem_u32(&full_bar[s])));    // W3d: the 4 producer warps')
    m = re.findall(r"#define OKC_W3D_VARIANTS\(X\) [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "#define OKC_W3D_VARIANTS(X) " + " ".join(f"X({s}, {p}, {n}, {rc}, {rp})" for s, p, n, rc, rp in VARIANTS) + "\n")
    m = re.findall(r"\nW3D_VARIANTS = [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "\nW3D_VARIANTS = " + repr(VARIANTS) + "\n")
    assert "land_bar[st]" not in text.split("// ---- W2 consumer warpgroup h")[0].split("if (warp >= PW) {")[1]
    # watchdog (lab safety after L35's hang): a barrier wait that spins 2^20 times sets okc_w3d_hang and gives up, so a
    # protocol bug returns garbage (the lab sees the flag and drops W3d) instead of hanging the whole check run
    bw = ('__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  while (!done)\n'
          '    asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n'
          '                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n')
    assert text.count(bw) == 1, text.count(bw)
    text = text.replace(bw, '__device__ unsigned int okc_w3d_hang = 0;   // watchdog flag (make_w3d.py)\n'
                        + bw.replace('  uint32_t done = 0;\n  while (!done)\n', '  uint32_t done = 0, spins = 0;\n  while (!done) {\n')
                            .replace(' : "memory");\n}\n', ' : "memory");\n'
                                     '    if (!done && ++spins > (1u << 20)) {\n      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_w3d_hang), "r"(1u) : "memory");\n      break;\n    }\n  }\n}\n'))
    host = ('\nint64_t attend_w3d_hang() {   // lab: 1 when a W3d barrier wait gave up (watchdog); reads and clears the flag\n'
            '  unsigned int h = 0, z = 0;\n'
            '  cudaMemcpyFromSymbol(&h, okc_w3d::okc_w3d_hang, sizeof(h));\n'
            '  cudaMemcpyToSymbol(okc_w3d::okc_w3d_hang, &z, sizeof(z));\n'
            '  return h;\n}\n')
    k = text.index('"""\n\nCUDA_W3D_DECL = ')
    text = text[:k] + host + text[k:]
    text = text.replace("std::vector<int64_t> attend_w3d_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);'",
                        "std::vector<int64_t> attend_w3d_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);\\nint64_t attend_w3d_hang();'")
    assert "int64_t attend_w3d_hang();" in text
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_w3d.py from cuda_w3b.py (do not edit): W3d, W3b with four copying producer warps (see\n'
            'make_w3d.py). Original docstring follows.\n' + text[3:i] + text[i:])


def sim():
    """Producer warp q's cp.async groups and barriers for nst 0..12 and S in the variants: each stage relayed (full
    arrived) exactly once, after its group completed; empty(i-S) waited only after this warp arrived full(i-S); the
    page entries of stage i read only after their group completed; every (key, chunk) of a stage copied once over the
    4 warps; the transpose of chunk q reads only keys this warp copied."""
    for S in sorted({v[0] for v in VARIANTS}):
        for nst in range(0, 13):
            done, relayed, pending, entries_done = set(), [], [], set(range(min(S, nst)))   # prologue waits all entries
            arrived = set()
            nrel = 0
            for i in range(nst):
                if i >= S:                                      # pre-empty relays: wait_group S-1-NP, relay <= i-S+NP
                    NP = 2
                    while len(pending) > S - 1 - NP:
                        _, g = pending.pop(0)
                        done.add(g)
                        if g + S < nst:
                            entries_done.add(g + S)
                    while nrel <= i - S + NP:
                        assert nrel in done, (S, nst, nrel)
                        relayed.append(nrel)
                        arrived.add(nrel)
                        nrel += 1
                    assert (i - S) in arrived, (S, nst, i)
                assert i in entries_done, (S, nst, i, "entries")
                pending.append(("g", i))                        # group i: rows of i + entries of i+S
                while len(pending) > S - 1:
                    _, g = pending.pop(0)
                    done.add(g)
                    if g + S < nst:
                        entries_done.add(g + S)
                while nrel <= i - S + 1:
                    assert nrel in done, (S, nst, nrel)
                    relayed.append(nrel)
                    arrived.add(nrel)
                    nrel += 1
            for _, g in pending:
                done.add(g)
            while nrel < nst:
                relayed.append(nrel)
                arrived.add(nrel)
                nrel += 1
            assert sorted(relayed) == list(range(nst)), (S, nst, relayed)
    seen = {}
    for q in range(4):
        for i2 in range(4):
            for lane in range(32):
                key, ch = 16 * q + 4 * i2 + (lane >> 3), lane & 7
                assert key >> 4 == q and (key & 15) < 16
                seen[(key, ch)] = seen.get((key, ch), 0) + 1
        for kp in range(16):   # chunk p = q reads keys 16p + ...
            key = 16 * q + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1)
            assert key >> 4 == q
    assert sorted(seen) == [(k, c) for k in range(64) for c in range(8)] and set(seen.values()) == {1}


if __name__ == "__main__":
    sim()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w3d.py up to date" if same else "cuda_w3d.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"protocol sim ok; wrote {OUT}")

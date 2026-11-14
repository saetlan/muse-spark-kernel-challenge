"""Derives S2c (cuda_ver3.py) from S2b (cuda_ver2.py) by exact text edits: only the producer warp
changes (AF25, AE16). S2b's producer loads the next stage's page entries with plain loads and moves
them between registers (`r0 = n0`), which stalls on the pending load: one dependent DRAM round
trip per 64-key stage. S2c's producer copies each stage's 64 page entries with 4-byte cp.async
STAGES stages ahead, into the unused pads (bytes 128..191) of the K rows of the ring (consumers read
K bytes 0..127 only); commit group i = stage i's K/V rows + stage i+STAGES's entries; after each
stage the producer waits until at most STAGES-1 groups are pending, which lands the next stage's
entries (and K/V of stage i+1-STAGES, which the empty-barrier wait implies anyway). Consumers,
barriers and math are S2b's (emulate_cuda_ver2.py, sim_ver2_barriers.py apply).
Usage: .venv/bin/python attention/local/make_s2c.py [--check]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = "--check" in sys.argv


def sub1(text, old, new):
    assert text.count(old) == 1, f"anchor found {text.count(old)} times: {old[:90]!r}"
    return text.replace(old, new)


OLD_PRODUCER_HEAD = '''    // ---- producer: whole-row 16 B copies of each 64-key stage, rows one stage ahead.
    auto rows_of = [&](int i, long& r0, long& r1) {
      const int s0 = min(start + i * SKEYS + lane, end - 1), s1 = min(start + i * SKEYS + 32 + lane, end - 1);
      r0 = static_cast<long>(pt[s0 / page]) * page + s0 % page;
      r1 = static_cast<long>(pt[s1 / page]) * page + s1 % page;
    };
    long r0 = 0, r1 = 0, n0 = 0, n1 = 0;
    if (nst > 0) rows_of(0, r0, r1);
    for (int i = 0; i < nst; ++i) {
      const int st = i % STAGES;
      if (i + 1 < nst) rows_of(i + 1, n0, n1);
'''
NEW_PRODUCER_HEAD = '''    // ---- producer: whole-row 16 B copies of each 64-key stage. Page entries ride the ring: stage
    // j's 64 entries sit in PT slot p = j % PT_SLOTS, in the K rows' unused pads (bytes 128..191) of
    // ring slot p % STAGES, rows 4*(p / STAGES) .. +3: entry e at row 4*(p / STAGES) + e / 16, byte
    // 128 + 4*(e % 16). Consumers read K bytes 0..127 only; row copies write bytes 0..127.
    constexpr int PT_SLOTS = 2 * STAGES;
    auto pt_slot = [&](int j, int e) -> int* {
      const int p = j % PT_SLOTS;
      return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 4)) * KP + 128 + 4 * (e & 15));
    };
    auto tok = [&](int j, int e) -> int { return min(start + j * SKEYS + e, end - 1); };
    auto issue_pt = [&](int j) {        // stage j's 64 page entries, 2 per lane (join the open group)
      if (j < nst) {
        cp4(pt_slot(j, lane), pt + tok(j, lane) / page);
        cp4(pt_slot(j, 32 + lane), pt + tok(j, 32 + lane) / page);
      }
    };
    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();
    for (int i = 0; i < nst; ++i) {
      const int st = i % STAGES;
'''
OLD_ROWS_USE = '''      uint8_t* skw = ring + st * STAGE_BYTES;
      uint8_t* svw = skw + SKEYS * KP;
'''
NEW_ROWS_USE = '''      // stage i's rows from its page entries (landed: waited and warp-synced after stage i-1)
      const long r0 = static_cast<long>(*pt_slot(i, lane)) * page + tok(i, lane) % page;
      const long r1 = static_cast<long>(*pt_slot(i, 32 + lane)) * page + tok(i, 32 + lane) % page;
      uint8_t* skw = ring + st * STAGE_BYTES;
      uint8_t* svw = skw + SKEYS * KP;
'''
OLD_PRODUCER_TAIL = '''      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&full_bar[st])) : "memory");
      r0 = n0;
      r1 = n1;
    }
'''
NEW_PRODUCER_TAIL = '''      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&full_bar[st])) : "memory");
      issue_pt(i + STAGES);
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));   // stage i+1's entries landed
      __syncwarp();
    }
'''
CP16 = '''__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
'''
CP4 = '''__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}
'''


def main():
    text = open(os.path.join(HERE, "cuda_ver2.py")).read()
    _, rest = text.split('CUDA_VER2_SRC = r"""', 1)
    src = 'CUDA_VER2_SRC = r"""' + rest
    src = sub1(src, CP16, CP16 + CP4)
    src = sub1(src, OLD_PRODUCER_HEAD, NEW_PRODUCER_HEAD)
    src = sub1(src, OLD_ROWS_USE, NEW_ROWS_USE)
    src = sub1(src, OLD_PRODUCER_TAIL, NEW_PRODUCER_TAIL)
    src = src.replace("ver2", "ver3").replace("VER2", "VER3")
    doc = ('"""S2c: S2b (cuda_ver2.py) with the producer\'s page entries carried by the cp.async ring\n'
           '(generated by make_s2c.py; see its docstring). Consumers, barriers and math are S2b\'s.\n"""\n\n')
    out = os.path.join(HERE, "cuda_ver3.py")
    if CHECK:
        same = os.path.exists(out) and open(out).read() == doc + src
        print(("up to date: " if same else "STALE: ") + out)
        sys.exit(0 if same else 1)
    open(out, "w").write(doc + src)
    print("wrote", out)


if __name__ == "__main__":
    main()

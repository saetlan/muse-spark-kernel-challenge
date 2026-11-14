"""Derives the page-table-ring variants from the verified sources by exact text edits (each
anchor must match once):
  cuda_dec3.py (D2)     -> cuda_dec4.py (D3)
  cuda_unified.py (UD)  -> cuda_unified2.py (UD2)

Why (L9b, AE14): D2 loads chunk i+S's page entries with a plain load one iteration before the
K/V copies that need them, so every iteration waits one dependent DRAM round trip (~2 us under
load): 4 KB per ~2 us per warp, 2.1-2.5 TB/s. Changes:
1. page entries ride the cp.async ring: commit group i holds chunk i's K/V rows and chunk
   i+STAGES's page entries (4-byte cp.async by lanes 0..15), so the entries a copy needs landed
   STAGES-1 iterations earlier; only the prologue waits for one round trip;
2. the entries live in the unused 16-byte row pads (bytes 128..143) of the ring's K tiles, so the
   ring keeps its size;
3. the end-of-slice warp merge reuses the ring (dynamic shared memory) instead of 16 KB of static
   shared memory: w4 s3 fits 4 CTAs per SM (was 3);
4. UD2 only: the ring prologue (page entries, first K/V stages) is issued before the query fold,
   RoPE table and new row, so their loads overlap the K/V round trip.
Usage: .venv/bin/python attention/local/make_pt_ring.py [--check]
  --check: generate in memory and fail if cuda_dec4.py / cuda_unified2.py differ (presubmit)."""
import os
import re
import sys

CHECK = "--check" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))


def sub1(text, old, new):
    assert text.count(old) == 1, f"anchor found {text.count(old)} times: {old[:80]!r}"
    return text.replace(old, new)


CP4 = '''__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}
'''

STATIC_MERGE = '''  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];
'''

OLD_RING = '''  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Row of token (start + c*CHUNK + (lane & 15)) of the warp's i-th chunk, clamped into the slice.
  auto row_of = [&](int i) -> long {
    const int s = min(start + (warp + i * WARPS) * CHUNK + (lane & 15), end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto issue = [&](int i, long my_row) {
    if (i < mine) {
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  long row_next = 0;
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i, i < mine ? row_of(i) : 0);
  if (STAGES - 1 < mine) row_next = row_of(STAGES - 1);
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1, row_next);
    if (i + STAGES < mine) row_next = row_of(i + STAGES);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
'''

NEW_RING = '''  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
  // Page entries ride the ring: chunk j's 16 entries sit in PT slot p = j % PT_SLOTS, in the unused
  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at
  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.
  constexpr int PT_SLOTS = 2 * STAGES;
  auto pt_slot = [&](int j, int e) -> int* {
    const int p = j % PT_SLOTS;
    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));
  };
  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j's page entries (join the open commit group)
    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);
  };
  // One commit group per call: chunk i's K/V rows (its page entries landed in an earlier group,
  // waited and warp-synced) and chunk i + STAGES's page entries.
  auto issue = [&](int i) {
    if (i < mine) {
      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    issue_pt(i + STAGES);
    asm volatile("cp.async.commit_group;");
  };
  // Prologue: the first STAGES chunks' page entries (the one exposed round trip), then the first
  // STAGES - 1 stages of K/V rows.
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);
'''

NEW_LOOP_HEAD = '''  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
'''

OLD_TAIL = '''  asm volatile("cp.async.wait_all;");
'''
NEW_TAIL = '''  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);
'''

CP16_END = '''  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
'''


def emit(path, text):
    if CHECK:
        same = os.path.exists(path) and open(path).read() == text
        print(("up to date: " if same else "STALE (edited by hand or parent changed): ") + path)
        if not same:
            sys.exit(1)
        return
    open(path, "w").write(text)
    print("wrote", path)


def common(src):
    src = sub1(src, CP16_END, CP16_END + CP4)
    src = sub1(src, STATIC_MERGE, "")
    src = sub1(src, OLD_TAIL, NEW_TAIL)
    return src


def make_dec4():
    text = open(os.path.join(HERE, "cuda_dec3.py")).read()
    head, rest = text.split('CUDA_DEC3_SRC = r"""', 1)
    src = 'CUDA_DEC3_SRC = r"""' + rest
    src = common(src)
    src = sub1(src, OLD_RING, NEW_RING + NEW_LOOP_HEAD)
    src = src.replace("dec3", "dec4").replace("DEC3", "DEC4")
    doc = ('"""D3: D2 (cuda_dec3.py) with its page entries carried by the cp.async ring (generated by\n'
           'make_pt_ring.py; see its docstring): commit group i = chunk i\'s K/V rows + chunk i+STAGES\'s\n'
           'page entries (4-byte copies into the K tiles\' unused row pads), so no iteration waits on a\n'
           'dependent page-table load; the warp merge reuses the ring instead of static shared memory.\n'
           'Math, chunk assignment and fragment layout are D2\'s (emulate_cuda_dec3.py applies).\n"""\n\n')
    emit(os.path.join(HERE, "cuda_dec4.py"), doc + src)


def make_ud2():
    text = open(os.path.join(HERE, "cuda_unified.py")).read()
    head, rest = text.split('CUDA_UD_SRC = r"""', 1)
    src = 'CUDA_UD_SRC = r"""' + rest
    src = common(src)
    pt_line = "  const int* pt = page_table + static_cast<long>(b) * max_pages;\n"
    src = sub1(src, pt_line, "")
    src = sub1(src, OLD_RING, NEW_LOOP_HEAD)
    rope_anchor = "  // ---- RoPE table for this request's position"
    src = sub1(src, rope_anchor,
               "  // ---- ring prologue first (page entries, then the first K/V stages): the fold below\n"
               "  // overlaps their round trips.\n" + pt_line + NEW_RING + "\n" + rope_anchor)
    src = re.sub(r"\bokc_ud\b", "okc_ud2", src)
    src = re.sub(r"\bud_(kernel|setup|attrs)\b", r"ud2_\1", src)
    src = re.sub(r"\bvoid ud\(", "void ud2(", src)
    src = src.replace("OKC_UD_", "OKC_UD2_").replace("CUDA_UD_", "CUDA_UD2_").replace('"ud launch failed', '"ud2 launch failed')
    doc = ('"""UD2: UD (cuda_unified.py) with the page-table ring of make_pt_ring.py (generated; see its\n'
           'docstring): page entries carried by the cp.async ring in the K tiles\' row pads, the warp\n'
           'merge in the ring\'s shared memory, and the ring prologue issued before the query fold, RoPE\n'
           'table and new row so those loads overlap the K/V round trip. Math, chunk assignment, new-row\n'
           'patch and cluster merge are UD\'s (emulate_cuda_ud.py applies).\n"""\n\n')
    emit(os.path.join(HERE, "cuda_unified2.py"), doc + src)


if __name__ == "__main__":
    make_dec4()
    make_ud2()

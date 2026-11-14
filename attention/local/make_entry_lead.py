"""Generates attention/local/cuda_w3e.py (from cuda_w3b.py), cuda_uw2e.py (from cuda_uw2.py), cuda_w1ne.py (from cuda_w1n.py)
and cuda_w2e.py (from cuda_w2.py): the W2c-family copier
with its page entries prefetched 2S stages ahead instead of S (AF66 / AF67 follow-up).
In W2c / W3b / UW2 the copier's cp.async group i holds stage i's K / V rows AND stage i + S's page entries, and after each
stage it waits `cp.async.wait_group S - 1` so that stage i + 1's entries have landed; groups complete in order, so this
also waits for stage i - S + 1's rows: the copier stalls on a DRAM round trip every stage although the ring's empty
barrier already bounds the rows in flight. Here the entries ride 2S stages ahead (ring of 3S slots: stage j's entries are
written at iteration j - 2S and read at iteration j, slot j % 3S, rewritten at j + S), the prologue issues 2S stages of
entries, and the per-stage wait is `wait_group 2S - 1` (groups <= i - 2S + 1: stage i + 1's entries; the rows they also
cover are older than the ring, already landed and consumed). The copier then blocks only on empty slots. Only these four
spots change (asserted); host shared memory grows by S x 256 B.
Run: .venv/bin/python attention/local/make_entry_lead.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


def lead(text, tag, renames, host_old, host_new):
    for pat, rep in renames:
        text = re.sub(pat, rep, text)
    text = sub1(text, "  constexpr int PT_SLOTS = 2 * STAGES;\n",
                "  constexpr int PT_SLOTS = 3 * STAGES;               // " + tag + ": entries ride 2S stages ahead (AF66)\n", "pt slots")
    text = sub1(text, "    for (int j = 0; j < STAGES; ++j) issue_pt(j);\n", "    for (int j = 0; j < 2 * STAGES; ++j) issue_pt(j);\n", "prologue")
    n = text.count("issue_pt(i + STAGES);")
    assert n in (2, 3), n
    text = text.replace("issue_pt(i + STAGES);", "issue_pt(i + 2 * STAGES);")
    text = sub1(text, 'asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));   // stage i+1\'s entries landed',
                'asm volatile("cp.async.wait_group %0;" ::"n"(2 * STAGES - 1));   // ' + tag + ": stage i+1's entries (rows older than the ring)",
                "group wait")
    text = sub1(text, host_old, host_new, "host smem")
    return text


def sim():
    """Copier group / slot model for S in 3..8: stage i's entries landed before they are read; the entry slot read at
    iteration i is never overwritten before (written at i - 2S, next write at i + S); the wait never needs rows newer than
    the ones the empty barrier already released."""
    for S in range(3, 9):
        P = 3 * S
        writes = {}
        for i in range(-2 * S, 40):
            j = i + 2 * S
            writes.setdefault(j % P, []).append((i, j))   # at iteration i the entries of stage j go to slot j % P
        for j in range(0, 40):
            # the read of stage j at iteration j sees the write of stage j (iteration j - 2S) and no later write before it
            ws = [(it, st) for (it, st) in writes[j % P] if it <= j]
            assert ws[-1][1] == j, (S, j, ws[-3:])
            # group arithmetic: after iteration i - 1's wait (groups <= i - 1 - (2S - 1) complete), stage i's entries
            # (group i - 2S, or the prologue for i < 2S) have landed
            assert i_entry_group(j, S) <= (j - 1) - (2 * S - 1) or j < 2 * S
        # the wait at iteration i needs groups <= i - 2S + 1 complete: their rows are stage <= i - 2S + 1 < i - S + 1, which
        # the copier already waited empty for (stage i - S's slot) -> never a wait on rows still in flight
        for i in range(S, 40):
            assert i - 2 * S + 1 <= i - S


def i_entry_group(j, S):
    return j - 2 * S


def main():
    sim()
    outs = {
        "cuda_w3e.py": lead(open(os.path.join(HERE, "cuda_w3b.py")).read(), "W3e",
                            ((r"\bokc_w3b", "okc_w3e"), (r"\bOKC_W3B_", "OKC_W3E_"), (r"\battend_w3b", "attend_w3e"),
                             (r"\bCUDA_W3B_", "CUDA_W3E_"), (r"\bW3B_VARIANTS", "W3E_VARIANTS")),
                            "static_cast<size_t>(2 * stages) * 64 * 4", "static_cast<size_t>(3 * stages) * 64 * 4"),
        "cuda_uw2e.py": lead(open(os.path.join(HERE, "cuda_uw2.py")).read(), "UW2e",
                             ((r"\bokc_uw2", "okc_uw2e"), (r"\bOKC_UW2_", "OKC_UW2E_"), (r"\buw2\(", "uw2e("), (r"\buw2_setup\b", "uw2e_setup"),
                              (r"\buw2_attrs\b", "uw2e_attrs"), (r"\buw2_kernel\b", "uw2e_kernel"), (r"\bCUDA_UW2_", "CUDA_UW2E_"),
                              (r'"uw2: ', '"uw2e: '), (r'"uw2 ', '"uw2e ')),
                             "static_cast<size_t>(2 * stages) * 64 * 4", "static_cast<size_t>(3 * stages) * 64 * 4"),
        "cuda_w1ne.py": lead(open(os.path.join(HERE, "cuda_w1n.py")).read().replace(
                                 "S * (okc_w1n::OFF_VT + 2 * 8 * 528) + 2 * S * 64 * 4", "S * (okc_w1n::OFF_VT + 2 * 8 * 528) + 3 * S * 64 * 4"),
                             "W1-Ne",
                             ((r"\bokc_w1n", "okc_w1ne"), (r"\bOKC_W1N_", "OKC_W1NE_"), (r"\battend_w1n", "attend_w1ne"),
                              (r"\bCUDA_W1N_", "CUDA_W1NE_")),
                             "static_cast<size_t>(2 * stages) * 64 * 4", "static_cast<size_t>(3 * stages) * 64 * 4"),
        "cuda_w2e.py": lead(open(os.path.join(HERE, "cuda_w2.py")).read(), "W2e",
                            ((r"\bokc_w2", "okc_w2e"), (r"\bOKC_W2_", "OKC_W2E_"), (r"\bOKC_W2ABL_", "OKC_W2EABL_"),
                             (r"\battend_w2", "attend_w2e"), (r"\bCUDA_W2_", "CUDA_W2E_")),
                            "static_cast<size_t>(2 * stages) * 64 * 4", "static_cast<size_t>(3 * stages) * 64 * 4"),
    }
    stale = 0
    for name, text in outs.items():
        text = ('"""Generated by make_entry_lead.py (do not edit): the copier\'s page entries 2S stages ahead (AF66).\n'
                'Original docstring follows.\n' + text[3:])
        out = os.path.join(HERE, name)
        if "--check" in sys.argv:
            same = os.path.exists(out) and open(out).read() == text
            print(f"{name} {'up to date' if same else 'STALE'}")
            stale += not same
        else:
            open(out, "w").write(text)
            print(f"wrote {out}")
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()

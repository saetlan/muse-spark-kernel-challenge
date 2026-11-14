"""Every kernel launched with more than 48 KB of dynamic shared memory must have that limit raised first.

Without cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) such a launch is refused
with "invalid argument" (E17 lost its whole tuner to one recon instance). This script:
  1. expands the submission's CUDA source with clang -E (host side, stub headers), so macro parameters are
     substituted into every launch;
  2. collects each launch's kernel instantiation and dynamic shared-memory expression (m13_launch calls and
     <<<grid, block, smem, stream>>> launches), and every instantiation that gets the attribute;
  3. for each launch without the attribute and a non-zero size, appends static_assert(size <= 48 KB) to the
     source and lets clang evaluate the exact expressions;
  4. mutation-tests itself: with M18's recon attributes removed, the check must fail.

Run: .venv/bin/python moe/local/check_smem_attrs.py moe/submission_m13t.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))

DEFAULT_LIMIT = 48 * 1024


def _clang(src, extra):
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "k.cu"
        cu.write_text(src)
        cmd = ["clang", "-x", "cuda", "--cuda-gpu-arch=sm_90a", "--cuda-host-only", "-nocudainc", "-nocudalib",
               "-std=c++20", "-Wno-unused-variable", "-I", str(LOCAL / "cuda_stubs")] + extra + [str(cu)]
        res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def _args(text, start):
    """Top-level comma-separated arguments of the call whose '(' is at text[start]."""
    assert text[start] == "("
    depth, cur, out = 0, [], []
    for ch in text[start:]:
        if ch == "(":
            depth += 1
            if depth > 1:
                cur.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.append("".join(cur).strip())
                return out
            cur.append(ch)
        elif ch == "," and depth == 1 and _angle_depth(cur) == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    raise ValueError("unbalanced call")


def _angle_depth(chars):
    """Open template brackets in the current argument (template args contain commas: m18::k<2, 4, true, 2, 1>)."""
    s = "".join(chars)
    s = re.sub(r"<<|>>|<=|>=|->", "", s)
    return s.count("<") - s.count(">")


def _norm(expr):
    return re.sub(r"\s+", "", expr)


def collect(expanded):
    """[(kernel, smem expression)] launched, and the set of kernels whose dynamic smem limit is raised."""
    launches, raised = [], set()
    for m in re.finditer(r"(?<![\w:])m13_launch\s*\(", expanded):
        before = expanded[max(0, m.start() - 40):m.start()]
        if re.search(r"void\s+$", before):  # the template's own definition
            continue
        a = _args(expanded, m.end() - 1)
        launches.append((_norm(a[0]), a[3]))
    for m in re.finditer(r"([\w:]+\s*<[^;]*?>)\s*<<<", expanded):
        kern = m.group(1)
        body = expanded[m.end():expanded.index(">>>", m.end())]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) >= 3:
            launches.append((_norm(kern), parts[2]))
    for m in re.finditer(r"(?<![\w:])cudaFuncSetAttribute\s*\(", expanded):
        a = _args(expanded, m.end() - 1)
        if "MaxDynamicSharedMemorySize" in a[1]:
            raised.add(_norm(a[0]))
    return launches, raised


def check_source(src, label):
    """Raises SystemExit naming each launch above 48 KB whose limit is never raised."""
    full = '#include "cuda_stub_prelude.h"\n' + src
    rc, out, err = _clang(full, ["-E", "-P"])
    if rc != 0:
        raise SystemExit(f"check_smem_attrs: clang -E failed for {label}:\n{err[-3000:]}")
    launches, raised = collect(out)
    if not launches:
        raise SystemExit(f"check_smem_attrs: no launches found in {label} (parser broken?)")
    unraised = {}
    for kern, smem in launches:
        if kern in raised or _norm(smem) in ("0", "static_cast<size_t>(0)", "0u"):
            continue
        unraised.setdefault(kern, smem)
    # E22b: no size exemption. The 48 KB default covers static + dynamic shared memory together (E22: 24 KB static +
    # 36 KB dynamic was refused), and clang can't see static sizes, so every non-zero dynamic launch needs its limit raised.
    if unraised:
        lines = "\n".join(f"  {k} launched with {v} B of dynamic shared memory, limit never raised" for k, v in unraised.items())
        raise SystemExit(f"check_smem_attrs: {label}:\n{lines}")
    return len(launches), len(raised), 0


def main(path):
    import check_cuda_syntax
    src = check_cuda_syntax.cuda_source(path)
    n, r, u = check_source(src, Path(path).name)
    # mutation: drop M18's recon attributes (the E17 bug); the check must catch it
    planted = re.sub(r"\n\s*M18_RECON_ATTR\(\d\)", "", src)
    if planted == src:
        raise SystemExit("check_smem_attrs: mutation found no M18_RECON_ATTR to remove")
    try:
        check_source(planted, "mutant without M18 recon attributes")
    except SystemExit as exc:
        if "limit never raised" in str(exc):
            pass
        else:
            raise
    else:
        raise SystemExit("check_smem_attrs: the mutant without M18 recon attributes passed (check is blind)")
    planted2 = re.sub(r"\n\s*cudaFuncSetAttribute\(m11::sr_kernel<[^;]*;", "", src)
    if planted2 == src:
        raise SystemExit("check_smem_attrs: mutation found no M11 SR attribute to remove")
    try:
        check_source(planted2, "mutant without M11 SR attributes")
    except SystemExit as exc:
        if "limit never raised" not in str(exc):
            raise
    else:
        raise SystemExit("check_smem_attrs: the mutant without M11 SR attributes passed (check is blind)")
    print(f"ok: {Path(path).name}: {n} launch sites, {r} raised limits, every non-zero dynamic launch covered; "
          f"mutants without M18 recon / M11 SR attributes caught")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(LOCAL.parent / "submission_m13t.py"))

"""Simulates S2c's producer page-entry ring (cuda_ver3.py) from the source text: PT_SLOTS, the slot
formula, the prologue (count, wait), the row read of stage i, the lead of issue_pt inside the
loop and the loop's wait_group count are parsed out of the CUDA source and evaluated; the
producer's schedule is replayed for STAGES 2..4 and 0..40 stages with the most adversarial
completion cp.async allows (a commit group completes only when a wait_group forces it). Checks:
every row read sees that stage's entries landed (never stale or missing); no entry copy overwrites
entries not yet read; entries sit in the K rows' pads (bytes 128..191 of a 192-byte row), never
on K/V bytes, and slots never overlap. (K/V hand-off to the consumers is the mbarrier protocol of
S2b, unchanged: sim_ver2_barriers.py.)
Run: .venv/bin/python attention/local/sim_ver3_pt.py [file.py:VAR]   (exit 1 on a violation; V4 and UV4
carry the same producer: cuda_ver4.py:CUDA_VER4_SRC, cuda_uv4.py:CUDA_UV4_SRC)"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KP, VP, SKEYS = 192, 144, 64
STAGE_BYTES = SKEYS * (KP + VP)


def source(spec):
    path, var = spec.split(":")
    s = importlib.util.spec_from_file_location("m", os.path.join(HERE, path))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return getattr(m, var)


def c2py(expr):
    return re.sub(r"(?<![/])/(?![/])", "//", expr)


def grab(src, pattern, what):
    m = re.search(pattern, src, flags=re.S)
    assert m, f"could not parse {what}"
    return m.group(1).strip()


def parse(src):
    a = "if (warp == 4) {" if "if (warp == 4) {" in src else "if (warp == WC) {"   # S2c / V4, UV4 (same producer)
    i = src.index(a)
    j = src.find("// ---- consumers", i)
    prod = src[i:j if j >= 0 else src.index("  } else {", i)]
    p = {}
    p["slots"] = c2py(grab(prod, r"constexpr int PT_SLOTS = ([^;]+);", "PT_SLOTS"))
    p["slot_p"] = c2py(grab(prod, r"auto pt_slot = \[&\]\(int j, int e\) -> int\* \{\s*const int p = ([^;]+);", "slot p"))
    p["addr"] = c2py(grab(prod, r"return reinterpret_cast<int\*>\(ring \+ (.+?)\);\s*\};", "slot address"))
    ipt = grab(prod, r"auto issue_pt = \[&\]\(int j\) \{(.+?)\n    \};", "issue_pt")
    p["entries"] = [c2py(e) for e in re.findall(r"cp4\(pt_slot\(j, ([^)]+)\)", ipt)]
    p["pro_n"] = c2py(grab(prod, r"for \(int j = 0; j < ([^;]+); \+\+j\) issue_pt\(j\);", "prologue count"))
    p["pro_wait"] = grab(prod, r"issue_pt\(j\);[^\n]*\n\s*asm volatile\(\"cp.async.commit_group;\"\);\s*asm volatile\(\"cp.async.wait_group (\d+);\"\);\s*__syncwarp\(\);", "prologue wait")
    p["reads"] = [(c2py(a), c2py(b)) for a, b in re.findall(r"\*pt_slot\(([^,]+), ([^)]+)\)\) \* page", prod)]
    p["lead"] = c2py(grab(prod, r"\n      issue_pt\(i \+ ([^)]+)\);", "loop lead"))
    p["loop_wait"] = c2py(grab(prod, r"issue_pt\(i \+ [^)]+\);\s*asm volatile\(\"cp.async.commit_group;\"\);\s*asm volatile\(\"cp.async.wait_group %0;\" ::\"n\"\(([^)]+)\)\);", "loop wait"))
    return p


def run(p, S, nst):
    env = dict(STAGES=S, STAGE_BYTES=STAGE_BYTES, KP=KP)
    slots = eval(p["slots"], {}, env)
    bad = []

    def addr(j, e):
        pp = eval(p["slot_p"], {}, dict(env, j=j, PT_SLOTS=slots))
        return eval(p["addr"], {}, dict(env, p=pp, e=e))

    used = {}
    for slot in range(slots):
        for e in range(64):
            a = addr(slot, e)
            stage, off = divmod(a, STAGE_BYTES)
            if not (0 <= stage < S) or a % 4 or off >= SKEYS * KP or off % KP < 128:
                bad.append(f"S{S}: entry ({slot},{e}) at stage offset {off} is not in a K-row pad")
            if a in used:
                bad.append(f"S{S}: entries {used[a]} and ({slot},{e}) share byte {a}")
            used[a] = (slot, e)
    if bad:
        return bad
    mem, groups, open_g = {}, [], []
    waited = -1
    lead = eval(p["lead"], {}, env)
    ents = [eval(e, {}, {"lane": lane}) for e in p["entries"] for lane in range(32)]
    if sorted(ents) != list(range(64)):
        return [f"S{S}: issue_pt copies entries {sorted(set(ents))[:8]}..., not 0..63 once each"]

    def issue_pt(j):
        if j < nst:
            for e in range(64):
                a = addr(j, e)
                prev = mem.get(a)
                if prev is not None and not prev[2]:
                    bad.append(f"S{S} n{nst}: PT{j} overwrites unread PT{prev[1]}")
                    return
                mem[a] = [len(groups), j, False]
            open_g.append(j)

    def commit():
        groups.append(list(open_g))
        open_g.clear()

    for j in range(eval(p["pro_n"], {}, env)):
        issue_pt(j)
    commit()
    waited = max(waited, len(groups) - 1 - int(p["pro_wait"]))
    for i in range(nst):
        for jexpr, eexpr in p["reads"]:
            for lane in range(32):
                j = eval(jexpr, {}, {"i": i})
                e = eval(eexpr, {}, {"lane": lane})
                cur = mem.get(addr(j, e))
                if j != i or cur is None or cur[1] != i or cur[0] > waited:
                    bad.append(f"S{S} n{nst}: stage {i} reads entry {e} of {'stage %d' % cur[1] if cur else 'nothing'}"
                               f"{' before it landed' if cur and cur[0] > waited else ''}")
                    return bad
                cur[2] = True
        issue_pt(i + lead)
        commit()
        waited = max(waited, len(groups) - 1 - eval(p["loop_wait"], {}, env))
        if bad:
            return bad
    return bad


def check(src, label):
    p = parse(src)
    problems = []
    for S in (2, 3, 4):
        for nst in range(0, 41):
            problems += run(p, S, nst)
            if problems:
                break
        if problems:
            break
    print(("ok: " if not problems else "FAIL: ") + label + ("" if not problems else f" -> {problems[0]}"))
    return not problems


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else "cuda_ver3.py:CUDA_VER3_SRC"
    sys.exit(0 if check(source(spec), spec) else 1)

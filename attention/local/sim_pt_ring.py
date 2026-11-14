"""Simulates the page-table ring protocol of D3 / UD2 (cuda_dec4.py, cuda_unified2.py) from the
source text itself: the slot formula, PT_SLOTS, the page-entry lead inside issue(), the
prologue counts, the loop's issue offset and the wait_group count are parsed out of the CUDA
source and evaluated, then one warp's schedule is replayed for every STAGES in {2, 3, 4} and
every chunk count 0..40 with the most adversarial completion order cp.async allows (a commit
group completes only when a wait_group forces it). Checks:
- every page-entry read (issue(i) computing chunk i's rows) sees chunk i's entries landed
  (their group waited) and warp-synced, never stale ones;
- every K/V fragment read of chunk i sees its rows landed;
- no copy overwrites bytes whose data hasn't been read yet (page entries or K/V);
- page entries sit in the K tiles' pads (bytes 128..143 of a 144-byte row), never on K/V bytes,
  and distinct slots never overlap.
Run: .venv/bin/python attention/local/sim_pt_ring.py [file.py:VAR ...]   (exit 1 on a violation)
Used by mutation_test_pt_ring.py with planted bugs."""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["cuda_dec4.py:CUDA_DEC4_SRC", "cuda_unified2.py:CUDA_UD2_SRC", "cuda_dec4p.py:CUDA_DEC4P_SRC"]
CHUNK, PITCH = 16, 144
STAGE_BYTES = 2 * CHUNK * PITCH


def source(spec):
    path, var = spec.split(":")
    spec_ = importlib.util.spec_from_file_location("m", os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(m)
    return getattr(m, var)


def c2py(expr):
    return re.sub(r"(?<![/])/(?![/])", "//", expr)


def grab(src, pattern, what):
    m = re.search(pattern, src, flags=re.S)
    assert m, f"could not parse {what}"
    return m.group(1).strip()


def parse(src):
    p = {}
    p["slots"] = c2py(grab(src, r"constexpr int PT_SLOTS = ([^;]+);", "PT_SLOTS"))
    p["slot_p"] = c2py(grab(src, r"auto pt_slot = \[&\]\(int j, int e\) -> int\* \{\s*const int p = ([^;]+);", "slot p"))
    p["addr"] = c2py(grab(src, r"return reinterpret_cast<int\*>\(ring \+ (.+?)\);\s*\};", "slot address"))
    issue = grab(src, r"auto issue = \[&\]\(int i\) \{(.+?)\n  \};", "issue body")
    p["pt_lead"] = c2py(grab(issue, r"issue_pt\(i \+ ([^)]+)\);", "issue_pt lead"))
    p["rows_from_slot"] = c2py(grab(issue, r"\*pt_slot\(([^,]+), lane & 15\)", "row read chunk"))
    p["pro_pt"] = c2py(grab(src, r"for \(int j = 0; j < ([^;]+); \+\+j\) issue_pt\(j\);", "prologue entries"))
    # (D3p: the PDL wait and the query loads sit between the commit and the wait; no cp.async there)
    p["pro_wait"] = grab(src, r"issue_pt\(j\);\s*asm volatile\(\"cp.async.commit_group;\"\);(?:(?!cp\.async).)*?asm volatile\(\"cp.async.wait_group (\d+);\"\);\s*__syncwarp\(\);", "prologue wait")
    p["pro_kv"] = c2py(grab(src, r"for \(int i = 0; i < ([^;]+); \+\+i\) issue\(i\);", "prologue stages"))
    p["loop_issue"] = c2py(grab(src, r"for \(int i = 0; i < mine; \+\+i\) \{\s*issue\(i \+ ([^)]+)\);", "loop issue"))
    p["loop_wait"] = c2py(grab(src, r"issue\(i \+ [^)]+\);\s*asm volatile\(\"cp.async.wait_group %0;\" ::\"n\"\(([^)]+)\)\);\s*__syncwarp\(\);", "loop wait"))
    return p


def run(p, S, mine):
    """Replays one warp; returns a list of violations."""
    env = dict(STAGES=S, STAGE_BYTES=STAGE_BYTES, PITCH=PITCH, CHUNK=CHUNK)
    slots = eval(p["slots"], {}, env)
    bad = []
    # layout checks
    used = {}
    for slot in range(slots):
        for e in range(16):
            j = slot
            pp = eval(p["slot_p"], {}, dict(env, j=j, PT_SLOTS=slots))
            addr = eval(p["addr"], {}, dict(env, p=pp, e=e))
            stage, off = divmod(addr, STAGE_BYTES)
            if not (0 <= stage < S) or addr % 4:
                bad.append(f"S{S}: entry ({slot},{e}) at {addr} outside the ring or misaligned")
            if off >= CHUNK * PITCH or off % PITCH < 128:
                bad.append(f"S{S}: entry ({slot},{e}) at stage offset {off} is not in a K-tile pad")
            if addr in used:
                bad.append(f"S{S}: entries {used[addr]} and ({slot},{e}) share byte {addr}")
            used[addr] = (slot, e)
    if bad:
        return bad

    def entry_addrs(j):
        pp = eval(p["slot_p"], {}, dict(env, j=j, PT_SLOTS=slots))
        return [eval(p["addr"], {}, dict(env, p=pp, e=e)) for e in range(16)]

    def kv_addrs(i):
        base = (i % S) * STAGE_BYTES
        return [base + r * PITCH + c for r in range(2 * CHUNK) for c in range(0, 128, 16)]

    # byte region -> (writer group, what, consumed?)
    mem = {}
    groups = []          # list of lists of (kind, chunk)
    open_group = []
    waited = -1          # groups[0..waited] complete (and warp-synced right after)
    lead = eval(p["pt_lead"], {}, env)

    def write(addrs, what):
        for a in addrs:
            prev = mem.get(a)
            if prev is not None and not prev[2]:
                bad.append(f"S{S} mine{mine}: {what} overwrites unread {prev[1]} at {a}")
                return
        for a in addrs:
            mem[a] = [len(groups), what, False]

    def read(addrs, what):
        for a in addrs:
            cur = mem.get(a)
            if cur is None or cur[1] != what:
                bad.append(f"S{S} mine{mine}: {what} reads {cur[1] if cur else 'nothing'} at {a}")
                return
            if cur[0] > waited:
                bad.append(f"S{S} mine{mine}: {what} read before its group {cur[0]} landed (waited {waited})")
                return
        for a in addrs:
            mem[a][2] = True

    def issue_pt(j):
        if j < mine:
            write(entry_addrs(j), f"PT{j}")
            open_group.append(("pt", j))

    def commit():
        groups.append(list(open_group))
        open_group.clear()

    def wait(n_pending):
        nonlocal waited
        waited = max(waited, len(groups) - 1 - n_pending)

    def issue(i):
        if i < mine:
            jr = eval(p["rows_from_slot"], {}, dict(env, i=i))
            if jr != i:
                bad.append(f"S{S}: issue({i}) reads chunk {jr}'s page entries")
            read(entry_addrs(jr), f"PT{jr}")
            write(kv_addrs(i), f"KV{i}")
            open_group.append(("kv", i))
        issue_pt(i + lead)
        commit()

    for j in range(eval(p["pro_pt"], {}, env)):
        issue_pt(j)
    commit()
    wait(int(p["pro_wait"]))
    for i in range(eval(p["pro_kv"], {}, env)):
        issue(i)
    for i in range(mine):
        issue(i + eval(p["loop_issue"], {}, env))
        wait(eval(p["loop_wait"], {}, env))
        read(kv_addrs(i), f"KV{i}")
        if bad:
            return bad
    return bad


def check(src, label):
    p = parse(src)
    problems = []
    for S in (2, 3, 4):
        for mine in range(0, 41):
            problems += run(p, S, mine)
            if problems:
                break
    print(("ok: " if not problems else "FAIL: ") + label + ("" if not problems else f" -> {problems[0]}"))
    return not problems


if __name__ == "__main__":
    specs = sys.argv[1:] or DEFAULT
    good = all([check(source(s), s) for s in specs])
    print("sim_pt_ring:", "ok" if good else "violations")
    sys.exit(0 if good else 1)

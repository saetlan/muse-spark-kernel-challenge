#!/usr/bin/env python3
"""Build the data files of the Open Kernel Challenge standings dashboard from okc.

Usage: python3 build_data.py <out_dir>

Reads (read-only okc commands): okc leaderboard <problem> --json for the five problems,
okc jobs --json, okc problems --json. Merges the jobs into jobs_cache.json (okc only returns the
last 50 jobs, so the cache keeps the older ones), joins them with notes_detail/detail_<track>.json
(what each run implemented, extracted from the track notes), reads each track's best scored run with okc status <job>
--json (cached in job_results/), and writes into <out_dir>:
  standings.json  quota.json  history.json  runs_detail.json  bottlenecks.json
Prints a summary of what moved. Never launches a check or a scored run.
"""
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bottlenecks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OKC = os.path.expanduser("~/.okc/bin/okc")
NAMES = {
    "muse_bf16_proj": "bf16_proj",
    "muse_attention_decode": "attention_decode",
    "muse_moe_w4a16": "moe_w4a16",
    "muse_decode_layer": "decode_layer",
    "muse_prefill_layer": "prefill_layer",
}


def okc(*args):
    r = subprocess.run([OKC, *args], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"okc {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout)


def recover_scored(cache):
    """okc jobs only lists the last 50 jobs of all tracks, so a busy stretch can push scored runs out before a refresh
    caches them. Every scored-run ID named in a track's notes (`scored `<16 hex>``) that the cache lacks is fetched once
    with okc status (read-only) and cached like a listed job. Returns the 8-char IDs recovered."""
    import re
    found = []
    for t in NAMES.values():
        notes = os.path.join(ROOT, f"muse_{t}.md")
        if not os.path.exists(notes):
            continue
        ids = set(re.findall(r"scored[^`|]{0,40}`([0-9a-f]{16})`", open(notes).read()))
        for jid in sorted(ids - set(cache)):
            r = subprocess.run([OKC, "status", jid, "--json"], capture_output=True, text=True, cwd=ROOT)
            if r.returncode != 0:
                continue
            d = json.loads(r.stdout)
            res = d.get("result") or {}
            if d.get("mode") != "run" or d.get("board") not in NAMES:
                continue
            cache[jid] = {"job_id": jid, "problem": d["board"], "mode": "run", "status": d.get("status", "done"),
                          "passed": res.get("passed"), "score_us": res.get("score_us"),
                          "submitted_at": d.get("created_at") or 0}
            found.append(jid[:8])
    return found


def when(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%b %-d %H:%M")


def main(out):
    os.makedirs(out, exist_ok=True)
    boards = {p: okc("leaderboard", p, "--json") for p in NAMES}
    problems = {p["name"]: p for p in okc("problems", "--json")["problems"]}
    jobs = okc("jobs", "--json")

    # speedup of a check = leader's baseline time / our time (the leader row gives time and speedup)
    base = {p: b["rows"][0]["value"] * b["rows"][0]["speedup"] for p, b in boards.items()}

    cache_path = os.path.join(HERE, "jobs_cache.json")
    cache = {j["job_id"]: j for j in json.load(open(cache_path))}
    new = [j["job_id"][:8] for j in jobs if j["job_id"] not in cache]
    for j in jobs:
        cache[j["job_id"]] = j  # fresher status wins
    recovered = recover_scored(cache)
    allj = sorted(cache.values(), key=lambda j: -j["submitted_at"])
    json.dump(allj, open(cache_path, "w"))

    det = {}
    for t in NAMES.values():
        p = os.path.join(HERE, "notes_detail", f"detail_{t}.json")
        det[t] = json.load(open(p)) if os.path.exists(p) else []
    bykey = {e["job"][:8]: e for L in det.values() for e in L}

    rows, seen = [], set()
    for j in allj:
        if j["status"] == "duplicate":
            continue
        t, k = NAMES[j["problem"]], j["job_id"][:8]
        e = bykey.get(k, {})
        sp = round(base[j["problem"]] / j["score_us"], 4) if j.get("score_us") else e.get("speedup")
        status = "failed" if j["status"] == "done" and j["passed"] is False else ("passed" if j["status"] == "done" else j["status"])
        rows.append(dict(track=t, kind="scored" if j["mode"] == "run" else "check", status=status, when=when(j["submitted_at"]),
                         sort=j["submitted_at"], job=k, speedup=sp, name=e.get("name", ""), file=e.get("file", ""), verdict=e.get("verdict", "")))
        seen.add(k)
    earliest = min(r["sort"] for r in rows)
    for ti, (t, L) in enumerate(det.items()):
        for i, e in enumerate(x for x in L if x["job"][:8] not in seen):
            rows.append(dict(track=t, kind=e["kind"], status="passed" if e.get("speedup") else "no score", when=e["when"] or "earlier",
                             sort=earliest - 1e6 + i * 10 + ti, job=e["job"][:8], speedup=e.get("speedup"), name=e.get("name", ""),
                             file=e.get("file", ""), verdict=e.get("verdict", "")))
    rows.sort(key=lambda r: -r["sort"])
    json.dump(rows, open(os.path.join(out, "history.json"), "w"))

    # scored-run details; a scored run the notes don't cover yet gets an empty card (re-extract, see SKILL.md)
    scored, covered = [], set()
    for t, L in det.items():
        for e in L:
            if e["kind"] == "scored":
                covered.add(e["job"][:8])
                scored.append(dict(track=t, job=e["job"][:8], when=e["when"], name=e["name"], file=e["file"], speedup=e.get("speedup"),
                                   implemented="\n".join(e["implemented"]), worked="\n".join(e["worked"]), failed="\n".join(e["failed"]), verdict=e["verdict"]))
    missing = []
    for r in reversed(rows):
        if r["kind"] == "scored" and r["job"] not in covered and r["status"] in ("passed", "failed"):
            missing.append((r["track"], r["job"]))
            scored.append(dict(track=r["track"], job=r["job"], when=r["when"], name=r["name"] or "new scored run", file=r["file"], speedup=r["speedup"],
                               implemented="", worked="", failed="", verdict="Not extracted from the track notes yet."))
    json.dump(scored, open(os.path.join(out, "runs_detail.json"), "w"))

    standings = []
    for p, n in NAMES.items():
        b, y = boards[p], boards[p]["you"]
        rws = b["rows"]
        top = lambda k: next((r["speedup"] for r in rws if r["rank"] == k), None)
        sc = [h for h in rows if h["track"] == n and h["kind"] == "scored" and h["speedup"]]
        best = max(sc, key=lambda h: h["speedup"]) if sc else None
        ck = [h for h in rows if h["track"] == n and h["kind"] == "check" and h["speedup"] and h["status"] == "passed"]
        bc = max(ck, key=lambda h: h["speedup"]) if ck else None
        live = [h for h in rows if h["track"] == n and h["status"] in ("running", "queued")]
        standings.append(dict(track=n, rank=y["rank"], ranked=b["ranked"], ours=y["speedup"], leader_team=b["leader"]["team"],
                              leader=b["leader"]["speedup"], gap=round(b["leader"]["speedup"] - y["speedup"], 4), top10=top(10), top100=top(100),
                              best_check=bc["speedup"] if bc else None, best_check_job=bc["job"] if bc else "",
                              best_check_gain_pct=round((bc["speedup"] / y["speedup"] - 1) * 100, 1) if bc else None,
                              board_job=best["job"] if best else "", board_file=best["file"] if best else "", in_flight=len(live)))
    json.dump(standings, open(os.path.join(out, "standings.json"), "w"), indent=1)
    json.dump([dict(track=NAMES[k], checks_used=v["limits"]["checks"]["used"], checks_left=v["limits"]["checks"]["left"],
                    scored_used=v["limits"]["scored"]["used"], scored_left=v["limits"]["scored"]["left"]) for k, v in problems.items()],
              open(os.path.join(out, "quota.json"), "w"))

    full_ids = {j["job_id"][:8]: j["job_id"] for j in allj}
    when_of = {r["job"]: r["when"] for r in rows}
    bn_rows, bn_report = bottlenecks.build(standings, out, full_ids, when_of)

    print("as of", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    for s in standings:
        print(f"{s['track']:18s} rank {s['rank']:>4}/{s['ranked']:<4} ours {s['ours']:.4f}x  #1 {s['leader']:.4f}x  best check "
              f"{s['best_check']}  in flight {s['in_flight']}")
    print("new jobs since last refresh:", ", ".join(new) or "none", "(okc lists the last 50: older ones may be missing)" if len(new) >= 50 else "")
    print("scored runs recovered from the notes:", ", ".join(recovered) or "none")
    print("scored runs without notes detail:", ", ".join(f"{t}/{j}" for t, j in missing) or "none")
    print("time split (bottlenecks.json) from each track's best scored run:")
    for line in bn_report:
        print("  " + line)
    print("rows:", len(rows), "history,", len(scored), "scored details,", len(bn_rows), "time-split rows; files in", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/okc_dashboard")

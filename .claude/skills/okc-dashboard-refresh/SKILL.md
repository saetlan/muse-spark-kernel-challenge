---
name: okc-dashboard-refresh
description: Refresh the Open Kernel Challenge standings dashboard (ranks, speedups, best checks, run limits, full run history, scored-run details, and the per-kernel time split of each track's best scored run) from okc. Use when the user says refresh/update the standings, dashboard or leaderboard page, or after a scored run finishes. Read-only on okc: it never launches a check or a scored run.
---

# Refresh the okc standings dashboard

The dashboard is a private claude.ai Artifact (type Dashboard):
`https://claude.ai/artifact/4TLjrhe4KDF4a7wT3dbCmT`. Its numbers are files uploaded to the artifact, so
it does **not** update by itself. This skill rebuilds every data file the page reads and swaps them in.
The page code (`files/index.html`) only gets its snapshot time changed.

No spending: only `okc leaderboard`, `okc jobs`, `okc problems`, `okc status` (read-only). Do not run `okc check`,
`okc submit` or anything on Modal here (CLAUDE.md rule 1). Run `okc` as `~/.okc/bin/okc`.

## Steps

1. **Build the data.** From the project root, with a scratch folder from the session:
   `python3 .claude/skills/okc-dashboard-refresh/build_data.py <scratchpad>/dash_data`
   It writes `standings.json`, `quota.json`, `history.json`, `runs_detail.json`, `bottlenecks.json` and
   prints a summary: ranks and speedups, jobs new since the last refresh, scored runs the notes don't
   cover yet, and one time-split line per track (see "Time split" below). Read the time-split lines:
   a "kernels matching no rule" or "per-load speedups give ... but the judge says" line needs a fix in
   `bottlenecks.py` before you upload.
   It also updates `jobs_cache.json` next to it: okc only returns the last 50 jobs (all tracks), so the cache
   keeps older ones. Never delete the cache. When more than 50 jobs ran since the last refresh, some fall out
   of that window before they're cached: the build then fetches every scored-run ID named in the track notes
   (`scored `<16 hex>``) that the cache lacks with `okc status` and caches it ("scored runs recovered from the
   notes" line), so board runs and their time split stay right. Checks lost that way stay missing. `job_results/` caches `okc status` of each board run (results
   never change); never delete it either.
2. **Upload the five files as assets** to the artifact: `Artifact` with `action: "publish"`,
   `url` = the dashboard, `asset: true`, `file_path` = each file (JSON files go one per call; send the
   five calls in the same message). Each result gives a `/_blob/<id>` url.
3. **Point the datasets at the new files.** `ArtifactData` `action: "list"`, `collection: "datasets"`
   to get each document's `version`, then one `batch` of `update`s (pin each with `if_version`):

   | dataset id | file | title |
   |---|---|---|
   | `standings` | `standings.json` | Track standings |
   | `quota` | `quota.json` | Daily run limits |
   | `jobs` | `history.json` | Full run history |
   | `runs_detail` | `runs_detail.json` | Scored run details |
   | `bottlenecks` | `bottlenecks.json` | Time split per kernel |

   Each update sets `source: {kind: "file", url: "<new /_blob url>", name: "<file name>", format: "json"}`,
   `updated: {at: "<UTC ISO now>", by: "okc, via Claude"}` and `edit: {via: "claude", at: "<UTC ISO now>"}`.
4. **Delete the old assets** (`Artifact` `action: "list"`, `scope: "assets"`, then `action: "delete"` with
   the asset id as `path`) for the five files you just replaced, so they don't pile up.
5. **Update the snapshot time in the page.** The header note says "Snapshot of okc at about HH:MM UTC".
   `ArtifactData` `get` `files/index.html`, then `str_replace` the old time with the new one (pin
   `if_version`). Change nothing else in the page.
6. **Report**: what moved since the last refresh (rank, score, finished jobs, new scored runs), and any
   scored run listed under "without notes detail" (step 7).
7. **Only if asked, or when the summary lists scored runs without notes detail:** re-extract what the
   run implemented. Spawn one agent for that track (see `notes_detail/` for the JSON shape: job, when,
   kind, file, speedup, name, implemented, worked, failed, verdict). The agent reads only that track's
   notes file (`muse_<track>.md` at the project root) and writes `detail_<track>.json`; copy it over
   `notes_detail/detail_<track>.json`, then redo steps 1-3. Facts only from the notes; leave unknown
   fields empty. Notes can lag a run, so expect gaps right after a scored run.

## Time split (`bottlenecks.json`, built by `bottlenecks.py`)

The "Where the time goes" cards and the "Inside the H200" view read it. For each track it takes the
best scored run (the standings' `board_job`) and its `okc status <job> --json` result: per timed case,
our time, SGLang's, the speed of light, and the judge's per-kernel profile.
- bf16_proj: one bar per m, one segment per layer (judged time).
- Other tracks: one bar per case, kernels grouped into stages by `STAGE_RULES` (regexes on kernel
  names, first match wins). The stage names of decode_layer and prefill_layer must stay those the
  page's `STAGE_UNITS` knows. A kernel no rule matches goes to "other kernels" and the build reports it:
  add a rule (the judge truncates long names, so match a prefix).
- Every row also carries the case's judged load, weight, our / SGLang / speed-of-light times, and the
  time in kernels we didn't write (`LIB`: SGLang's own, cuBLAS, CUTLASS / FlashAttention, DeepGEMM,
  Marlin, torch). The card line under each chart turns these into the per-load speedups the judge
  scores (TASK.md: a load's time is sum(weight x time) over its shapes; the score is the geometric
  mean over loads), and the build checks they reproduce the run's speedup.
- A new board run is fetched once; text on the page that names kernels comes from the data, so a
  new run needs no page edit.

## What is not refreshed here

- Checks that a track runs between refreshes appear once okc lists them. Runs older than the last 50
  that were never cached come only from the notes (often a day, no file name).
- The time split covers scored runs only (checks don't reach the board).

## Data rules

- Speedup of a check = the leader's time x the leader's speedup (the baseline), divided by our time.
  This can differ from the judge's number by about 0.1%.
- "Best check" is the best passed check of any day; a track whose best run was scored directly (MoE)
  shows a check that may be lower than its best scored run.
- "In flight" counts jobs that are running or queued at the moment of the refresh.

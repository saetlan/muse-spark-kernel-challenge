"""bottlenecks.json for the dashboard's "Where the time goes" and "Inside the H200" sections, rebuilt from each track's best
scored run (the board entry) at every refresh, instead of a hand-made snapshot.

Source: `okc status <job> --json` (read-only). Its result holds, per timed case, a `benchmark` record (median_ns = our time,
baseline_ns = SGLang's, speed_of_light_ns = the judge's bound) and a `profile` record (the judge's per-kernel profile text).
Results never change, so they are cached in job_results/<job>.json (only the records this needs).

Rows: {track, load, stage, order, us, total, ref, ref_kind, job, job_speedup, job_when, kernels, lib_us, lib_top, lib_top_us,
       case_id, judge_load, weight, case_us, sglang, sol}
  bf16_proj      one case per layer: stage = layer, us = the layer's judged time, total = sum of the five, ref = sum of SoL
  other tracks   one case per load: stages = the profile's kernels grouped by STAGE_RULES (first matching rule wins, unmatched
                 kernels go to the track's catch-all stage and are reported), us = profiled kernel time, total = the judged
                 time, ref = SoL (attention, MoE) or SGLang's time (decode and prefill layers). Profiled kernel times can
                 exceed or fall short of the judged total (overlap, launch gaps); the page shows the total at the end.
  every track    kernels = profiled kernels in the load; lib_us = µs of the stage in kernels we didn't write (LIB: SGLang's own,
                 cuBLAS, CUTLASS / FlashAttention, DeepGEMM, Marlin, torch); lib_top / lib_top_us = the largest of those, named by
                 LIB_NAMES (bf16_proj: the layer's judged time split in the ratio of its profile).
                 case_id = the timed case the row belongs to; judge_load / weight / case_us / sglang / sol = that case's load, weight,
                 our time, SGLang's time and speed-of-light time (µs). The judge's score (TASK.md "Scoring"): a load's time is
                 sum(weight x time) over its cases, the speedup is the geometric mean over loads of SGLang's load time over ours;
                 build() checks that these fields reproduce the run's speedup.
Stage names for the layer tracks must stay those the page's H200 view knows (STAGE_UNITS in index.html).
"""
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "job_results")
OKC = os.path.expanduser("~/.okc/bin/okc")
BF16_LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]
REF_KIND = {"bf16_proj": "SoL", "attention_decode": "SoL", "moe_w4a16": "SoL", "decode_layer": "SGLang", "prefill_layer": "SGLang"}


def _moe_gemm(name):
    """MoE's own 4-bit GEMMs (moe/ submissions): GATED is template arg 5 of gemm / occ / nu / dyn_kernel and arg 3 of ck / ss / wr_kernel."""
    m = re.search(r"(gemm|occ|nu|dyn|ck|ss|wr)_kernel<([^>]*)>", name)
    if not m or not re.search(r"\bm\d+b?::", name):
        return None
    args = [a.strip() for a in m.group(2).split(",")]
    i = 4 if m.group(1) in ("gemm", "occ", "nu", "dyn") else 2
    if len(args) > i and args[i] in ("true", "false"):
        return "W13 GEMM" if args[i] == "true" else "W2 GEMM"
    return None


ATTN = r"attend|okc_ud\d|okc_uw\d|ud4|uw2|w1n::|w2::|w3b::|dec4|att__|flash|FlashAttn|_prologue|_combine\b|^_combine"
EXPERTS = (r"\bm\d+b?::|marlin|act_and_mul|_post_expert|r14::|_r14_|_nvfp4_dequant|_gmm_|_gather_rows|_gather_routes|route_kernel|"
           r"scatter_gather|enable_3x_ke|moe")
PROJ = r"f12v|f12m|pj__|p9::|dgrs|nvjet|tiny_n_gemm|tinygemm|deep_gemm|splitKreduce|_hl_|_fma_kernel|_swap_kernel|cublas"
# kernels we didn't write (our own are namespaced okc_* / m19:: / f12m:: / dgrs:: ... or are Triton kernels named _*)
LIB = r"sglang::|marlin|flash|FlashAttn|nvjet|cublas|cutlass|deep_gemm|\bat::|_ZN2at|^(void )?tinygemm_kernel"
LIB_NAMES = [(r"marlin", "SGLang Marlin MoE GEMM"), (r"flash|FlashAttn", "FlashAttention 3"), (r"nvjet|cublas", "cuBLAS GEMM"),
             (r"deep_gemm", "DeepGEMM BF16 GEMM"), (r"at4cuda6detail|at::cuda::detail", "torch CUTLASS kernel (at::cuda)"),
             (r"norm_residual", "SGLang norm + residual"), (r"routing_kernel", "SGLang routing"),
             (r"tiny_n_gemm|tinygemm", "SGLang tiny GEMM"), (r"at::native|_ZN2at", "torch elementwise")]


def lib_name(name):
    return next((label for rule, label in LIB_NAMES if re.search(rule, name)), name[:40])
GLUE = r"norm|routing|local_ids|elementwise|vectorized_layer_"   # the judge truncates long names
# track -> (ordered [(stage, regex or callable)], catch-all stage); every stage has a rule, so the catch-all ("other kernels")
# only collects kernels nobody classified yet, and the build reports them
STAGE_RULES = {
    "attention_decode": ([("prologue + combine + gaps", r"prologue|combine"), ("attend", ATTN)], "other kernels"),
    "moe_w4a16": ([("norm + combine", r"norm|combine|weighted_sum"), ("SGLang Marlin (both GEMMs)", r"marlin"),
                   ("W13 GEMM", lambda n: _moe_gemm(n) == "W13 GEMM"), ("W2 GEMM", lambda n: _moe_gemm(n) == "W2 GEMM")],
                  "other kernels"),
    "decode_layer": ([("attention", ATTN), ("experts", EXPERTS), ("5 projections", PROJ), ("norms + routing + glue", GLUE)],
                     "other kernels"),
    "prefill_layer": ([("experts", EXPERTS), ("attention", ATTN), ("dense projections", PROJ), ("norms + glue", GLUE)],
                      "other kernels"),
}
STAGE_ORDER = {
    "attention_decode": ["prologue + combine + gaps", "attend", "other kernels"],
    "moe_w4a16": ["W13 GEMM", "W2 GEMM", "SGLang Marlin (both GEMMs)", "norm + combine", "other kernels"],
    "decode_layer": ["attention", "5 projections", "experts", "norms + routing + glue", "other kernels"],
    "prefill_layer": ["experts", "dense projections", "attention", "norms + glue", "other kernels"],
}


def job_result(job_id):
    """The benchmark and profile records of a finished job, from the cache or okc status (read-only)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, job_id + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    r = subprocess.run([OKC, "status", job_id, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"okc status {job_id}: {r.stderr.strip() or r.stdout.strip()}")
    d = json.loads(r.stdout)
    res = d.get("result") or {}
    keep = {"job": job_id, "board": d.get("board"), "speedup": res.get("speedup"), "score_us": res.get("score_us"),
            "records": [{k: e.get(k) for k in ("event", "case", "median_ns", "baseline_ns", "speed_of_light_ns", "report")}
                        for e in res.get("records") or [] if e.get("event") in ("benchmark", "profile")]}
    if not keep["records"]:
        raise RuntimeError(f"okc status {job_id}: no benchmark / profile records")
    json.dump(keep, open(path, "w"))
    return keep


def parse_profile(report):
    """[(kernel name, µs)] from the judge's profile text ('name  12.3 µs  45.6%  calls'); the total line is skipped."""
    out = []
    for line in (report or "").splitlines()[1:]:
        m = re.match(r"(.*?)\s+([\d.,]+) (µs|ms|ns)\s+[\d.]+%\s+\d+\s*$", line)
        if not m or line.startswith("total"):
            continue
        v = float(m.group(2).replace(",", "")) * {"µs": 1.0, "ms": 1e3, "ns": 1e-3}[m.group(3)]
        out.append((m.group(1).strip(), v))
    return out


def stage_of(track, name):
    rules, rest = STAGE_RULES[track]
    for stage, rule in rules:
        if (rule(name) if callable(rule) else re.search(rule, name)):
            return stage, True
    return rest, False


def case_meta(e, case_id):
    c = e["case"]
    return dict(case_id=case_id, judge_load=c.get("load", ""), weight=c.get("weight", 1), case_us=round(e["median_ns"] / 1e3, 2),
                sglang=round(e["baseline_ns"] / 1e3, 2), sol=round(e["speed_of_light_ns"] / 1e3, 2))


def judge_speedup(bench):
    """Geometric mean over loads of SGLang's load time over ours, a load's time being sum(weight x time) (TASK.md)."""
    loads = {}
    for e in bench:
        w = e["case"].get("weight", 1)
        a = loads.setdefault(e["case"].get("load", ""), [0.0, 0.0])
        a[0] += w * e["baseline_ns"]
        a[1] += w * e["median_ns"]
    return math.exp(sum(math.log(b / o) for b, o in loads.values()) / len(loads)) if loads else 0.0


def load_label(track, case):
    load, layer = case.get("load", ""), case.get("layer", "")
    if track == "moe_w4a16":
        bs = case.get("bs", 0)
        return f"{load} ({bs} token{'s' if bs != 1 else ''})" if (load.startswith("verify") or bs == 1) else load
    if track == "prefill_layer":
        if load == "first turn":
            ctx = case.get("ctx", 0)
            return f"first turn ctx {ctx // 1024}K {layer}" if ctx else f"first turn ctx 0 {layer}"
        return f"{load} {layer} ({case.get('q_len')} tok)"
    return f"{load} {layer}".strip()


def build(standings, out_dir, jobs_by_short, when_of):
    """standings: rows with track / board_job (8-char id); jobs_by_short: 8-char id -> full job id. Returns (rows, report)."""
    rows, report = [], []
    for s in standings:
        track, short = s["track"], s.get("board_job")
        if not short or short not in jobs_by_short:
            report.append(f"{track}: no scored run to read")
            continue
        jr = job_result(jobs_by_short[short])
        bench = [e for e in jr["records"] if e["event"] == "benchmark"]
        prof = {json.dumps(e["case"], sort_keys=True): e["report"] for e in jr["records"] if e["event"] == "profile"}
        meta = dict(job=short, job_speedup=round(jr["speedup"], 4) if jr.get("speedup") else s.get("ours"), job_when=when_of.get(short, ""))
        lib_total = prof_total = 0.0
        kind = REF_KIND[track]
        unmatched = {}
        if track == "bf16_proj":
            loads = {}
            for e in bench:
                loads.setdefault((e["case"]["m"], e["case"]["load"]), []).append(e)
            for (m, _), es in sorted(loads.items()):
                total = sum(e["median_ns"] for e in es) / 1e3
                ref = sum(e["speed_of_light_ns"] for e in es) / 1e3
                for e in sorted(es, key=lambda e: BF16_LAYERS.index(e["case"]["layer"])):
                    us = e["median_ns"] / 1e3
                    cm = case_meta(e, bench.index(e))
                    ks = parse_profile(prof.get(json.dumps(e["case"], sort_keys=True), ""))
                    lib = [(n, v) for n, v in ks if re.search(LIB, n)]
                    scale = us / sum(v for _, v in ks) if ks else 0.0
                    top = max(lib, key=lambda kv: kv[1]) if lib else None
                    lib_total += sum(v for _, v in lib) * scale
                    prof_total += us
                    rows.append(dict(track=track, load=f"m = {m}", stage=e["case"]["layer"], order=BF16_LAYERS.index(e["case"]["layer"]),
                                     us=round(us, 2), total=round(total, 2), ref=round(ref, 2), ref_kind=kind, kernels=len(ks),
                                     lib_us=round(sum(v for _, v in lib) * scale, 2), lib_top=lib_name(top[0]) if top else "",
                                     lib_top_us=round(top[1] * scale, 2) if top else 0.0, **cm, **meta))
        else:
            order = STAGE_ORDER[track]
            for ci, e in enumerate(bench):
                cm = case_meta(e, ci)
                total = e["median_ns"] / 1e3
                ref = (e["speed_of_light_ns"] if kind == "SoL" else e["baseline_ns"]) / 1e3
                ks = parse_profile(prof.get(json.dumps(e["case"], sort_keys=True), ""))
                label = load_label(track, e["case"])
                if not ks:
                    rows.append(dict(track=track, load=label, stage="all kernels (no profile)", order=99, us=round(total, 2),
                                     total=round(total, 2), ref=round(ref, 2), ref_kind=kind, kernels=0, lib_us=0.0, lib_top="",
                                     lib_top_us=0.0, **cm, **meta))
                    continue
                st, lib, top = {}, {}, {}
                for name, v in ks:
                    stage, hit = stage_of(track, name)
                    st[stage] = st.get(stage, 0.0) + v
                    prof_total += v
                    if re.search(LIB, name):
                        lib[stage] = lib.get(stage, 0.0) + v
                        lib_total += v
                        if v > top.get(stage, ("", 0.0))[1]:
                            top[stage] = (lib_name(name), v)
                    if not hit:
                        unmatched[name[:60]] = unmatched.get(name[:60], 0.0) + v
                for stage, v in sorted(st.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 98):
                    rows.append(dict(track=track, load=label, stage=stage, order=order.index(stage) if stage in order else 98,
                                     us=round(v, 2), total=round(total, 2), ref=round(ref, 2), ref_kind=kind, kernels=len(ks),
                                     lib_us=round(lib.get(stage, 0.0), 2), lib_top=top.get(stage, ("", 0.0))[0],
                                     lib_top_us=round(top.get(stage, ("", 0.0))[1], 2), **cm, **meta))
        sp = judge_speedup(bench)
        if jr.get("speedup") and abs(sp / jr["speedup"] - 1) > 1e-3:
            report.append(f"{track}: per-load speedups give {sp:.4f}x but the judge says {jr['speedup']:.4f}x: "
                          "the scoring rule in case_meta / judge_speedup no longer matches TASK.md")
        top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:4]
        rest = STAGE_RULES.get(track, ([], None))[1]
        report.append(f"{track}: job {short} ({meta['job_speedup']}x), {len(bench)} cases, "
                      f"{100 * lib_total / prof_total if prof_total else 0:.0f}% of profiled time in SGLang / library kernels"
                      + (f"; kernels matching no rule, counted as '{rest}' (add a rule in STAGE_RULES): "
                         + ", ".join(f"{n} {v:.1f} us" for n, v in top) if top else ""))
    json.dump(rows, open(os.path.join(out_dir, "bottlenecks.json"), "w"))
    return rows, report

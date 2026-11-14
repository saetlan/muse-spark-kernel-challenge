"""CPU dry run of v12 (parts/v12.py: SGLang's cluster norm-residual kernel at the layer boundaries) and the DE14 lab.

A stand-in for sglang.kernels.ops.muse_spark_v12.norm_residual records each build (_jit_module) and each launch
(fused_norm_residual with cluster_size C, per token count); the boundary stand-in gets FusedPostNormResidualPreNorm's
attributes. Fake timings (us): SGLang's boundary 4.0 per call; cluster C per call at <= 16 tokens 3.8 / 3.0 / 2.5 /
2.8 (C 2 / 4 / 8 / 16), at 17-64 tokens 3.9 / 3.2 / 3.5 / 6.0, above 64 tokens 4.5 for every C; a layer call +300;
the attention drop-in -10, the projections +3 (as dryrun_v4).

lab      lab_de14.py: the builds run in the boundary section in the order C8 / C4 / C16; picks C8 for 1 / 8 / 16 tokens and C4 for 32,
         none at 128; the layer section keeps them (>= 0.5%); V12_BOUNDARY printed = installed; every whole-layer
         comparison ran with the routing locked (no UNLOCKED case), the real routing restored after prepare;
         kernel_fn launches C8 at 1 / 8 / 16 tokens, C4 at 32, SGLang's kernel at 128 / 256.
bad16    the same, C16 the fastest everywhere but its outputs off by 1.0: C16 must never be picked.
slow     a fake clock: each build 9.1 s (DE14's), each timed graph 0.2 s, lab budget 50 s (section deadline 35 s): C8
         and C4 are built, C16 skipped for time, every token count still timed, the same picks.
clean    submission_v12.py with V12_BOUNDARY from DRYRUN_V12_LOG (or {}): prepare builds exactly the picked sizes
         (after the glue's prepare), checks and installs them; kernel_fn follows.
cleanbad the clean file with a failing build: v12 falls back to SGLang's dispatch everywhere.

Run: .venv/bin/python decode_layer/local/dryrun_v12.py lab_de14.py [bad16]
     DRYRUN_V12_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v12.py submission_v12.py [cleanbad]
"""

import ast
import os
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
ORDER = []
MODE = {"bad16": False, "fail_build": False, "slow": False}
CLOCK = [0.0]


def cluster_cost(tokens, size):
    if MODE["bad16"] and size == 16:
        return 1.0  # the fastest, so only the error gate can keep it out
    if tokens <= 16:
        return {2: 3.8, 4: 3.0, 8: 2.5, 16: 2.8}[size]
    if tokens <= 64:
        return {2: 3.9, 4: 3.2, 8: 3.5, 16: 6.0}[size]
    return 4.5


def install_nr():
    m = types.ModuleType("sglang.kernels.ops.muse_spark_v12.norm_residual")

    def _cluster_launch_config(hidden, tokens, size):
        assert size in (2, 4, 8, 16) and hidden == 8192
        return min(256, hidden // (size * 8)), tokens < 192

    def _jit_module(hidden, threads, prefetch, size):
        ORDER.append(f"build C{size}")
        base.hit(f"build C{size}")
        CLOCK[0] += 9.1  # DE14's measured build time per module
        if MODE["fail_build"]:
            raise RuntimeError("nvcc failed (planted)")
        return object()

    def fused_norm_residual(branch, stream, post_eps, pre_eps, post_norm_is_input_scale=False, post_weight=None,
                            pre_weight=None, alpha=None, beta=None, want_router=False, pre_norm_bf16=False,
                            cluster_size=None, **kw):
        assert cluster_size in (2, 4, 8, 16) and not pre_norm_bf16 and branch.dtype == torch.bfloat16
        assert stream.dtype == torch.float32 and post_eps == 1e-5 and pre_eps == 1e-6
        tokens = branch.shape[0]
        base.hit(f"cluster C{cluster_size} T{tokens}")
        s = stream + branch.float() + (1.0 if MODE["bad16"] and cluster_size == 16 else 0.0)
        return s.to(torch.bfloat16), s, (s if want_router else None)

    m._cluster_launch_config, m._jit_module, m.fused_norm_residual = _cluster_launch_config, _jit_module, fused_norm_residual
    sys.modules[m.__name__] = m


def patch_boundary():
    cls = sys.modules["sglang.kernels.ops.muse_spark_v12.fused_norm_residual"].FusedPostNormResidualPreNorm
    if getattr(cls, "_v12_patched", False):
        return
    cls._v12_patched = True
    real = cls.__call__

    def call(self, stream, branch, branch_dtype, router_dtype=None):
        base.hit(f"sglang boundary T{branch.shape[0]}")
        return real(self, stream, branch, branch_dtype, router_dtype)

    cls.__call__ = call
    cls._ensure_cached_weights = lambda self, dtype: base.hit("cache weights")
    cls._post_norm = types.SimpleNamespace(eps=1e-5, has_weight=True)
    cls._pre_norm = types.SimpleNamespace(eps=1e-6, has_weight=True)
    cls._post_norm_is_input_scale = False
    cls._use_residual_gate = True
    cls._cc_post_weight = cls._cc_pre_weight = torch.ones(8192, dtype=torch.bfloat16)
    cls._cc_alpha = cls._cc_beta = torch.ones(1)


def boundary_cost(ran_before, ran_after):
    cost = 0.0
    for k, v in ran_after.items():
        n = v - ran_before.get(k, 0)
        if n <= 0:
            continue
        if k.startswith("sglang boundary T"):
            cost += 4.0 * n
        elif k.startswith("cluster C"):
            size, tokens = k[len("cluster C"):].split(" T")
            cost += cluster_cost(int(tokens), int(size)) * n
    return cost


def kernel_paths(lab, state):
    got = {}
    for tokens, (b, q, layer) in {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"), 32: (32, 1, "full"),
                                  128: (16, 8, "sliding"), 256: (32, 8, "full")}.items():
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
        got[tokens] = sorted(k for k in base.CALLS if k.startswith(("cluster C", "sglang boundary")))
    return got


def want_paths(picks):
    out = {}
    for tokens in (1, 8, 16, 32, 128, 256):
        size = next((s for below, s in sorted(picks.items()) if tokens < below), 1)
        out[tokens] = [f"cluster C{size} T{tokens}"] if size > 1 else [f"sglang boundary T{tokens}"]
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de14.py"
    MODE["bad16"] = "bad16" in sys.argv[2:]
    MODE["fail_build"] = "cleanbad" in sys.argv[2:]
    MODE["slow"] = "slow" in sys.argv[2:]
    install_nr()
    lab = dryrun_v4.load(path)
    install_nr()  # load() reinstalls the base stubs; keep ours
    patch_boundary()
    clean = not hasattr(lab, "sec_boundary")
    lines, printed = [], []
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            before = dict(base.CALLS)
            fn()
            CLOCK[0] += 0.2
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            t = 100.0 + boundary_cost(before, base.CALLS)
            if "v3 lid" in ran:
                t += 300.0 - (10.0 if "att" in ran else 0.0) + (3.0 if any(k.startswith("pj__") for k in ran) else 0.0)
            return t

        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
        if MODE["slow"]:
            real_init = lab.Lab.__init__
            lab.Lab.__init__ = lambda self, budget_s: real_init(self, 50.0)
            lab.Lab.elapsed = lambda self: CLOCK[0]
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    real_glue = lab._glue_prepare
    lab._glue_prepare = lambda w: (ORDER.append("glue prepare"), real_glue(w))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    text = "\n".join(lines)
    v12 = [p for p in printed if p.startswith("[v12]")]
    picks = state["v12_boundary"]
    paths = kernel_paths(lab, state)
    if clean:
        log_name = os.environ.get("DRYRUN_V12_LOG")
        want = {}
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V12_BOUNDARY = " in x][-1]
            want = ast.literal_eval(ln.split("V12_BOUNDARY = ", 1)[1])
        sizes = sorted(set(want.values()))
        builds = [o for o in ORDER if o.startswith("build")]
        glue_at = ORDER.index("glue prepare")
        expect = {} if MODE["fail_build"] else want
        checks = {
            "V12_BOUNDARY = the lab's printed picks": lab.V12_BOUNDARY == want,
            "builds = the picked sizes, after the glue's prepare":
                sorted(builds) == sorted(f"build C{s}" for s in sizes) and all(ORDER.index(b) > glue_at for b in builds),
            "v12 line printed once": len(v12) == 1 and ("boundaries: cluster" in v12[0]) == bool(expect),
            "installed = expected": picks == expect,
            "kernel_fn follows it (1-256 tokens)": paths == want_paths(expect),
            "no lab code in the file": not hasattr(lab, "sec_boundary") and not hasattr(lab, "Lab"),
        }
    else:
        want = {2: 8, 9: 8, 17: 8, 33: 4}
        printed_picks = [ln for ln in lines if ln.startswith("V12_BOUNDARY = ")]
        builds = [o for o in ORDER if o.startswith("build")]
        glue_at = ORDER.index("glue prepare")
        checks = {
            "builds C8, C4, C16 in that order (slow: C8, C4), in the lab (after the submission's prepare)":
                builds == (["build C8", "build C4"] if MODE["slow"] else ["build C8", "build C4", "build C16"])
                and all(ORDER.index(b) > glue_at for b in builds),
            "slow: C16 skipped for time, every token count timed": (not MODE["slow"]) or (
                "C16 skipped (time)" in text and "boundary stopped" not in "; ".join(lab.LAB_SUMMARY)
                and all(f"boundary T{t} " in "; ".join(lab.LAB_SUMMARY) for t in (1, 8, 16, 32, 128))),
            "boundary picks (fake timings)": picks == want,
            "C16 never picked" + (" (bad outputs)" if MODE["bad16"] else ""): 16 not in picks.values(),
            "bad C16 shows its error": (not MODE["bad16"]) or ("1e+00" in text or "2e-01" in text or "e-01" in text),
            "V12_BOUNDARY printed = state": bool(printed_picks)
                and printed_picks[-1] == "V12_BOUNDARY = " + repr(dict(sorted(picks.items()))),
            "layer section kept every pick": all(f"layer <{b} kept" in "; ".join(lab.LAB_SUMMARY) for b in want),
            "routing locked in every whole-layer comparison": "UNLOCKED" not in text and "modes, routing locked" in text,
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "kernel_fn follows the picks (1-256 tokens)": paths == want_paths(picks),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v12 ({'clean file' if clean else 'lab'}{' bad16' if MODE['bad16'] else ''}{' slow' if MODE['slow'] else ''}"
          f"{' cleanbad' if MODE['fail_build'] else ''}): all checks passed (picks {picks})")


if __name__ == "__main__":
    main()

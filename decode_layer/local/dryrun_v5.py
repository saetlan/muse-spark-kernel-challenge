"""CPU dry run of the DE7 lab (lab_de7.py = submission_v5 + lab/common.py + lab/de7.py), scenario-driven:
the M11 extension is a stub that takes the L8 pf argument (validated) and logs every launch's config; fake
timings make pf / M11 faster or slower per scenario, and each scenario checks the exact outcome the lab
must reach, then that the timed kernel_fn launches M11 with the installed table's configs.

Scenarios (DRYRUN_SCENARIO): normal (pf wins by a lot; M11 beats Marlin at 128/256), dirty (a pf changes
the output: nothing may switch), err (M11 far from Marlin: 128/256 stay Marlin; small classes keep their
base), tiny (pf wins < 3%: small classes keep their base), slowlayer (the new table slows the layer: v4's
table comes back).

Run: .venv/bin/python decode_layer/local/dryrun_v5.py [lab file] [scenario]
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v2  # noqa: E402
import dryrun_v3  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
M11_LOG = []
PF_OK = (0, 1, 2, 0x11, 0x12, 0x21, 0x22)
SCENARIO = [os.environ.get("DRYRUN_SCENARIO", "normal")]


class Ext5:
    def m11_gemm(self, w, sf, c, ids, a, out, n_pairs, start, rt, a_per_pair, out_cols, gated, tt, nt, depth, ks,
                 grid, pf):
        base.hit(f"m11 g{gated}")
        assert ids.dtype == torch.int32 and start == 0 and n_pairs == ids.numel() and grid > 0
        assert out.shape == (n_pairs, out_cols) and a.dtype == torch.bfloat16
        assert pf in PF_OK, f"pf {pf:#x} is not a lab option"
        M11_LOG.append((gated, tt, nt, depth, ks, grid, pf))
        return out.fill_(float(pf) if SCENARIO[0] == "dirty" else 0.25)

    def m12_setup(self):
        raise AssertionError("v5 must not call m12_setup")


def fake_time(self, fn, reps=15, cold=True):
    n0, before = len(M11_LOG), dict(base.CALLS)
    fn()
    calls = M11_LOG[n0:]
    marlin = any(k.startswith("marlin") and v > before.get(k, 0) for k, v in base.CALLS.items())
    gain = 0.3 if SCENARIO[0] == "tiny" else 6.0
    t = 200.0 + (40.0 if marlin else 0.0)
    for g, tt, nt, d, ks, grid, pf in calls:
        t += 20.0 - (gain if pf & 15 else 0.0) - (0.1 * gain if pf >> 4 == 1 else 0.0)
        t += 0.0 if SCENARIO[0] == "tiny" else grid / 1000  # tiny: only the < 3% pf gain is in play
    if SCENARIO[0] == "slowlayer" and IN_LAYER[0] and any(pf for *_, pf in calls):
        t += 100.0  # the new table (pf on) makes the whole layer slower
    return t


IN_LAYER = [False]
IN_PF = [False]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de7.py"
    if len(sys.argv) > 2:
        SCENARIO[0] = sys.argv[2]
    base.install_stubs()
    dryrun_v2.install_more()
    dryrun_v3.install_v3()
    dryrun_v4.install_v4()
    base.patch_torch()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **kw: (base.hit(f"nvcc {kw['name']}"), Ext5())[1]
    lab = base.load_lab(path)
    lab.ROPE_POSITIONS = 1 << 10
    lab.att_kernel_fn = dryrun_v4.att_stub
    lab.att__warm_up = lambda: base.hit("att warm")
    real_pj_warm = lab.pj__warm

    def pj_warm(st):
        dryrun_v4.WARMING[0] = True
        try:
            real_pj_warm(st)
        finally:
            dryrun_v4.WARMING[0] = False

    lab.pj__warm = pj_warm
    real_d1 = lab._v4_attention_d1

    def d1_block(data, qkv, state):
        out = real_d1(data, qkv, state)
        dryrun_v4.write_cache({**{k: data[k] for k in lab.V4_ATT_KEYS}, "qkv": qkv})
        return out * 0 + qkv[:, :1024]

    lab._v4_attention_d1 = d1_block
    real_rel = lab._v3_rel
    lab._v3_rel = lambda a, b: ((0.5 if SCENARIO[0] == "err" else 1e-3) if IN_PF[0] else real_rel(a, b))
    if not hasattr(lab, "sec_pf"):  # a clean file with an M11 table appended by make_v4.py --table
        return clean(lab)
    for name, flag in (("sec_pf", IN_PF), ("sec_layer", IN_LAYER)):
        real = getattr(lab, name)

        def wrapped(*a, _real=real, _flag=flag):
            _flag[0] = True
            try:
                return _real(*a)
            finally:
                _flag[0] = False
        setattr(lab, name, wrapped)
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(base.weights())
    text = "\n".join(lines)
    table = dict(state["m11_configs"])
    old = dict(state["m11_configs_old"])
    print("table:", {k: (None if v is None else (lab._cfg_text(v["s1"]), lab._cfg_text(v["s2"]))) for k, v in table.items()})

    # the timed path: M11 launches carry the installed table's configs (pf included); 40 tokens -> Marlin
    launched = {}
    for t, (b, q) in {1: (1, 1), 8: (1, 8), 40: (5, 8), 128: (16, 8), 256: (32, 8)}.items():
        n0 = len(M11_LOG)
        base.CALLS.clear()
        data = lab.lab_input(b, q, "sliding", seed=t, device="cpu")
        out = lab.kernel_fn(data, state)
        assert out[0].shape == (t, 8192)
        launched[t] = (M11_LOG[n0:], any(k.startswith("marlin") for k in base.CALLS))

    def follows(t):
        cfg = lab.m11_config_for(state, t)
        calls, marlin = launched[t]
        if cfg is None:
            return marlin and not calls
        want = [(1, 2, cfg["s1"]["nt"], cfg["s1"]["d"], cfg["s1"]["ks"], cfg["s1"]["c"] * state["sms"], cfg["s1"].get("pf", 0)),
                (0, cfg["s2"]["tt"], cfg["s2"]["nt"], cfg["s2"]["d"], cfg["s2"]["ks"], cfg["s2"]["c"] * state["sms"], cfg["s2"].get("pf", 0))]
        return calls == want and not marlin

    small, large = (5, 9, 17, 33), (192, 321)
    pf_of = lambda cfg: (cfg["s1"].get("pf", 0), cfg["s2"].get("pf", 0))
    strip = lambda cfg: None if cfg is None else {k: {kk: vv for kk, vv in v.items() if kk != "pf"} for k, v in cfg.items()}
    expect = {
        "normal": all(pf_of(table[b]) == (0x11, 0x11) for b in small) and all(table[b] is not None and pf_of(table[b]) == (0x11, 0x11) for b in large),
        "dirty": all(strip(table[b]) == strip(old[b]) and pf_of(table[b]) == (0, 0) for b in small) and all(table[b] is None for b in large),
        "err": all(strip(table[b]) == strip(old[b]) for b in small) and all(table[b] is None for b in large),
        "tiny": all(strip(table[b]) == strip(old[b]) and pf_of(table[b]) == (0, 0) for b in small),
        "slowlayer": all(strip(table[b]) == strip(old[b]) for b in small) and all(table[b] is None for b in large),
    }[SCENARIO[0]]
    t_lines = [ln for ln in lines if ln.startswith("T") and ": s1 " in ln]
    checks = {
        f"scenario {SCENARIO[0]}: the lab's picks": expect,
        "7 classes, 33-64 tokens on Marlin": sorted(table) == [5, 9, 17, 33, 65, 192, 321] and table[65] is None,
        "kernel_fn follows the table (1, 8, 40, 128, 256 tokens)": all(follows(t) for t in launched),
        "V5_M11_CONFIGS printed = state": ("V5_M11_CONFIGS = " + repr(state["m11_configs"])) in text,
        "6 classes measured": len(t_lines) == 6,
        "exactness reported": all((" exact 11" in ln) == (SCENARIO[0] != "dirty") for ln in t_lines),
        "no failures": "failures 0" in text and "FAILED" not in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("V5_M11_CONFIGS"),
        "log under 12K": len(text) < 12000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v5 ({SCENARIO[0]}): all checks passed")


def clean(lab):
    """The appended table is what prepare installs (its classes above 32 tokens kept when M11 matches Marlin,
    dropped in the err scenario), the [table] check ran, and kernel_fn launches the installed configs."""
    lab.lab_input = base.load_lab(DECODE / "lab_de7.py").lab_input  # the lab's judge-like input generator
    real_glue = lab._glue_prepare

    def glue(weights):
        state = real_glue(weights)
        IN_PF[0] = True  # from here on _v3_rel answers per scenario (the table's Marlin check)
        return state
    lab._glue_prepare = glue
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    IN_PF[0] = False
    file_table = dict(lab.V3_M11_CONFIGS)
    got = dict(state["m11_configs"])
    large = [b for b, cfg in file_table.items() if cfg is not None and b > 33]
    launched = {}
    for t, (b, q) in {1: (1, 1), 8: (1, 8), 40: (5, 8), 128: (16, 8), 256: (32, 8)}.items():
        n0 = len(M11_LOG)
        base.CALLS.clear()
        data = lab.lab_input(b, q, "sliding", seed=t, device="cpu")
        lab.kernel_fn(data, state)
        launched[t] = (M11_LOG[n0:], any(k.startswith("marlin") for k in base.CALLS))

    def follows(t):
        cfg = lab.m11_config_for(state, t)
        calls, marlin = launched[t]
        if cfg is None:
            return marlin and not calls
        want = [(1, 2, cfg["s1"]["nt"], cfg["s1"]["d"], cfg["s1"]["ks"], cfg["s1"]["c"] * state["sms"], cfg["s1"].get("pf", 0)),
                (0, cfg["s2"]["tt"], cfg["s2"]["nt"], cfg["s2"]["d"], cfg["s2"]["ks"], cfg["s2"]["c"] * state["sms"], cfg["s2"].get("pf", 0))]
        return calls == want and not marlin

    keep = SCENARIO[0] != "err"
    checks = {
        "table check printed": any(p.startswith("[table] M11 classes above 32 tokens vs Marlin") for p in printed),
        "classes <= 32 tokens = the file's table": all(got[b] == file_table[b] for b in file_table if b <= 33),
        f"classes above 32 tokens {'kept' if keep else 'dropped to Marlin'}": all((got[b] == file_table[b]) if keep else got[b] is None for b in large),
        "kernel_fn follows the installed table (1, 8, 40, 128, 256 tokens)": all(follows(t) for t in launched),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v5 clean ({SCENARIO[0]}): all checks passed")


if __name__ == "__main__":
    main()

"""CPU dry run of lab_l23.py's _lab_main end to end: attend_w1, attend_w1dbg and attend_ver5 are all served by the
W1 emulator (emulate_cuda_w1.attend_w1) on the current inputs, _merge by emulate.combine (the port of _combine),
_prologue_q by emulate.prologue, inputs from the judge's generate_input (batch 2, page 1); the microbenchmark
build is mocked out. Pass 1: every kernel 0 unwritten rows, partials vs FP32 < 2e-2, 0 bad rows, merged vs
reference like the FP32 partials' own merge; w1 vs w1dbg bitwise equal. Pass 2 (planted): attend_w1 leaves split 1
unwritten (NaN) and attend_ver5 adds 3 to m of split 0 -> w1 shows unwritten rows and bad split 1, v5 shows
max |dm| ~3 and a worse merged error, w1dbg stays clean.
Run: .venv/bin/python attention/local/dryrun_lab_l23.py"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
import emulate_cuda_w1 as ew  # noqa: E402
import reference as ref  # noqa: E402

sub = emulate.load_submission()


def run(plant):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l23.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    def emu(q, o, m, l_, nsplit):
        oo, mm, ll = ew.attend_w1(cur["data"], q.float(), nsplit)
        o.copy_(oo)
        m.copy_(mm)
        l_.copy_(ll)

    class W1:
        def attend_w1(self, q, kc, vc, ks, pt, seq, o, m, l_, batch, hq, hkv, mp, nsplit, page, window, sm, st, sbov, wf):
            assert (st, sbov, wf) == (4, 528, 0)
            emu(q, o, m, l_, nsplit)
            if plant:
                rows = torch.arange(o.shape[0]) % nsplit == 1
                o[rows] = float("nan")
                m[rows] = float("nan")
                l_[rows] = float("nan")

        def attend_w1_setup(self):
            return 0

    class W1Dbg:
        def attend_w1dbg(self, q, kc, vc, ks, pt, seq, o, m, l_, dbg, batch, hq, hkv, mp, nsplit, page, window, sm):
            emu(q, o, m, l_, nsplit)

    class V5:
        def attend_ver5(self, q, kc, vc, ks, pt, seq, o, m, l_, batch, hq, hkv, mp, nsplit, page, window, sm, st, minb):
            assert (st, minb) == (4, 1)
            emu(q, o, m, l_, nsplit)
            if plant:
                m[torch.arange(m.shape[0]) % nsplit == 0] += 3.0

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.cuda.synchronize = lambda: None
    lab._Lab = lambda: FakeLab()
    lab._l23_build = lambda name, src, decl, funcs: W1() if name == "okc_l23_w1" else W1Dbg()
    lab._CUDA["mod"] = V5()
    lab._lab_soft_cache = lambda gen, rows: (None, None)
    lab._kernel_fn_submitted = lambda data: None
    lab._XUHost.bench = staticmethod(lambda lab_: logs.append("xu mocked"))

    def inputs(bclass, length, window, k_cache, v_cache, gen):
        case = dict(batch=2, ctx=2600 if window else 3000, heads="tp8", layer="sliding" if window else "full",
                    q_len=length, page=1, dist="soft" if window else "coherent")
        data = ref.generate_input(seed=2300 + bclass + window, device="cpu", **case)
        cur["data"] = data
        return data

    lab._lab_inputs = inputs
    lab._prologue_q = lambda data: emulate.prologue(data, sub).reshape(-1, data["config"]["num_q_heads"], 64)
    lab._merge = lambda data, o, m, l_, ns: emulate.combine(data, o, m, l_, ns, 1 << max(0, (ns - 1).bit_length()), sub)
    lab._lab_main()
    return logs


def field(m, pat):
    x = re.search(pat, m)
    return x.group(1) if x else None


def check(logs, plant):
    bad = [m for m in logs if "FAILED" in m]
    lines = {k: [m for m in logs if re.match(rf"\w+ {k}: unwritten", m)] for k in ("w1", "w1dbg", "v5")}
    heads = [m for m in logs if "FP32 partials merged vs reference" in m]
    own = [float(field(m, r"merged vs reference ([\d.e+-]+)")) for m in heads]
    ok = not bad and len(heads) == 4 and all(len(v) == 4 for v in lines.values()) and max(own) < 0.15
    for k, ls in lines.items():
        for m in ls:
            unw = field(m, r"unwritten o/m/l rows \((\d+), ")
            perr = float(field(m, r"partials vs FP32 ([\d.e+-]+)"))
            nbad = int(field(m, r"bad rows (\d+)/"))
            dm = float(field(m, r"max \|dm\| ([\d.e+-]+)"))
            merged = float(field(m, r"merged vs ref ([\d.e+-]+)"))
            if plant and k == "w1":
                ok = ok and int(unw) > 0 and "splits [1]" in m and nbad > 0
            elif plant and k == "v5":
                ok = ok and abs(dm - 3.0) < 0.01 and nbad == 0
            else:
                ok = ok and int(unw) == 0 and perr < 2e-2 and nbad == 0 and dm < 1e-3 and merged < 0.15
    eq = [m for m in logs if "w1 vs w1dbg: bitwise equal" in m]
    ok = ok and len(eq) == 4 and all(("True" in m) != plant for m in eq)
    return ok, len(bad)


r1 = check(run(False), False)
r2 = check(run(True), True)
print("pass 1 (clean):", r1, "pass 2 (planted):", r2)
print("dryrun_lab_l23:", "ok" if r1[0] and r2[0] else "FAILED")
sys.exit(0 if r1[0] and r2[0] else 1)

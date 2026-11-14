"""CPU dry run of lab_l22.py's _lab_main end to end: the unit module is mocked (w1u writes the exact result of
each CHAIN variant), the debug module by the W1 emulator itself (attend_w1dbg runs emulate_cuda_w1.attend_w1
with DUMP on the current inputs and copies the dump and the partials out), lab inputs come from the judge's
generate_input on the CPU (batch 2, page 1). Pass 1: every unit line ~0 and every dump line clean on the
three cases. Pass 2: the mock swaps two Q.K columns of head 1 -> the "S vs dump operands" line of h1 flags
it and prints the column map. Run: .venv/bin/python attention/local/dryrun_lab_l22.py"""
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
    lab = emulate.load_submission(os.path.join(PROB, "lab_l22.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class Unit:
        def w1u(self, A, B, V, D, sbo, swap, cp, am, ch, pscale):
            dec = lambda x: x.view(torch.float8_e4m3fn).float()  # noqa: E731
            S = dec(A) @ dec(B).T
            if ch == 3:
                D.copy_((S * pscale).to(torch.float8_e4m3fn).float() @ dec(V))
            elif ch == 2:
                D.add_(S)
            else:
                D.copy_(S)

    class Dbg:
        def attend_w1dbg(self, q, kc, vc, ks, pt, seqlens, o_part, m_part, l_part, dbg, batch, hq, hkv, max_pages,
                         nsplit, page, window, sm_scale):
            data = cur["data"]
            ew.DUMP = {}
            o, m, l_ = ew.attend_w1(data, q.float(), nsplit)
            d = ew.DUMP["dbg"]
            ew.DUMP = None
            if plant:
                s1 = d[4096:8192].view(64, 64)
                s1[:, [5, 40]] = s1[:, [40, 5]].clone()
            dbg[:d.numel()] = d
            o_part.copy_(o)
            m_part.copy_(m)
            l_part.copy_(l_)

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
    lab._l22_build = lambda name, src, decl, funcs: Unit() if name == "okc_l22_unit" else Dbg()
    lab._lab_soft_cache = lambda gen, rows: (None, None)

    def inputs(bclass, length, window, k_cache, v_cache, gen):
        case = dict(batch=2, ctx=2600 if window else 3000, heads="tp8", layer="sliding" if window else "full",
                    q_len=length, page=1, dist="soft" if window else "coherent")
        data = ref.generate_input(seed=2200 + bclass + window, device="cpu", **case)
        cur["data"] = data
        return data

    lab._lab_inputs = inputs
    lab._prologue_q = lambda data: emulate.prologue(data, sub).reshape(-1, data["config"]["num_q_heads"], 64)
    lab._lab_main()
    return logs


def check(logs, plant):
    bad = [m for m in logs if "FAILED" in m]
    units = [m for m in logs if m.startswith("unit SBO")]
    unit_ok = len(units) == 4 and all(float(re.search(r"rel err ([\d.e+-]+)", m).group(1)) < 1e-5 for m in units)
    s_lines = [m for m in logs if "S vs dump operands" in m]
    errs = [float(re.search(r"S vs dump operands ([\d.e+-]+)", m).group(1)) for m in s_lines]
    regions_clean = all(re.search(r"Ktile 0/4096 differ, Vt 0/4096, Q bytes 0/4096", m) for m in s_lines)
    rows_clean = all("rows differ 0/64, V staging bytes differ 0/8192" in m for m in logs if "rows differ" in m)
    parts = [float(x) for m in logs for x in re.findall(r"vs FP32: ([\d.e+-]+)", m)]
    ok = not bad and unit_ok and len(s_lines) == 6 and regions_clean and rows_clean and max(parts) < 0.05
    if plant:
        ok = ok and errs[1] > 1e-3 and all(e < 1e-5 for i, e in enumerate(errs) if i % 2 == 0)
        ok = ok and any("h1 S col -> dump col" in m for m in logs)
    else:
        ok = ok and max(errs) < 1e-5 and not any("col -> dump col" in m for m in logs)
    return ok, dict(bad=bad, units=len(units), unit_ok=unit_ok, s=errs, parts=parts)


r1 = check(run(False), False)
r2 = check(run(True), True)
print("pass 1 (clean):", r1)
print("pass 2 (planted S column swap on h1):", r2)
print("dryrun_lab_l22:", "ok" if r1[0] and r2[0] else "FAILED")
sys.exit(0 if r1[0] and r2[0] else 1)

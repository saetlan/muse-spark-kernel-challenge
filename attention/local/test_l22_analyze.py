"""Checks lab_l22_block.py's dump analysis against the W1 emulator (an independent implementation of the
kernel's index math): emulate_cuda_w1 records the stage-0 dump of block (split 0, request 0 x pair 0) in
cuda_w1dbg.py's layout; the analysis must find every region clean on it, and each planted corruption
(cache row, K tile byte, Vt byte, Q byte, S column swap, P scale, P.V rows, partial) must show in its line.
Run: .venv/bin/python attention/local/test_l22_analyze.py"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_w1 as ew  # noqa: E402
import reference as ref  # noqa: E402
from emulate import load_submission, prologue  # noqa: E402

ns = {"torch": torch, "_log": print, "_why": lambda e: repr(e)}
exec(compile(open(os.path.join(HERE, "lab_l22_block.py")).read(), "lab_l22_block.py", "exec"), ns)
analyze, Host = ns["_l22_analyze"], ns["_L22Host"]
sub = load_submission()


def nums(lines):
    text = " ".join(lines)
    return dict(
        rows=int(re.search(r"rows differ (\d+)", text).group(1)),
        vs=int(re.search(r"V staging bytes differ (\d+)", text).group(1)),
        kt=[int(x) for x in re.findall(r"Ktile (\d+)/4096", text)],
        vt=[int(x) for x in re.findall(r"Vt (\d+)/4096", text)],
        qb=[int(x) for x in re.findall(r"Q bytes (\d+)/4096", text)],
        s_dump=[float(x) for x in re.findall(r"S vs dump operands ([\d.e+-]+)", text)],
        s_true=[float(x) for x in re.findall(r"vs true ([\d.e+-]+); P", text)],
        p=[float(x) for x in re.findall(r"; P ([\d.e+-]+);", text)],
        pv=[float(x) for x in re.findall(r"PV vs dump Vt ([\d.e+-]+)", text)],
        part=[float(x) for x in re.findall(r"vs FP32: ([\d.e+-]+)", text)])


def run(case, nsplit, seed):
    data = ref.generate_input(seed=seed, device="cpu", **case)
    q = prologue(data, sub)
    ew.DUMP = {}
    o, m, l_ = ew.attend_w1(data, q, nsplit)
    dbg = ew.DUMP["dbg"]
    ew.DUMP = None
    return data, q.float().reshape(-1, data["config"]["num_q_heads"], 64), dbg, o, m, l_


bad = 0
for case, nsplit in ((dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"), 5),
                     (dict(batch=2, ctx=2200, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), 4)):
    data, q, dbg, o, m, l_ = run(case, nsplit, 2201)
    small, pt0 = Host.compact(data)
    lines = analyze(small, q, dbg, nsplit, o, m, l_, pt_orig=pt0)
    for x in lines:
        print("  " + x)
    v = nums(lines)
    clean = (v["rows"] == 0 and v["vs"] == 0 and v["kt"] == [0, 0] and v["vt"] == [0, 0] and v["qb"] == [0, 0]
             and max(v["s_dump"] + v["s_true"]) < 1e-5 and max(v["p"]) < 1e-4 and max(v["pv"]) < 1e-5
             and max(v["part"]) < 0.05)
    print(("ok   " if clean else "FAIL ") + f"clean emulator dump on {case['layer']} (nsplit {nsplit}): {v}")
    bad += not clean

    def mutant(name, fn, check):
        global bad
        d2, o2 = dbg.clone(), o.clone()
        fn(d2, o2)
        w = nums(analyze(small, q, d2, nsplit, o2, m, l_, pt_orig=pt0))
        caught = check(w)
        print(("ok   " if caught else "FAIL ") + f"  mutant {name}: {'caught' if caught else 'MISSED'}")
        bad += not caught

    mutant("row", lambda d, o_: d.__setitem__(58630, d[58630] + 1), lambda w: w["rows"] == 1)
    mutant("ktile", lambda d, o_: d.__setitem__(32768 + 4096 + 17, (d[32768 + 4096 + 17] + 1) % 256), lambda w: w["kt"] == [0, 1])
    mutant("vt", lambda d, o_: d.__setitem__(40960 + 5, (d[40960 + 5] + 1) % 256), lambda w: w["vt"] == [1, 0])
    mutant("qbyte", lambda d, o_: d.__setitem__(24576 + 100, (d[24576 + 100] + 8) % 256), lambda w: w["qb"][0] == 1)

    def swap_cols(d, o_):
        s0 = d[0:4096].view(64, 64)
        s0[:, [3, 9]] = s0[:, [9, 3]].clone()
    mutant("S cols", swap_cols, lambda w: w["s_dump"][0] > 1e-3)
    mutant("P scale", lambda d, o_: d[8192:12288].mul_(1.5), lambda w: w["p"][0] > 0.1)

    def swap_pv(d, o_):
        p0 = d[16384 + 4096:16384 + 8192].view(64, 64)
        p0[[0, 20]] = p0[[20, 0]].clone()
    mutant("PV rows", swap_pv, lambda w: w["pv"][1] > 1e-2)
    mutant("partial", lambda d, o_: o_.mul_(-1), lambda w: max(w["part"]) > 0.5)

print("test_l22_analyze:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

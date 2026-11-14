"""CPU dry run of an L25 lab file's _lab_main: the anatomy module is mocked (an_w1 writes known slot values per CTA:
consumer slots 10/20/30/40/5 cycles per stage x stages, copier 7 / 3, transposers 11 / 13, entry 1000, exit
1000 + 2000 x stages, 8 stages); the printed per-stage figures must come back exactly; the bisection's seven
kernels (W1-M masks 0/1/2/4/8/15, W1-C) each run once per case with the right arguments; inputs small.
Run: .venv/bin/python attention/local/dryrun_lab_l25.py [lab file name in attention/, default lab_l25.py]"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "lab_l25.py"
lab = emulate.load_submission(os.path.join(PROB, name))
logs = []
lab._log = lambda m: (logs.append(m), print("[lab]", m))


class AW1:
    def an_w1(self, *a):
        assert len(a) == 21, len(a)
        buf = a[-1].view(-1, 64)
        nst = 8
        for k, v in enumerate((10, 20, 30, 40, 5)):
            buf[:, [6 * w + k for w in range(8)]] = v * nst
        buf[:, 48], buf[:, 49], buf[:, 52], buf[:, 54] = 7 * nst, 3 * nst, 11 * nst, 13 * nst
        buf[:, 56], buf[:, 57], buf[:, 59] = 1000, 1000 + 2000 * nst, nst

    def an_w1_setup(self):
        return 0

    def an_w1_attrs(self, s, v, f):
        return [140, 0, 128]


class W1M:
    def attend_w1m(self, *a):
        assert len(a) == 19 and a[-2] == 4 and a[-1] in (0, 1, 2, 4, 8, 15), (len(a), a[-2:])

    def attend_w1m_setup(self):
        return 0


class W1C:
    def attend_w1c(self, *a):
        assert len(a) == 20 and a[-3:] == (4, 528, 0), (len(a), a[-3:])

    def attend_w1c_setup(self):
        return 0

    def attend_w1c_attrs(self, s, v, f):
        return [133, 0, 128]


class Ev:
    def __init__(self, **k):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.01


real_device, real_gen = torch.device, torch.Generator
lab.torch.device = lambda s="cpu", *a: real_device("cpu")
lab.torch.Generator = lambda device=None: real_gen()
lab.torch.cuda.empty_cache = lambda: None
lab.torch.cuda.synchronize = lambda: None
lab.torch.cuda.Event = Ev
real_empty = torch.empty
lab.torch.empty = lambda *a, **k: real_empty(*((16,) if a and a[0] == (512 << 20) // 8 else a), **{x: y for x, y in k.items() if x != "device"})


class FakeLab:
    def __init__(self):
        self.t0 = lab._time.monotonic()

    def left(self):
        return 1000.0

    def time_graph(self, fn, reps=15):
        fn()
        return 50.0


lab._Lab = lambda: FakeLab()
import torch.utils.cpp_extension as ce  # noqa: E402
ce.load_inline = lambda **k: {"okc_l25_w1m": W1M(), "okc_l25_w1c": W1C(), "okc_l25_aw1": AW1()}[k["name"]]


def small_inputs(batch, length, window, k_cache, v_cache, gen):
    cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
               rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
    t = batch * length
    return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)


lab._lab_inputs = small_inputs
lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
lab._kernel_fn_submitted = lambda data: None
lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
ones = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
lab._merge = lambda data, o, m, l_, ns: ones(data)
lab._lab_reference = ones
lab._lab_main()
lines = [m for m in logs if " anat s" in m]
bis = [m for m in logs if "bisection err:" in m]
tim = [m for m in logs if "v9 path" in m]
ok = len(lines) == 4 and not [m for m in logs if "FAILED" in m] and len(bis) == 2 and len(tim) == 2
ok = ok and all(m.count(" 0.00") == 7 and "w1m0 " in m and "w1m15 " in m and "w1c " in m for m in bis)
for m in lines:
    ok = ok and "wait full 10, QK 20, softmax+pack 30, PV 40, O upd 5" in m and "wait empty 7, wait groups 3" in m \
        and "wait land 11 / 13" in m and "= 2000 cyc/stage" in m
print("dryrun_lab_l25:", "ok" if ok else "FAILED")
sys.exit(0 if ok else 1)

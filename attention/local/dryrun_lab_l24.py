"""CPU dry run of lab_l24.py (adapted from dryrun_lab_l21.py)'s _lab_main: both builds mocked (the unit test's w1u writes the exact product
for the variants that keep W1's layout reading and garbage for the others; attend_w1 checks its argument
count), inputs / reference / timing stubbed. Pass 1: no W1 winner -> 7 unit lines with errors 0 / large as
planted, the base diagnostic not printed, six KEEP lines, ROUTES = {}. Pass 2: W1 (1, 4, 512, 1) 10% faster
-> every class routed to ("w1", 1, 4, 512, 1).
Run: .venv/bin/python attention/local/dryrun_lab_l21.py"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l24.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))

    class W1:
        def attend_w1(self, *a):
            assert len(a) == 20, len(a)

        def attend_w1_setup(self):
            return 0

        def attend_w1_attrs(self, s, v, f):
            return [140, 0, 128]

    class W1N:
        def attend_w1n(self, *a):
            assert len(a) == 18, len(a)

        def attend_w1n_setup(self):
            return 0

    class W1D:
        def attend_w1dbg(self, *a):
            assert len(a) == 18 and a[9].numel() == 65536, len(a)

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()
            self.cur = None

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            self.cur = None
            fn()
            return 90.0 if fast is not None and self.cur == fast else 100.0

    fl = FakeLab()
    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.cuda.synchronize = lambda: None
    lab._Lab = lambda: fl
    lab._l24_build = lambda name, src, decl, funcs, verbose=False: {"okc_l24_w1": W1(), "okc_l24_w1n": W1N(), "okc_l24_w1d": W1D()}[name]
    real_w1 = lab._w1_run

    def w1_run(data, route):
        fl.cur = route
        return real_w1(data, route)

    lab._w1_run = w1_run
    ones = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
    lab._merge = lambda data, o, m, l_, ns: ones(data)

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16),
                    positions=torch.zeros(t, dtype=torch.long), cache_loc=torch.arange(t, dtype=torch.long),
                    cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
    lab._lab_reference = ones
    lab._kernel_fn_submitted = ones
    lab._CUDA.update(mod=object(), refused=set())
    for t in (lab.V6_ROUTES, lab.V7_ROUTES, lab.V8_ROUTES):
        lab.CUDA_ROUTE.update(t)
    lab._lab_main()
    bad = [m for m in logs if "FAILED" in m]
    timed = [m for m in logs if re.match(r"[FS]v\d+ \('w1", m)]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(timed) == 30 and len(keeps) == 6 and any(m.startswith("w1 s4 attrs") for m in logs)
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and all(lab.CUDA_ROUTE.get(c) == fast for c in lab._L24_CLASSES) and lab._cuda_ok(dict(q_len=8, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad, len(timed), len(keeps)


r1, r2 = run(None), run(("w1n", 1, 4))
print("dryrun_lab_l24:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

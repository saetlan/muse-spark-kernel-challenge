"""CPU dry run of lab_l29.py's _lab_main: the W2 module is mocked (attend_w2 must get (4, 1), attend_w2abl abl in 1..3),
_prologue_q / _merge / reference / timing stubbed, small inputs; expects 3 case lines with v10 path, w2c and three
ablation timings, no FAILED. Run: .venv/bin/python attention/local/dryrun_lab_l29.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

lab = emulate.load_submission(os.path.join(PROB, "lab_l29.py"))
logs = []
lab._log = lambda m: (logs.append(m), print("[lab]", m))
calls = []


class W2:
    def attend_w2(self, *a):
        assert len(a) == 19 and a[-2:] == (4, 1), a[-2:]
        calls.append(0)

    def attend_w2abl(self, *a):
        assert len(a) == 18 and a[-1] in (1, 2, 3), a[-1]
        calls.append(a[-1])

    def attend_w2_setup(self):
        return 0

    def attend_w2abl_setup(self):
        return 0


class FakeLab:
    def __init__(self):
        self.t0 = lab._time.monotonic()

    def left(self):
        return 1000.0

    def time_graph(self, fn, reps=15):
        fn()
        return 100.0


real_device, real_gen = torch.device, torch.Generator
lab.torch.device = lambda s="cpu", *a: real_device("cpu")
lab.torch.Generator = lambda device=None: real_gen()
lab.torch.cuda.empty_cache = lambda: None
lab._Lab = lambda: FakeLab()
import torch.utils.cpp_extension as ce  # noqa: E402
ce.load_inline = lambda **k: W2()


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
ones = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
lab._lab_reference = ones
lab._merge = lambda data, o, m, l_, ns: ones(data)
lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
lab._lab_main()
lines = [m for m in logs if m.startswith(("Fv16:", "Fv32:", "Sv32:"))]
ok = len(lines) == 3 and not [m for m in logs if "FAILED" in m] and all(m.count("us") >= 5 for m in lines)
ok = ok and sorted(set(calls)) == [0, 1, 2, 3]
print("dryrun_lab_l29:", "ok" if ok else "FAILED")
sys.exit(0 if ok else 1)

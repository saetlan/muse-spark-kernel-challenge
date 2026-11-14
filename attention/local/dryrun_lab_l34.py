"""CPU dry run of lab_l34.py's _lab_main: the UD4 module is mocked (ud4 checks its argument count, variant and output
shape; it writes ones), inputs / reference / timing stubbed. Pass 1: no winner -> 3 KEEP, ROUTES = {}. Pass 2: ('ud4', 8, 8, 2)
10% faster -> Sd16 and Sd32 routed to it; _cuda_ok accepts it for sliding decode only.
Run: .venv/bin/python attention/local/dryrun_lab_l34.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l34.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class UW2:
        def ud4(self, *a):
            assert len(a) == 23 and a[-3:] in ((4, 3, 16), (8, 2, 16), (4, 3, 8), (8, 2, 8)), (len(a), a[-3:])
            assert a[10].shape == (a[11], 1, 16 * 64)
            a[10].copy_(torch.ones_like(a[10]))
            cur["r"] = ("ud4", a[-1], a[-3], a[-2])

        def ud4_setup(self):
            return 0

        def ud4_attrs(self, w, c):
            return [150, 32, 6000]

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["r"] = None
            fn()
            return 90.0 if fast is not None and cur["r"] == fast else 100.0

    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: UW2()

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
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._lab_main()
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 3 and any(m.startswith("ud4 w4 cs16 attrs") for m in logs)
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and lab.CUDA_ROUTE.get((False, False, 16)) == fast and lab._cuda_ok(
            dict(q_len=1, window=2048, num_q_heads=16, num_kv_heads=2), fast) and not lab._cuda_ok(
            dict(q_len=8, window=2048, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad


r1, r2 = run(None), run(("ud4", 8, 8, 2))
print("dryrun_lab_l34:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

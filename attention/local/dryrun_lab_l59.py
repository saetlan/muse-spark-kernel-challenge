"""CPU dry run of lab_l59.py's _lab_main: the UD9 module (ud9: argument count, variant; ud9_attrs; ud9_setup) is
mocked; inputs / reference / timing stubbed. Pass 1: no winner -> 3 KEEP lines. Pass 2: Sd32 ("ud9", 4, 8, 3) 10%
faster -> routed. Run: .venv/bin/python attention/local/dryrun_lab_l59.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l59.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}
    variants = set(map(tuple, lab.UD9_VARIANTS))

    class Mod:
        def ud9(self, *a):
            assert len(a) == 23, len(a)
            w, s, cs = a[20], a[21], a[22]
            assert (w, s, cs) in variants, (w, s, cs)
            cur["v"] = ("ud9", cs, w, s)

        def ud9_setup(self):
            return 0

        def ud9_attrs(self, w, s, cs, gy):
            assert (w, s, cs) in variants, (w, s, cs)
            return [128, 0, 50000, 33, 2]

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["v"] = None
            fn()
            return 90.0 if fast is not None and cur.get("v") == fast else 100.0

    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: Mod()

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        assert window == 2048 and length == 1
        cfg = dict(batch=batch, q_len=1, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        return dict(qkv=torch.zeros((batch, 1, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(batch, dtype=torch.long),
                    cache_loc=torch.arange(batch, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 90000), dtype=torch.int32), config=cfg)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    real_empty = torch.empty
    lab.torch.empty = lambda *a, **k: torch.ones(*a, **{x: y for x, y in k.items() if x != "device"})
    try:
        lab._lab_main()
    finally:
        lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 3
    if fast is not None:
        ok = ok and lab.CUDA_ROUTE.get((False, False, 32)) == fast and lab._cuda_ok({"num_q_heads": 16, "num_kv_heads": 2, "q_len": 1, "window": 2048}, fast)
    return ok, bad


r1, r2 = run(None), run(("ud9", 8, 4, 3))
print("dryrun_lab_l59:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

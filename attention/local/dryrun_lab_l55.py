"""CPU dry run of lab_l55.py's _lab_main: UW2g and UD7 modules mocked (argument counts, variants, buffer sizes, tickets);
inputs / reference / timing stubbed. Pass 1: no winner -> 4 KEEP. Pass 2: Sv1 ("uw2g", 6, 0, 16, 3) faster -> routed."""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l55.py"))
    ugv = {tuple(int(x) for x in v) for v in re.findall(r"X\((\d+), (\d+), (\d+)\)", lab.CUDA_UW2G_SRC.split("#define OKC_UW2G_VARIANTS(X)")[1].split("\n")[0])}
    u7v = set(map(tuple, lab.UD7_VARIANTS))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class UG:
        def uw2g(self, *a):
            assert len(a) == 26 and tuple(a[20:23]) in ugv, (len(a), a[20:23])
            bp = a[11] * (a[13] // 2)
            assert a[24].numel() >= bp * a[23] * 128 * 68 and a[25].numel() >= bp * a[22]
            cur["v"] = ("uw2g",) + tuple(a[20:24])

        def uw2g_setup(self):
            return 0

    class U7:
        def ud7(self, *a):
            assert len(a) == 26 and (a[20], a[21], a[22]) in u7v, (len(a), a[20:23])
            bp = a[11] * (a[13] // 2)
            assert a[24].numel() >= bp * a[23] * 16 * 68 and a[25].numel() >= bp * a[22]
            cur["v"] = ("ud7", a[22], a[20], a[21], a[23])

        def ud7_setup(self):
            return 0

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
    ce.load_inline = lambda **k: UG() if "uw2g" in k["name"] else U7()

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                    cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 90000), dtype=torch.int32), config=cfg)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    real_empty = torch.empty
    lab.torch.empty = lambda *a, **k: torch.ones(*a, **{x: y for x, y in k.items() if x != "device"})
    try:
        lab._lab_main()
    finally:
        lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 4
    if fast is not None:
        ok = ok and lab.CUDA_ROUTE.get((False, True, 1)) == fast
    return ok, bad


r1, r2 = run(None), run(("uw2g", 6, 0, 16, 3))
print("dryrun_lab_l55:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

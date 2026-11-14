"""CPU dry run of lab_l32.py's _lab_main: one mocked module for the whole-row kernels (attend_w3b / uw2b / attend_w1nb
check argument counts, variants and partial-buffer sizes), _prologue_q / _merge / inputs / reference / timing stubbed.
Pass 1: no winner -> 6 KEEP, ROUTES = {}. Pass 2: ('uw2b', 4, 1, 4) and ('w3b', 6, 1, 4, 104) 10% faster -> Sv16 and
the classes timing that W3b variant are routed and _cuda_ok accepts them.
Run: .venv/bin/python attention/local/dryrun_lab_l32.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l32.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}
    w3v = {tuple(v[:4]) for v in lab.W3B_VARIANTS}

    class Mod:
        def attend_w3b(self, *a):
            assert len(a) == 21 and tuple(a[17:21]) in w3v, (len(a), a[17:])
            rows = a[9] * 8 * a[10] * a[13] * (a[19] // 2)
            assert a[7].numel() == rows and a[6].numel() == rows * 64
            cur["r"], cur["parts"] = ("w3b",) + tuple(a[17:21]), a[13] * (a[19] // 2)

        def attend_w3b_setup(self):
            return 0

        def attend_w3b_attrs(self, *a):
            return [96, 0, 256, 1]

        def uw2b(self, *a):
            assert len(a) == 23 and a[-3:] in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)), (len(a), a[-3:])
            a[10].copy_(torch.ones_like(a[10]))
            cur["r"] = ("uw2b",) + tuple(a[-3:])

        def uw2b_setup(self):
            return 0

        def uw2b_attrs(self, *a):
            return [134, 32, 0]

        def attend_w1nb(self, *a):
            assert len(a) == 18 and a[17] in (3, 4, 6), (len(a), a[17])
            assert a[7].numel() == a[9] * 8 * a[10] * a[13]
            cur["r"], cur["parts"] = ("w1nb", 1, a[17]), a[13]

        def attend_w1nb_setup(self):
            return 0

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["r"] = None
            fn()
            return 90.0 if fast and cur["r"] in fast else 100.0

    real_device, real_gen, real_empty = torch.device, torch.Generator, torch.empty
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.empty = lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"})
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: Mod()

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                    cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)

    def merge(data, o, m, l_, ns):
        assert ns == cur["parts"], (ns, cur["parts"])
        return torch.ones((data["config"]["batch"], 8, 16 * 64))

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64), dtype=torch.bfloat16)
    lab._merge = merge
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    lab._lab_main()
    lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 6 and any(m.startswith("setup (w3b") for m in logs)
    if not fast:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and lab.CUDA_ROUTE.get((False, True, 16)) == ("uw2b", 4, 1, 4)
        ok = ok and all(lab.CUDA_ROUTE.get(c) == ("w3b", 6, 1, 4, 104) for c in ((True, True, 16), (True, True, 32)))
        ok = ok and lab._cuda_ok(dict(q_len=8, window=2048, num_q_heads=16, num_kv_heads=2), ("uw2b", 4, 1, 4))
        ok = ok and not lab._cuda_ok(dict(q_len=8, window=0, num_q_heads=16, num_kv_heads=2), ("uw2b", 4, 1, 4))
    return ok, bad


r1, r2 = run(None), run({("uw2b", 4, 1, 4), ("w3b", 6, 1, 4, 104)})
print("dryrun_lab_l32:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

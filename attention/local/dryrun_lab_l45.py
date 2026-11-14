"""CPU dry run of lab_l45.py's _lab_main (W4 module mocked: argument count, variant, nsplit multiple of npair,
partial sizes; nsplit merge). Run: .venv/bin/python attention/local/dryrun_lab_l45.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l45.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}
    variants = [tuple(v) for v in lab.W4_VARIANTS]

    class W3:
        def attend_w4(self, *a):
            assert len(a) == 20, len(a)
            v = tuple(a[17:20])
            assert v in variants, v
            batch, hq, nsplit, npair = a[9], a[10], a[13], a[18]
            assert nsplit % npair == 0
            rows = batch * 8 * hq * nsplit
            assert a[7].numel() == rows and a[8].numel() == rows and a[6].numel() == rows * 64, (a[7].numel(), rows)
            cur["r"] = ("w4",) + v
            cur["parts"] = nsplit

        def attend_w4_setup(self):
            return 0

        def attend_w4_attrs(self, s, p, m):
            return [120, 0, 0]

    class Probe:
        def probe(self, *a):
            assert len(a) == 11 and (a[8], a[9]) in ((16, 128), (16, 256), (8, 256), (4, 256)) and a[10] in (0, 1, 2, 3, 4), a[5:]
            cur["r"] = ("probe",) + tuple(a[8:])

        def probe_empty(self, out, c):
            cur["r"] = ("empty", c)

        def probe_setup(self):
            return 0

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
    real_empty = torch.empty
    lab.torch.empty = lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"})
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: W3()

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
    lab._kernel_fn_submitted = lambda data: None
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64), dtype=torch.bfloat16)
    lab._merge = merge
    lab._lab_main()
    lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 4 and sum(m.startswith("w4 s") for m in logs) == len(variants)
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and all(lab.CUDA_ROUTE.get(c) == fast for c in ((True, True, 16), (True, True, 32), (True, True, 1), (False, True, 32)))
        ok = ok and lab._cuda_ok(dict(q_len=8, window=0, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad


r1, r2 = run(None), run(("w4", 3, 2, 1))
print("dryrun_lab_l45:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

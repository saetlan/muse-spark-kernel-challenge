"""CPU dry run of lab_l41.py's _lab_main (W3t module mocked: argument count, variant, partial sizes, hang getter).
Run: .venv/bin/python attention/local/dryrun_lab_l41.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l41.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}
    variants = [tuple(v[:4]) for v in lab.W3T_VARIANTS]

    class W3:
        def attend_w3t(self, *a):
            assert len(a) == 21, len(a)
            v = tuple(a[17:21])
            assert v in variants, v
            batch, hq, nsplit, nwg = a[9], a[10], a[13], a[19]
            rows = batch * 8 * hq * nsplit * (nwg // 2)
            assert a[7].numel() == rows and a[8].numel() == rows and a[6].numel() == rows * 64, (a[7].numel(), rows)
            cur["r"] = ("w3t",) + v
            cur["parts"] = nsplit * (nwg // 2)

        def attend_w3t_hang(self):
            return 0

        def attend_w3t_setup(self):
            return 0

        def attend_w3t_attrs(self, s, p, nw, rc):
            return [96 if rc else 168, 0, 0, 1]

    class Probe:
        def probe(self, *a):
            assert len(a) == 11 and (a[8], a[9]) in ((16, 128), (16, 256), (8, 256), (4, 256)) and a[10] in (0, 1, 2, 3, 4), a[5:]
            cur["r"] = ("probe",) + tuple(a[8:])

        def probe_empty(self, out, c):
            cur["r"] = ("empty", c)

        def probe_setup(self):
            return 0

    class Anat:
        def an_uw2(self, *a):
            assert len(a) == 24, len(a)
            an, cs, batch = a[23], a[22], a[11]
            ctas = cs * batch * (a[13] // 2)
            assert an.numel() == ctas * 32, (an.numel(), ctas)
            v = an.view(ctas, 32)
            for k in range(32):
                v[:, k] = 1000 * k + 7
            v[:, 0], v[:, 1], v[:, 28], v[:, 29] = 0, 1000, 29000, 28000
            cur["r"] = ("anat",)

        def an_uw2_setup(self):
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
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    lab._lab_main()
    lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 2 and sum(m.startswith("w3t s") for m in logs) == len(variants)
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and all(lab.CUDA_ROUTE.get(c) == fast for c in ((True, True, 16), (True, True, 32)))
        ok = ok and lab._cuda_ok(dict(q_len=8, window=0, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad


r1, r2 = run(None), run(("w3t", 5, 0, 4, 104))
print("dryrun_lab_l41:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

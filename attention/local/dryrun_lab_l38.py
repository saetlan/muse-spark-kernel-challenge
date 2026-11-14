"""CPU dry run of lab_l38.py's _lab_main: the D3p module and the Triton PDL kernels are mocked (attend_dec4p checks its
argument count, variant and partial sizes; _prologue_p / _combine_p check their grids and write the output), inputs /
reference / timing stubbed. Pass 1: no winner -> 3 KEEP, ROUTES = {}, 4 tail lines. Pass 2: ('dec4p', 2, 4, 3, 1) 10%
faster -> Fd1 routed, _cuda_ok accepts it for full decode only.
Run: .venv/bin/python attention/local/dryrun_lab_l38.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l38.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class Mod:
        def attend_dec4p(self, *a):
            assert len(a) == 20 and (a[17], a[18]) in ((8, 2), (8, 3), (4, 4), (4, 3)) and a[19] in (0, 1), (len(a), a[17:])
            assert a[7].numel() == a[9] * a[10] * a[13]
            cur["r"] = ("dec4p", None, a[17], a[18], a[19])

        def attend_dec4p_setup(self):
            return 0

    class Kern:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def call(*a, **k):
                if self.name == "combine":
                    assert "launch_pdl" in k and k["CB"] >= 1
                    a[5].copy_(torch.ones_like(a[5]))
                else:
                    assert "TRIGGER_EARLY" in k
            return call

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["r"] = None
            fn()
            r = cur["r"]
            return 90.0 if fast and r and (r[2], r[3], r[4]) == fast else 100.0

    real_device, real_gen, real_empty = torch.device, torch.Generator, torch.empty
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.empty = lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"})
    real_rand = torch.rand
    lab.torch.rand = lambda *a, **k: real_rand(a[0], generator=k.get("generator"))
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: Mod()
    lab._prologue_p = Kern("prologue")
    lab._combine_p = Kern("combine")
    lab._inv_freq = lambda theta, dev: torch.ones(32)

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                    cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 90000), dtype=torch.int32), config=cfg)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._lab_main()
    lab.torch.empty = real_empty
    lab.torch.rand = real_rand
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    tails = [m for m in logs if m.startswith("tail F")]
    ok = not bad and len(keeps) == 3 and len(tails) == 4
    if not fast:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and lab.CUDA_ROUTE.get((True, False, 1)) == ("dec4p", 2, 4, 3, 1)
        ok = ok and lab._cuda_ok(dict(q_len=1, window=0, num_q_heads=16, num_kv_heads=2), ("dec4p", 2, 4, 3, 1))
        ok = ok and not lab._cuda_ok(dict(q_len=1, window=2048, num_q_heads=16, num_kv_heads=2), ("dec4p", 2, 4, 3, 1))
    return ok, bad


r1, r2 = run(None), run((4, 3, 1))
print("dryrun_lab_l38:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

"""CPU dry run of lab_l39.py's _lab_main: v14's module is mocked (attend_w3b checks argument count, variant and partial
sizes for the multi-wave split; attend_dec4 for the Fd16 routes), _prologue_q / _merge / inputs / reference / timing
stubbed. Pass 1: no winner -> 4 KEEP, 4 tail lines. Pass 2: ('w3bm', 3, 6, 1, 4, 104) 10% faster -> Fv16 / Fv32 / ...
routed, _cuda_ok accepts it for verify.
Run: .venv/bin/python attention/local/dryrun_lab_l39.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l39.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class Mod:
        def attend_w3b(self, *a):
            assert len(a) == 21 and tuple(a[17:21]) in ((6, 1, 4, 104), (4, 1, 4, 104)), a[17:]
            rows = a[9] * 8 * a[10] * a[13] * 2
            assert a[7].numel() == rows and a[6].numel() == rows * 64
            cur["parts"] = a[13] * 2
            cur["ns"] = a[13]

        def attend_dec4(self, *a):
            assert len(a) == 19
            cur["parts"] = a[13]
            cur["ns"] = a[13]

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["route"] = None
            fn()
            return 90.0 if fast is not None and cur.get("route") == fast else 100.0

    real_device, real_gen, real_empty, real_rand = torch.device, torch.Generator, torch.empty, torch.rand
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.empty = lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"})
    lab.torch.rand = lambda *a, **k: real_rand(a[0], generator=k.get("generator"))
    lab._Lab = lambda: FakeLab()
    lab._CUDA["mod"] = Mod()
    real_run = lab._cuda_run

    def tracked(data, route):
        out = real_run(data, route)
        cur["route"] = route
        return out
    lab._cuda_run = tracked

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                    cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 90000), dtype=torch.int32), config=cfg)

    def merge(data, o, m, l_, ns):
        assert ns == cur["parts"], (ns, cur["parts"])
        return torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * data["config"]["q_len"], 16, 64), dtype=torch.bfloat16)
    lab._merge = merge
    lab._lab_main()
    lab.torch.empty, lab.torch.rand = real_empty, real_rand
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    tails = [m for m in logs if m.startswith("tail F")]
    ok = not bad and len(keeps) == 4 and len(tails) == 4
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and lab.CUDA_ROUTE.get((True, True, 16)) == fast and lab.CUDA_ROUTE.get((True, True, 32)) == fast
        ok = ok and lab._cuda_ok(dict(q_len=8, window=0, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad


r1, r2 = run(None), run(("w3bm", 3, 6, 1, 4, 104))
print("dryrun_lab_l39:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

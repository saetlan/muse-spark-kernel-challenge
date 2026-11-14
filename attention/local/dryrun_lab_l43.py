"""CPU dry run of lab_l43.py's _lab_main: the D3t module is mocked (attend_dec4t checks argument count, (warps, stages)
and partial sizes; attrs; hang getter), _prologue_q / _merge / inputs / reference / timing stubbed. Pass 1: no winner ->
3 KEEP. Pass 2: ('dec4t', 3, 4, 3, 'w') 10% faster -> Fd16 routed, _cuda_ok accepts it for decode only.
Run: .venv/bin/python attention/local/dryrun_lab_l43.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l43.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class Mod:
        def attend_dec4t(self, *a):
            assert len(a) == 19 and (a[17], a[18]) in ((8, 2), (8, 3), (4, 4), (4, 3)), (len(a), a[17:])
            assert a[7].numel() == a[9] * a[10] * a[13] and a[6].numel() == a[7].numel() * 64
            cur["parts"] = a[13]

        def attend_dec4t_setup(self):
            return 0

        def attend_dec4t_attrs(self, *a):
            return [144, 0, 256]

        def attend_dec4t_hang(self):
            return 0

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            cur["route"] = None
            fn()
            return 90.0 if fast is not None and cur.get("route") == fast else 100.0

    real_device, real_gen, real_empty = torch.device, torch.Generator, torch.empty
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab.torch.empty = lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"})
    lab._Lab = lambda: FakeLab()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: Mod()
    real = lab._d3t_run

    def tracked(data, route):
        out = real(data, route)
        cur["route"] = route
        return out
    lab._d3t_run = tracked

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                    cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 90000), dtype=torch.int32), config=cfg)

    def merge(data, o, m, l_, ns):
        assert ns == cur["parts"]
        return torch.ones((data["config"]["batch"], 1, 16 * 64))

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"], 16, 64), dtype=torch.bfloat16)
    lab._merge = merge
    lab._lab_main()
    lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m or "HANG" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    ok = not bad and len(keeps) == 3
    if fast is None:
        ok = ok and any(m.startswith("ROUTES = {}") for m in logs)
    else:
        ok = ok and lab.CUDA_ROUTE.get((True, False, 16)) == fast
        ok = ok and lab._cuda_ok(dict(q_len=1, window=0, num_q_heads=16, num_kv_heads=2), fast)
        ok = ok and not lab._cuda_ok(dict(q_len=8, window=0, num_q_heads=16, num_kv_heads=2), fast)
    return ok, bad


r1, r2 = run(None), run(("dec4t", 3, 4, 3, "w"))
print("dryrun_lab_l43:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

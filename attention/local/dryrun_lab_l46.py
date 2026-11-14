"""CPU dry run of lab_l46.py's _lab_main: v14's module (attend_dec4: argument count, variant, partial sizes) and the UD4
anatomy module (an_ud4: argument count, stamp-buffer size) are mocked; _prologue_q / _merge / inputs / reference / timing
stubbed. Pass 1: no winner -> 3 KEEP + 3 anatomy lines. Pass 2: ('dec4', 3, 4, 4, 'w') 10% faster -> Fd16 routed.
Run: .venv/bin/python attention/local/dryrun_lab_l46.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l46.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))
    cur = {}

    class Mod:
        def attend_dec4(self, *a):
            assert len(a) == 19 and (a[17], a[18]) in ((8, 2), (8, 3), (4, 4), (4, 3)), (len(a), a[17:])
            assert a[7].numel() == a[9] * a[10] * a[13]
            cur["parts"] = a[13]

    class Anat:
        def an_ud4(self, *a):
            assert len(a) == 24
            an, cs, batch = a[23], a[22], a[11]
            ctas = cs * batch * (a[13] // 2)
            assert an.numel() == ctas * 32
            v = an.view(ctas, 32)
            for k in range(32):
                v[:, k] = 1000 * k + 7
            v[:, 0], v[:, 1], v[:, 28], v[:, 29] = 0, 1000, 29000, 28000

        def an_ud4_setup(self):
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
    lab._CUDA["mod"] = Mod()
    import torch.utils.cpp_extension as ce
    ce.load_inline = lambda **k: Anat()
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
        assert ns == cur["parts"]
        return torch.ones((data["config"]["batch"], 1, 16 * 64))

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64), dtype=torch.uint8), torch.zeros((300, 1, 2, 64), dtype=torch.uint8))
    lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 1, 16 * 64))
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"], 16, 64), dtype=torch.bfloat16)
    lab._merge = merge
    lab._inv_freq = lambda theta, dev: torch.ones(32)
    lab._lab_main()
    lab.torch.empty = real_empty
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    anat = [m for m in logs if m.startswith("anat UD4 bs")]
    ok = not bad and len(keeps) == 3 and len(anat) == 3
    if fast is not None:
        ok = ok and lab.CUDA_ROUTE.get((True, False, 16)) == fast
    return ok, bad


r1, r2 = run(None), run(("dec4", 3, 4, 4, "w"))
print("dryrun_lab_l46:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

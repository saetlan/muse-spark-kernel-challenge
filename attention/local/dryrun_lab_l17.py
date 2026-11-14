"""CPU dry run of lab_l17.py's _lab_main: mocked module (uv4 / ud3 check their argument counts and write
ones: plumbing only; a second pass makes one UV4 variant 10% faster), stubbed inputs / reference /
timing. Every planned class (Fv16, Fv32, Sv1/16/32, Sd16/32, Fv1) must reach its ROUTE/KEEP line,
the probe and the final ROUTES line; with no winner each class must keep its measured route (L16)
or v5's path (UV4 / UD3 guesses removed); with a winner each class whose candidate list holds it
must be routed to it, and the others must keep their routes.
Run: .venv/bin/python attention/local/dryrun_lab_l17.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

PLANNED = [(True, True, 16), (True, True, 32), (False, True, 1), (False, True, 16), (False, True, 32),
           (False, False, 16), (False, False, 32), (True, True, 1)]


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l17.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))

    class Mod:
        def uv4(self, *a):
            assert len(a) == 26, len(a)
            a[10].fill_(1.0)

        def uv4_attrs(self, s, mt, cs):
            return [128, 0, 8000]

        def ud3(self, *a):
            assert len(a) == 23, len(a)
            a[10].fill_(1.0)

        def ud3_attrs(self, w, cs):
            return [100, 0, 300]

        def attend_ver4(self, *a):
            assert len(a) == 20, len(a)

        def attend_ver4_attrs(self, *a):
            return [140, 0, 64]

        def attend_dec4(self, *a):
            assert len(a) == 19, len(a)

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()
            self.cur = None

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            fn()
            return 90.0 if fast is not None and fast == self.cur else 100.0

    fl = FakeLab()
    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab._Lab = lambda: fl
    lab._CUDA.update(mod=Mod(), refused=set())
    lab._TICKETS.append(torch.zeros(4096, dtype=torch.int32))
    lab.CUDA_ROUTE.update(lab.V6_ROUTES)
    lab.CUDA_ROUTE.update(lab.V7_ROUTES)
    real_run = lab._cuda_run

    def cuda_run(data, route):
        fl.cur = route
        return real_run(data, route)

    lab._cuda_run = cuda_run

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16),
                    positions=torch.zeros(t, dtype=torch.long), cache_loc=torch.arange(t, dtype=torch.long),
                    cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)

    def submitted(data):
        fl.cur = None
        cfg = data["config"]
        return torch.ones((cfg["batch"], cfg["q_len"], 16 * 64))

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
    lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
    lab._kernel_fn_submitted = submitted
    lab.torch.cuda.empty_cache = lambda: None
    measured = {c: r for c, r in lab.V6_ROUTES.items() if not (r[0] == "ud3" and c[2] > 1)}
    lab._lab_main()
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    want = {c: measured.get(c) for c in PLANNED}
    if fast is not None:
        cands = {**{c: lab._L17_FV[c] for c in lab._L17_FV}, (False, True, 1): lab._L17_SV, (False, True, 16): lab._L17_SV,
                 (False, True, 32): lab._L17_SV}
        want.update({c: fast for c, cl in cands.items() if fast[1:] in cl})
    ok = (not bad and len(keeps) == len(PLANNED) and all(lab.CUDA_ROUTE.get(c) == want[c] for c in PLANNED)
          and any(m.startswith("probe Fv16 L2-hot") for m in logs) and any(m.startswith("ROUTES = ") for m in logs)
          and all(lab.CUDA_ROUTE.get(c) == r for c, r in measured.items() if c not in PLANNED))
    return ok, bad, len(keeps), {c: lab.CUDA_ROUTE.get(c) for c in PLANNED}


ok1, *info1 = run(None)
ok2, *info2 = run(("uv4", 8, 3, 1))
print("dryrun_lab_l17:", "ok" if ok1 and ok2 else f"FAILED no-winner {ok1} {info1} / winner {ok2} {info2}")
sys.exit(0 if ok1 and ok2 else 1)

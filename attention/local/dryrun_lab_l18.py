"""CPU dry run of lab_l18.py's _lab_main: mocked module (attend_ver5 checks its argument count; plumbing only),
run_fa3t mocked (FA3 / Triton do not run on the CPU), stubbed inputs / reference / timing. Every planned
class (Fv16, Fv32, Sv16, Sv32, Fv1, Sv1) must reach its ROUTE/KEEP line, the probe and the final ROUTES
line; with no winner the V5 guesses must be gone and the measured routes kept; with a V5 winner its
classes must be routed to it; with an fa3t winner on Fv16 it must sit in FA3T_ROUTE with no CUDA route.
Run: .venv/bin/python attention/local/dryrun_lab_l18.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

PLANNED = [(True, True, 16), (True, True, 32), (False, True, 16), (False, True, 32), (True, True, 1), (False, True, 1)]


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l18.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))

    class Mod:
        def attend_ver5(self, *a):
            assert len(a) == 19, len(a)

        def attend_ver5_attrs(self, s, mb):
            return [110 if mb == 2 else 140, 0, 64]

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
    lab.CUDA_ROUTE.update(lab.V6_ROUTES)
    lab.CUDA_ROUTE.update(lab.V7_ROUTES)
    lab.CUDA_ROUTE.update(lab.V8_ROUTES)
    measured = {c: r for c, r in lab.CUDA_ROUTE.items() if r[0] != "ver5"}
    real_l18 = lab._l18_run

    def l18_run(data, route):
        fl.cur = route
        return real_l18(data, route)

    lab._l18_run = l18_run
    cfg_out = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
    lab.run_fa3t = lambda data, ns, pg: cfg_out(data)
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
    lab._merge = lambda data, o, m, l_, ns: cfg_out(data)
    real_cuda_run = lab._cuda_run

    def cuda_run(data, route):
        if route[0] == "ver5":
            fl.cur = route if fl.cur is None or fl.cur[0] != "ver5" else fl.cur
        return real_cuda_run(data, route)

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
        return cfg_out(data)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
    lab._lab_reference = cfg_out
    lab._kernel_fn_submitted = submitted
    lab.torch.cuda.empty_cache = lambda: None
    lab._lab_main()
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    want = {c: measured.get(c) for c in PLANNED}
    want_fa3 = {}
    if fast is not None and fast[0] == "ver5":
        want.update({c: fast for c in PLANNED if fast[1] in lab._L18_CPS[c]})
    if fast is not None and fast[0] == "fa3t":
        for c in ((True, True, 16), (True, True, 32)):
            want[c], want_fa3[c] = None, fast[1:]
    ok = (not bad and len(keeps) == len(PLANNED) and all(lab.CUDA_ROUTE.get(c) == want[c] for c in PLANNED)
          and lab.FA3T_ROUTE == want_fa3 and any(m.startswith("ROUTES = ") for m in logs)
          and any(m.startswith("probe Fv16 L2-hot") for m in logs)
          and not any(r[0] == "ver5" and r != fast for r in lab.CUDA_ROUTE.values()))
    return ok, bad, len(keeps), {c: lab.CUDA_ROUTE.get(c) for c in PLANNED}, dict(lab.FA3T_ROUTE)


res = [run(None), run(("ver5", 2, 3, 2)), run(("fa3t", 8, True))]
ok = all(r[0] for r in res)
print("dryrun_lab_l18:", "ok" if ok else f"FAILED {[r for r in res if not r[0]]}")
sys.exit(0 if ok else 1)

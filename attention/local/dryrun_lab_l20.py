"""CPU dry run of lab_l20.py's _lab_main: the W1 build is mocked (attend_w1 checks its argument count), the
Triton PDL chain is mocked (run_pdl(data, tune) returns ones), inputs / reference / timing stubbed. Pass 1:
no winner -> six KEEP lines, the probe, ROUTES = {}, both tune classes swept and kept, TUNE unchanged.
Pass 2: W1 c1 s4 10% faster -> every class whose list holds it is routed to ("w1", 1, 4) (the lab's
_cuda_ok / _cuda_run accept it), the tune sweep skips the classes W1 took.
Run: .venv/bin/python attention/local/dryrun_lab_l20.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402


def run(fast):
    lab = emulate.load_submission(os.path.join(PROB, "lab_l20.py"))
    logs = []
    lab._log = lambda m: (logs.append(m), print("[lab]", m))

    class W1:
        def attend_w1(self, *a):
            assert len(a) == 18, len(a)

        def attend_w1_setup(self):
            return 0

        def attend_w1_attrs(self, s):
            return [168, 0, 64]

    class FakeLab:
        def __init__(self):
            self.t0 = lab._time.monotonic()
            self.cur = None

        def left(self):
            return 1000.0

        def time_graph(self, fn, reps=15):
            self.cur = None
            fn()
            return 90.0 if fast is not None and self.cur == fast else 100.0

    fl = FakeLab()
    real_device, real_gen = torch.device, torch.Generator
    lab.torch.device = lambda s="cpu", *a: real_device("cpu")
    lab.torch.Generator = lambda device=None: real_gen()
    lab.torch.cuda.empty_cache = lambda: None
    lab._Lab = lambda: fl
    lab._l20_build = lambda: lab._W1.update(mod=W1())
    real_w1 = lab._w1_run

    def w1_run(data, cps, s):
        fl.cur = ("w1", cps, s)
        return real_w1(data, cps, s)

    lab._w1_run = w1_run
    ones = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
    lab.run_pdl = lambda data, tune=None, early=False: ones(data)
    lab.sglang_path = lambda data: ones(data)
    lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
    lab._merge = lambda data, o, m, l_, ns: ones(data)

    def small_inputs(batch, length, window, k_cache, v_cache, gen):
        cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
        t = batch * length
        return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16),
                    positions=torch.zeros(t, dtype=torch.long), cache_loc=torch.arange(t, dtype=torch.long),
                    cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                    k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                    page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)

    lab._lab_inputs = small_inputs
    lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
    lab._lab_reference = ones
    lab._kernel_fn_submitted = ones
    lab._CUDA.update(mod=object(), refused=set())
    for t in (lab.V6_ROUTES, lab.V7_ROUTES, lab.V8_ROUTES):
        lab.CUDA_ROUTE.update(t)
    tune0 = dict(lab.TUNE)
    lab._lab_main()
    bad = [m for m in logs if "FAILED" in m]
    keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
    swept = [m for m in logs if m.startswith("tune Sv") and "TUNE (" in m]
    if fast is None:
        ok = (not bad and len(keeps) == 6 and any(m.startswith("ROUTES = {}") for m in logs) and len(swept) == 2
              and any(m.startswith("SGLANG_FALLBACK = []") for m in logs) and lab.PDL_ROUTE >= {(False, True, 32), (False, True, 16)}
              and any(m.startswith("probe Fv16 L2-hot") for m in logs) and lab.TUNE == tune0
              and not any(r[0] == "w1" for r in lab.CUDA_ROUTE.values()))
    else:
        routed = {c for c, cfgs in lab._L20_W1.items() if fast[1:] in cfgs}
        ok = (not bad and len(keeps) == 6 and all(lab.CUDA_ROUTE.get(c) == fast for c in routed)
              and lab._cuda_ok(dict(q_len=8, num_q_heads=16, num_kv_heads=2), fast)
              and any("is routed elsewhere" in m for m in logs))
    return ok, bad, len(keeps), len(swept)


r1, r2 = run(None), run(("w1", 1, 4))
print("dryrun_lab_l20:", "ok" if r1[0] and r2[0] else f"FAILED {r1} {r2}")
sys.exit(0 if r1[0] and r2[0] else 1)

"""CPU dry run of lab_l19.py's _lab_main: the anatomy build and an_ver5 are mocked (an_ver5 checks its
argument count and fills the stamp buffer with known per-CTA values), inputs / reference / timing
stubbed. Every case and config must print one report line (Fv16 also an L2-hot line) with the
expected per-stage means, and nothing may fail.
Run: .venv/bin/python attention/local/dryrun_lab_l19.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

lab = emulate.load_submission(os.path.join(PROB, "lab_l19.py"))
logs = []
lab._log = lambda m: (logs.append(m), print("[lab]", m))


class An:
    def an_ver5(self, *a):
        assert len(a) == 20, len(a)
        buf, nsplit, batch = a[19], a[13], a[9]
        v = buf[:nsplit * batch * 64].view(-1, 64)
        for w in range(8):
            v[:, 6 * w + 0], v[:, 6 * w + 1], v[:, 6 * w + 2], v[:, 6 * w + 3] = 100, 200, 300, 400
            v[:, 6 * w + 4], v[:, 6 * w + 5] = 10, 10000
        v[:, 48], v[:, 49], v[:, 50], v[:, 56], v[:, 57], v[:, 58] = 500, 100, 10000, 0, 12000, 3

    def an_ver5_setup(self):
        return 0

    def an_ver5_attrs(self, s, m):
        return [100, 0, 64]


class FakeLab:
    def __init__(self):
        self.t0 = lab._time.monotonic()

    def left(self):
        return 1000.0

    def time_graph(self, fn, reps=15):
        fn()
        return 100.0


real_device, real_gen = torch.device, torch.Generator
lab.torch.device = lambda s="cpu", *a: real_device("cpu")
lab.torch.Generator = lambda device=None: real_gen()
lab.torch.cuda.synchronize = lambda: None
lab.torch.cuda.empty_cache = lambda: None
lab._Lab = FakeLab
lab._l19_build = lambda: lab._AN19.update(mod=An())


class Mod:
    def attend_ver5(self, *a):
        assert len(a) == 19, len(a)

    def attend_ver5t(self, *a):
        assert len(a) == 19, len(a)

    def attend_ver5t_attrs(self, s, mb):
        return [100, 0, 64]


lab._CUDA.update(mod=Mod(), refused=set())
for t in (lab.V6_ROUTES, lab.V7_ROUTES, lab.V8_ROUTES):
    lab.CUDA_ROUTE.update(t)


def small_inputs(batch, length, window, k_cache, v_cache, gen):
    cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
               rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
    t = batch * length
    return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                cache_loc=torch.arange(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)


ones = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))  # noqa: E731
lab._lab_inputs = small_inputs
lab._lab_soft_cache = lambda gen, rows: (torch.zeros((300, 1, 2, 64)), torch.zeros((300, 1, 2, 64)))
lab._lab_reference = ones
lab._kernel_fn_submitted = ones
lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
lab._merge = lambda data, o, m, l_, ns: ones(data)
lab._lab_main()
reports = [m for m in logs if " an5 c" in m]
keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE ")]
ok = (not any("FAILED" in m for m in logs) and len(reports) == 8 and len(keeps) == 6
      and any(m.startswith("ROUTES = {}") for m in logs) and not any(r[0] == "ver5t" for r in lab.CUDA_ROUTE.values())
      and all("wait 10 qk 20 soft+V 30 pv 40 = 100 cyc" in m and "on one SM over the run" in m and "empty-wait 5% page-wait 1%" in m for m in reports))
print("dryrun_lab_l19:", "ok" if ok else f"FAILED ({len(reports)} reports, {len(keeps)} route lines)")
sys.exit(0 if ok else 1)

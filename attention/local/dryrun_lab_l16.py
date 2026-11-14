"""CPU dry run of lab_l16.py's _lab_main: mocked module (attend_ver4 writes zeros: plumbing only),
stubbed inputs / reference / timing. Every class must reach its ROUTE/KEEP line and the final
V6_ROUTES line; the v5 routes must be what the lab leaves when no V4 variant wins.
Run: .venv/bin/python attention/local/dryrun_lab_l16.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

lab = emulate.load_submission(os.path.join(PROB, "lab_l16.py"))
logs = []
lab._log = lambda m: (logs.append(m), print("[lab]", m))


class Mod:
    def attend_ver4(self, *a):
        assert len(a) == 20, len(a)

    def attend_dec4(self, *a):
        assert len(a) == 19, len(a)

    def ud3(self, *a):
        assert len(a) == 23, len(a)
        a[10].zero_()

    def attend_ver4_attrs(self, s, mt, mb):
        return [150, 0, 64]


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
lab._Lab = FakeLab
lab._CUDA.update(mod=Mod(), refused=set())
lab.CUDA_ROUTE.update(lab.V6_ROUTES)


def small_inputs(batch, length, window, k_cache, v_cache, gen):
    cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
               rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
    t = batch * length
    return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                cache_loc=torch.zeros(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                page_table=torch.zeros((batch, 3000), dtype=torch.int32), config=cfg)


lab._lab_inputs = small_inputs
lab._lab_soft_cache = lambda gen, rows: (torch.zeros((10, 1, 2, 64)), torch.zeros((10, 1, 2, 64)))
lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))
lab._kernel_fn_submitted = lambda data: torch.ones((data["config"]["batch"], 8, 16 * 64))
lab._prologue_q = lambda data: torch.zeros((data["config"]["batch"] * 8, 16, 64))
lab._merge = lambda data, o, m, l_, ns: torch.zeros((data["config"]["batch"], 8, 16 * 64))
lab.torch.cuda.empty_cache = lambda: None
lab._lab_main()
bad = [m for m in logs if "FAILED" in m]
keeps = [m for m in logs if m.startswith("KEEP") or m.startswith("ROUTE")]
ok = not bad and len(keeps) == 12 and any(m.startswith("V6_ROUTES = {}") for m in logs) and \
    not any(r[0] in ("ver4", "ud3") or len(r) == 5 for r in lab.CUDA_ROUTE.values())
print("dryrun_lab_l16:", "ok" if ok else f"FAILED bad={bad} keeps={len(keeps)} routes={lab.CUDA_ROUTE}")
sys.exit(0 if ok else 1)

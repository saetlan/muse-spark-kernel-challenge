"""CPU dry run of lab_l15.py's _lab_main with mocks: the CUDA modules are replaced by mocks whose
arities are checked against the C++ declarations in cuda_anat.py, and which write synthetic
stamps for the exact grid they are launched with (so a wrong row count or slot map shows up in
the report); Triton pieces, inputs, reference and graph timing are stubbed. Catches plumbing
errors (names, argument counts, buffer sizes) before a check is spent.
Run: .venv/bin/python attention/local/dryrun_lab_l15.py"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emulate  # noqa: E402

lab = emulate.load_submission(os.path.join(PROB, "lab_l15.py"))
logs = []
lab._log = lambda m: (logs.append(m), print("[lab]", m))


def arity(decl, fn):
    m = re.search(rf"\b{fn}\(([^)]*)\)", decl)
    return 0 if not m.group(1).strip() else len(m.group(1).split(","))


GHZ = 1.75


def stamp(buf, nrows, slots, cs=0, nsplit=1):
    """Synthetic stamps: entry gt staggered by waves of 396 CTAs, consecutive slots 100 ns apart."""
    b = buf.view(-1, 32)
    assert nrows <= b.shape[0], "stamp buffer too small"
    for i in range(nrows):
        g0 = 10 ** 9 + (i // 396) * 6000
        c0 = 10 ** 7
        t = c0
        b[i, 1] = t
        for s in slots:
            t += int(100 * GHZ)
            b[i, s] = t
        b[i, 28] = t + int(100 * GHZ)
        b[i, 0] = g0
        b[i, 29] = g0 + int((b[i, 28] - c0) / GHZ)
        b[i, 30] = i % 132


class Mod:
    def __init__(self, decl, calls):
        self.decl, self.calls = decl, calls

    def check(self, fn, args):
        n = arity(self.decl, fn)
        assert len(args) == n, f"{fn}: {len(args)} args, declaration has {n}"
        self.calls.append(fn)

    # micro
    def an_chase(self, *a):
        self.check("an_chase", a)
        a[3][0] = 64 * 900

    def an_null(self, *a):
        self.check("an_null", a)

    def an_bench(self, *a):
        self.check("an_bench", a)
        a[4][: a[1]] = 4096

    def an_null_setup(self, *a):
        self.check("an_null_setup", a)
        return 0

    # D3
    def an_dec4(self, *a):
        self.check("an_dec4", a)
        batch, nsplit, buf = a[9], a[13], a[-1]
        assert a[6].shape[0] == batch * 16 * nsplit
        stamp(buf, batch * nsplit, [2, 3, 4, 5, 7])

    def an_dec4_setup(self, *a):
        self.check("an_dec4_setup", a)
        return 0

    def an_dec4_attrs(self, *a):
        self.check("an_dec4_attrs", a)
        return [140, 0, 0]

    # UD2
    def an_ud2(self, *a):
        self.check("an_ud2", a)
        batch, nsplit, cs, buf = a[15], a[19], a[27], a[-1]
        assert nsplit % cs == 0
        stamp(buf, batch * nsplit, [2, 3, 4, 5, 6, 7, 9, 10, 11, 12])

    def an_ud2_setup(self, *a):
        self.check("an_ud2_setup", a)
        return 0

    def an_ud2_attrs(self, *a):
        self.check("an_ud2_attrs", a)
        return [128, 32, 4752]

    # S2c
    def an_ver3(self, *a):
        self.check("an_ver3", a)
        batch, nsplit, buf = a[9], a[13], a[-1]
        assert a[6].shape[0] == batch * 8 * 16 * nsplit
        stamp(buf, batch * nsplit, [2, 3, 4, 16, 17, 18, 19, 20])

    def an_ver3_setup(self, *a):
        self.check("an_ver3_setup", a)
        return 0

    def an_ver3_attrs(self, *a):
        self.check("an_ver3_attrs", a)
        return [165, 0, 0]


calls = []
decls = {"okc_l15_micro": lab.CUDA_AN_MICRO_DECL, "okc_l15_d3": lab.CUDA_AN_DEC4_DECL,
         "okc_l15_ud2": lab.CUDA_AN_UD2_DECL, "okc_l15_ver3": lab.CUDA_AN_VER3_DECL}
lab._an_build = lambda name, decl, src, funcs: Mod(decls[name], calls)

real_device = torch.device
lab.torch.device = lambda s="cpu", *a: real_device("cpu")
orig_zeros = torch.zeros


class FakeLab:
    def __init__(self):
        self.t0 = lab._time.monotonic()
        self.flush = orig_zeros(16)

    def left(self):
        return 1000.0

    def time_graph(self, fn, reps=15):
        fn()
        return 12.3


lab._Lab = FakeLab
lab._AnHost.time_hot = staticmethod(lambda fn, reps=15: (fn(), 4.5)[1])
real_gen = torch.Generator
lab.torch.Generator = lambda device=None: real_gen()


def small_inputs(batch, length, window, k_cache, v_cache, gen):
    max_pages = 3000
    cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=window,
               rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.47, sm_scale=0.125, kv_scale=1.0)
    t = batch * length
    return dict(qkv=torch.zeros((batch, length, 36 * 64), dtype=torch.bfloat16), positions=torch.zeros(t, dtype=torch.long),
                cache_loc=torch.zeros(t, dtype=torch.long), cache_seqlens=torch.full((batch,), 2900, dtype=torch.int32),
                k_cache=k_cache, v_cache=v_cache, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0),
                page_table=torch.zeros((batch, max_pages), dtype=torch.int32), config=cfg)


lab._lab_inputs = small_inputs
lab._lab_soft_cache = lambda gen, rows: (torch.zeros((10, 1, 2, 64)), torch.zeros((10, 1, 2, 64)))
lab.run = lambda data, tune=None: None
lab._lab_reference = lambda data: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
lab._an_q_prologue = lambda data: torch.zeros((data["config"]["batch"] * data["config"]["q_len"], 16, 64))
lab._an_combine = lambda data, o, m, l_, ns: torch.ones((data["config"]["batch"], data["config"]["q_len"], 16 * 64))
lab._an_triton_parts = lambda data: (lambda: None, lambda: None, lambda: None)
lab._inv_freq = lambda theta, device: torch.zeros(32)
lab.torch.cuda.empty_cache = lambda: None

# the micro section allocates 512 MB on the device: shrink for the dry run
src = open(os.path.join(HERE, "lab_l15_block.py")).read()
assert "words, page_words, hops = (512 << 20) // 4, (2 << 20) // 4, 64" in src
lab._lab_main()

bad = [m for m in logs if "FAILED" in m or "no stamps" in m]
need = ["AN bench mma e4m3 k32", "AN bench cvt", "AN chase 64 pages", "AN chase 1 page", "AN graph 1 cluster8", "AN D3 Sd1", "AN D3 Sd32 c2w4s3", "AN UD2 Sd32",
        "AN UD2 Fd1", "AN S2c Fv16", "AN S2c Fv1", "AN Sd1 triton", "AN UD2 Sd1 tickets after", "done in"]
missing = [k for k in need if not any(m.startswith(k) for m in logs)]
print(f"\ncalls: {sorted(set(calls))}")
print("dryrun_lab_l15:", "ok" if not bad and not missing else f"FAILED bad={bad} missing={missing}")
sys.exit(1 if bad or missing else 0)

"""CPU test of a v6 file's glue (default attention/submission_v6_lab.py): the compiled module is a
mock that runs emulate_cuda_ver4.attend_ver4 / the D2-D3 emulator from the exact positional
arguments v6 passes (a wrong order, split count or buffer shape shows up as a wrong result).
Checks: 1. the warm-up keeps every correct ver4 route (FP8 queries: tolerance 0.15) and the dec4
routes; 2. each routed verify class runs attend_ver4 with its route's stages / m-tiles / min
blocks and passes the judge's check_implementation; 3. a ver4 kernel returning a wrong result is
dropped at warm-up and its class falls back to v5's path; 4. non-FP8 kinds keep tolerance 2e-2.
Run: .venv/bin/python attention/local/test_v6_glue.py [file]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
import reference as ref  # noqa: E402
from emulate_cuda_dec3 import attend_dec3  # noqa: E402
from emulate_cuda_ud import run_ud  # noqa: E402
from emulate_cuda_ver4 import attend_ver4  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROB, "submission_v6_lab.py")
v6 = emulate.load_submission(path)
calls = []
corrupt = set()


def as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, q_len):
    cfg = dict(batch=batch, q_len=q_len, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
               sm_scale=sm_scale)
    return dict(k_cache=kc, v_cache=vc, k_scale=k_scale, v_scale=k_scale, page_table=pt, cache_seqlens=seqlens,
                config=cfg)


class Mod:
    def attend_ver4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, stages, mt, minb):
        calls.append(("ver4", nsplit, stages, mt, minb))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = attend_ver4(d, q, nsplit, stages, mt)
        o_part.copy_(o.flip(-1) if "ver4" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_dec4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps, stages):
        calls.append(("dec4", nsplit, warps, stages))
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        o, m, l_ = attend_dec3(d, q, nsplit, warps, stages)
        o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)


def ud3_mock(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
             max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs):
    calls.append(("ud3", cs, warps, stages))
    assert window > 0 and pt.shape[1] == max_pages and out.shape == (batch, 1, hq * 64)
    d = dict(qkv=qkv, positions=positions, cache_loc=cache_loc, k_cache=kc, v_cache=vc, k_scale=k_scale,
             v_scale=v_scale, page_table=pt, cache_seqlens=seqlens,
             config=dict(batch=batch, q_len=1, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
                         rope_theta=500000.0 if rope else 0.0, qk_scale_factor=qk_factor, sm_scale=sm_scale))
    res = run_ud(d, v6, cs, 1, warps, stages)       # UD3's math is UD's with one cluster (sim_ud3_merge.py: the merge)
    out.copy_((res.flip(-1) if "ud3" in corrupt else res).view_as(out))


Mod.ud3 = ud3_mock
v6.torch.cuda.is_current_stream_capturing = lambda: False
v6._prologue_q = lambda data: emulate.prologue(data, v6)
v6._merge = lambda data, o, m, l_, ns: emulate.combine(data, o, m, l_, ns, min(64, 1 << max(0, (ns - 1).bit_length())), v6)
v6.run = lambda data, tune=None: emulate.run(data, v6)
v6._kernel_fn_v22 = lambda data: (calls.append(("v22",)), emulate.run(data, v6))[1]
v6.run_pdl = lambda data, tune=None, early=False: (calls.append(("pdl",)), emulate.run(data, v6))[1]
v6._warm_up_v22 = lambda: v6._WARM.append(True)
v6._WARM_DEVICE = "cpu"
v6._cuda_build = lambda: v6._CUDA.update(mod=Mod(), refused=set())
v6._self_check_inputs = lambda bclass, length, full, device: ref.generate_input(
    seed=91 + bclass, device="cpu", batch=1, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
    q_len=length, page=1, dist="soft")
ROUTES = dict(v6.CUDA_ROUTE)
VER4 = {c: r for c, r in ROUTES.items() if r[0] == "ver4"}
UD3 = {c: r for c, r in ROUTES.items() if r[0] == "ud3"}
bad = 0


def run_class(cls, seed):
    full, verify, bclass = cls
    case = dict(batch=1, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
                q_len=8 if verify else 1, page=1, dist="soft")
    data = ref.generate_input(seed=seed, device="cpu", **case)
    calls.clear()
    sc = v6.shape_class
    v6.shape_class = lambda cfg, c=cls: c
    out = v6.kernel_fn(data)
    v6.shape_class = sc
    good, msg = ref.check_implementation(data, ref.collect(data, out))
    return good, list(calls), msg


# 1. warm-up keeps every correct route
v6._WARM.clear()
v6._warm_up()
ok = v6.CUDA_ROUTE == ROUTES and all(c in v6.CUDA_ROUTE for c in v6.V6_ROUTES) and (len(VER4), len(UD3)) in ((6, 3), (1, 1))
print(("PASS " if ok else "FAIL ") + f"warm-up keeps all {len(ROUTES)} routes ({len(VER4)} ver4, {len(UD3)} ud3)")
bad += not ok
for i, (cls, route) in enumerate(sorted(UD3.items())):
    good, cl, msg = run_class(cls, 350 + i)
    ok = good and cl and cl[0] == route
    print(("PASS " if ok else "FAIL ") + f"{cls} -> {route}: {cl} {msg[:60]}")
    bad += not ok
ok = not v6._cuda_ok(dict(num_q_heads=16, num_kv_heads=2, q_len=1, window=0), ("ud3", 8, 4, 3))
print(("PASS " if ok else "FAIL ") + "ud3 refuses full layers")
bad += not ok
# 2. each verify class runs its ver4 route
for i, (cls, route) in enumerate(sorted(VER4.items())):
    good, cl, msg = run_class(cls, 300 + i)
    ok = good and cl and cl[0][0] == "ver4" and cl[0][2:] == route[2:]
    print(("PASS " if ok else "FAIL ") + f"{cls} -> {route}: {cl} {msg[:60]}")
    bad += not ok
# 4. tolerance per kind
ok = v6._self_check_tol(("ver4", 1, 3, 2, 1)) == 0.15 and v6._self_check_tol(("dec4", 6, 4, 3)) == v6.SELF_CHECK_TOL == 2e-2
print(("PASS " if ok else "FAIL ") + "self-check tolerance: ver4 0.15, dec4 2e-2")
bad += not ok
# 3. a wrong ver4 result is dropped at warm-up; its class falls back to v5's path
corrupt.update({"ver4", "ud3"})
v6.CUDA_ROUTE.clear()
v6.CUDA_ROUTE.update(ROUTES)
v6._WARM.clear()
v6._warm_up()
ok = all(c not in v6.CUDA_ROUTE for c in list(VER4) + list(UD3)) and \
    all(v6.CUDA_ROUTE.get(c) == r for c, r in ROUTES.items() if c not in VER4 and c not in UD3)
print(("PASS " if ok else "FAIL ") + f"corrupted ver4/ud3 dropped at warm-up (left: {sorted(v6.CUDA_ROUTE)})")
bad += not ok
corrupt.clear()
for cls, want in (((False, True, 16), "pdl"), ((True, True, 16), "v22"), ((False, False, 16), "pdl"),
                  ((False, False, 1), "v22")):
    good, cl, msg = run_class(cls, 400)
    ok = good and cl and cl[0][0] == want
    print(("PASS " if ok else "FAIL ") + f"{cls} after drop -> {cl[:1]} {msg[:40]}")
    bad += not ok
# 5. whole-wave D3 routes (AF33): floor split, same module function, judge check passes
cfg16 = dict(batch=16, q_len=1, num_kv_heads=2, page_size=1, window=0)
cfg32 = dict(cfg16, batch=32)
ok = v6._nsplit_w(cfg16, 86001, 6, 128) == 49 and v6._nsplit_w(cfg32, 86001, 6, 128) == 24 and \
    v6._nsplit_w(dict(cfg16, batch=1), 80001, 3, 128) == 396
print(("PASS " if ok else "FAIL ") + "whole-wave split: Fd16 c6 -> 49, Fd32 c6 -> 24, Fd1 c3 -> 396")
bad += not ok
v6.CUDA_ROUTE[(True, False, 1)] = ("dec4", 3, 4, 3, "w")
good, cl, msg = run_class((True, False, 1), 500)
ok = good and cl and cl[0][0] == "dec4" and cl[0][2:] == (4, 3)
print(("PASS " if ok else "FAIL ") + f"(True, False, 1) -> dec4 whole-wave: {cl} {msg[:50]}")
bad += not ok
print("test_v6_glue:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)

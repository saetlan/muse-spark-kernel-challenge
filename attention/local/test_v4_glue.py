"""CPU test of submission_v4.py's host glue. The compiled module is replaced by mocks that run our
register-level emulators from the exact positional arguments v4 passes (a wrong argument order,
buffer shape, slice or cluster count shows up as a wrong result); Triton's prologue, merge and
run() by emulate.py's. Checks:
1. every kind (dec2, dec3, ud, ver2) routed on a class runs through kernel_fn and passes the
   judge's check_implementation, calling exactly its kind;
2. the warm-up self-check keeps correct routes, drops a route whose kernel is wrong (mock
   returns a corrupted result) and a kind whose setup was refused, and those classes then
   run the v2.2 path;
3. an unrouted class never reaches the mocks.
Run: .venv/bin/python attention/local/test_v4_glue.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
from emulate_cuda_dec2 import attend_dec2  # noqa: E402
from emulate_cuda_dec3 import attend_dec3  # noqa: E402
from emulate_cuda_ud import run_ud  # noqa: E402
from emulate_cuda_ver2 import attend_ver2  # noqa: E402

import reference as ref  # noqa: E402

v4 = emulate.load_submission(os.path.join(PROB, "submission_v4.py"))
calls = []
corrupt = set()     # kinds whose mock returns a wrong result (self-check test)


def as_data(kc, vc, k_scale, page_table, seqlens, batch, hq, hkv, page, window, sm_scale, q_len):
    cfg = dict(batch=batch, q_len=q_len, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
               sm_scale=sm_scale)
    return dict(k_cache=kc, v_cache=vc, k_scale=k_scale, v_scale=k_scale, page_table=page_table,
                cache_seqlens=seqlens, config=cfg)


def store(kind, o, m, l_, o_part, m_part, l_part):
    if kind in corrupt:
        o = o.flip(-1)
    o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)


class Mod:
    def attend_dec2(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps):
        calls.append(("dec2", nsplit, warps))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        store("dec2", *attend_dec2(d, q, nsplit, warps), o_part, m_part, l_part)

    def attend_dec3(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps, stages):
        calls.append(("dec3", nsplit, warps, stages))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        store("dec3", *attend_dec3(d, q, nsplit, warps, stages), o_part, m_part, l_part)

    def attend_dec4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps, stages):
        calls.append(("dec4", nsplit, warps, stages))   # D3: D2's math, chunking and layout
        assert pt.shape[1] == max_pages and o_part.shape == (batch * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        store("dec4", *attend_dec3(d, q, nsplit, warps, stages), o_part, m_part, l_part)

    def attend_ver2(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, stages):
        calls.append(("ver2", nsplit, stages))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        store("ver2", *attend_ver2(d, q, nsplit, stages), o_part, m_part, l_part)

    def ud(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, cpart_o, cpart_m,
           cpart_l, tickets, out, batch, hq, hkv, max_pages, nsplit, page, window, rope, qk_factor, sm_scale,
           warps, stages, cs):
        calls.append(("ud", nsplit, cs, warps, stages))
        assert nsplit % cs == 0 and int(tickets.abs().sum()) == 0 and pt.shape[1] == max_pages
        nc = nsplit // cs
        assert cpart_o.shape == (batch * hq * nc, 64) and cpart_m.shape == (batch * hq * nc,)
        assert window == 0 or nc == 1
        d = dict(qkv=qkv, positions=positions, cache_loc=cache_loc, k_cache=kc, v_cache=vc, k_scale=k_scale,
                 v_scale=v_scale, page_table=pt, cache_seqlens=seqlens,
                 config=dict(batch=batch, q_len=1, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
                             rope_theta=500000.0 if rope else 0.0, qk_scale_factor=qk_factor, sm_scale=sm_scale))
        res = run_ud(d, v4, cs, nc, warps, stages)
        out.copy_((res.flip(-1) if "ud" in corrupt else res).view_as(out))

    def ud_setup(self):
        return 0


def triton_merge(data, o, m, l_, nsplit):
    return emulate.combine(data, o, m, l_, nsplit, min(64, 1 << max(0, (nsplit - 1).bit_length())), v4)


v4.torch.cuda.is_current_stream_capturing = lambda: False
v4._prologue_q = lambda data: emulate.prologue(data, v4)
v4._merge = triton_merge
v4._kernel_fn_v22 = lambda data: emulate.run(data, v4)   # v2.2's path, emulated (SGLang isn't on the Mac)
v4.run = lambda data, tune=None: emulate.run(data, v4)  # the self-check's reference path
v4._warm_up_v22 = lambda: v4._WARM.append(True)
v4._WARM_DEVICE = "cpu"

ROUTES = {   # one class per kind, plus the cluster/second-level variants of UD
    (False, False, 1): ("ud", 16, 4, 4, 3),
    (False, False, 16): ("ud", 8, 4, 4, 3),
    (True, False, 1): ("ud", 8, 2, 4, 3),
    (True, False, 16): ("dec3", 8, 4, 3),
    (True, False, 32): ("dec2", 8, 4),
    (False, False, 32): ("dec4", 6, 4, 3),
    (True, True, 16): ("ver2", 2, 3),
}
bad = 0


def check_class(cls, route, expect_kind, seed):
    full, verify, bclass = cls
    batch = 2 if bclass > 1 else 1
    case = dict(batch=batch, ctx=900 if full else 2300, heads="tp8", layer="full" if full else "sliding",
                q_len=8 if verify else 1, page=1, dist="soft")
    data = ref.generate_input(seed=seed, device="cpu", **case)
    data["config"]["batch"] = batch
    calls.clear()
    if bclass != v4.batch_class(batch):   # make the small batch map to the class under test
        v4_shape_class = v4.shape_class
        v4.shape_class = lambda cfg: cls
    out = v4.kernel_fn(data)
    if bclass != v4.batch_class(batch):
        v4.shape_class = v4_shape_class
    ok, msg = ref.check_implementation(data, ref.collect(data, out))
    used = [c[0] for c in calls]
    ok = ok and used == ([expect_kind] if expect_kind else [])
    print(("PASS " if ok else "FAIL ") + f"{cls} -> {route}: calls {calls} {msg}")
    return not ok


# 1. warm-up with correct mocks keeps every route; each kind serves its class
v4._cuda_build = lambda: v4._CUDA.update(mod=Mod(), refused=set())
v4.CUDA_ROUTE.clear()
v4.CUDA_ROUTE.update(ROUTES)
v4._WARM.clear()
v4._TICKETS.clear()
v4._warm_up()
ok = v4.CUDA_ROUTE == ROUTES
print(("PASS " if ok else "FAIL ") + f"self-check keeps all {len(ROUTES)} correct routes (left: {sorted(v4.CUDA_ROUTE)})")
bad += not ok
for i, (cls, route) in enumerate(sorted(ROUTES.items())):
    bad += check_class(cls, route, route[0], 1200 + i)

# 2. a wrong kernel and a refused setup are dropped at warm-up; their classes run v2.2
corrupt.update({"ver2", "ud"})
v4._cuda_build = lambda: v4._CUDA.update(mod=Mod(), refused={"dec3"})
v4.CUDA_ROUTE.clear()
v4.CUDA_ROUTE.update(ROUTES)
v4._WARM.clear()
v4._TICKETS.clear()
v4._warm_up()
left = {cls: r for cls, r in ROUTES.items() if r[0] in ("dec2", "dec4")}
ok = v4.CUDA_ROUTE == left
print(("PASS " if ok else "FAIL ") + f"self-check drops corrupted ud/ver2 and refused dec3 (left: {v4.CUDA_ROUTE})")
bad += not ok
corrupt.clear()
bad += check_class((True, True, 16), ROUTES[(True, True, 16)], None, 1300)
bad += check_class((True, False, 16), ROUTES[(True, False, 16)], None, 1301)

# 3. a failed build leaves no route at all
v4._CUDA.clear()
v4._cuda_build = lambda: None
v4.CUDA_ROUTE.update(ROUTES)
v4._WARM.clear()
v4._warm_up()
ok = not v4.CUDA_ROUTE
print(("PASS " if ok else "FAIL ") + "failed build clears every route")
bad += not ok
bad += check_class((False, False, 1), ROUTES[(False, False, 1)], None, 1302)
print("test_v4_glue:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)

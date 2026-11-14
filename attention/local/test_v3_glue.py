"""CPU test of submission_v3.py's host glue: the compiled module is replaced by mocks that run our
register-level emulators from the exact positional arguments v3 passes (so a wrong argument order,
buffer shape or slice count shows up as a wrong result), Triton's prologue/merge by emulate.py's.
Every routed class runs through kernel_fn on small inputs of that class and must pass the
judge's check_implementation; unrouted classes must not touch the mocks.
Run: .venv/bin/python attention/local/test_v3_glue.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
from emulate_cuda_dec2 import attend_dec2  # noqa: E402
from emulate_cuda_fused import fold_queries, new_rows  # noqa: E402
from emulate_cuda_ver import attend_ver  # noqa: E402

import reference as ref  # noqa: E402

v3 = emulate.load_submission(os.path.join(PROB, "submission_v3.py"))
calls = []


def as_data(kc, vc, k_scale, page_table, seqlens, batch, hq, hkv, page, window, sm_scale, q_len, v_scale=None):
    cfg = dict(batch=batch, q_len=q_len, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
               sm_scale=sm_scale)
    return dict(k_cache=kc, v_cache=vc, k_scale=k_scale, v_scale=v_scale if v_scale is not None else k_scale,
                page_table=page_table, cache_seqlens=seqlens, config=cfg)


class Mod:
    def attend_dec2(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps):
        calls.append(("dec2", nsplit, warps))
        assert pt.shape[1] == max_pages
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        o, m, l_ = attend_dec2(d, q, nsplit, warps)
        o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)

    def attend_ver(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                   nsplit, page, window, sm_scale, stages):
        calls.append(("ver", nsplit, stages))
        assert pt.shape[1] == max_pages
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = attend_ver(d, q, nsplit, stages)
        o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)

    def fused_dec(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, o_part, m_part,
                  l_part, tickets, out, batch, hq, hkv, max_pages, nsplit, page, window, rope, qk_factor, sm_scale,
                  warps):
        calls.append(("fused", nsplit, warps))
        assert nsplit <= 128 and int(tickets.abs().sum()) == 0
        d = dict(qkv=qkv, positions=positions, cache_loc=cache_loc, k_cache=kc, v_cache=vc, k_scale=k_scale,
                 v_scale=v_scale, page_table=pt, cache_seqlens=seqlens,
                 config=dict(batch=batch, q_len=1, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
                             rope_theta=500000.0 if rope else 0.0, qk_scale_factor=qk_factor, sm_scale=sm_scale))
        q32 = fold_queries(d, v3)
        new_rows(d, v3)
        o, m, l_ = attend_dec2(d, q32, nsplit, warps)
        out.copy_(emulate.combine(d, o, m, l_, nsplit, 1 << max(0, (nsplit - 1).bit_length()), v3).view_as(out))


v3._CUDA["mod"] = Mod()
v3._TICKETS.append(torch.zeros(4096, dtype=torch.int32))
v3._WARM.append(True)
v3.torch.cuda.is_current_stream_capturing = lambda: False
v3._prologue_q = lambda data: emulate.prologue(data, v3)
v3._merge = lambda data, o, m, l_, nsplit: emulate.combine(data, o, m, l_, nsplit, min(64, 1 << max(0, (nsplit - 1).bit_length())), v3)
v3._kernel_fn_v22 = lambda data: emulate.run(data, v3)   # v2.2 Triton path, emulated (SGLang isn't on the Mac)

bad = 0
for (full, verify, bclass), route in sorted(v3.CUDA_ROUTE.items()):
    case = dict(batch=bclass, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
                q_len=8 if verify else 1, page=1, dist="soft")
    data = ref.generate_input(seed=900 + bclass, device="cpu", **case)
    calls.clear()
    out = v3.kernel_fn(data)
    ok, msg = ref.check_implementation(data, ref.collect(data, out))
    used = [c[0] for c in calls]
    ok = ok and used == [route[0]]
    print(("PASS " if ok else "FAIL ") + f"route {(full, verify, bclass)} -> {route}: calls {calls} {msg}")
    bad += not ok
# an unrouted class must not reach the CUDA mocks
case = dict(batch=1, ctx=300, heads="tp8", layer="sliding", page=1, dist="soft")
data = ref.generate_input(seed=950, device="cpu", **case)
calls.clear()
v3.kernel_fn(data)
ok = v3.shape_class(data["config"]) in v3.CUDA_ROUTE or not calls
print(("PASS " if ok else "FAIL ") + f"unrouted class {v3.shape_class(data['config'])} stays on v2.2 (calls {calls})")
bad += not ok
print("test_v3_glue:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)

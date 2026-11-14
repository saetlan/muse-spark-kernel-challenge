"""CPU test of submission_v5.py's glue: the PDL classes run run_pdl with the late trigger (emulated
as run: same math, the PDL lines are the only differences per check_pdl.py), CUDA routes keep
priority, every other class falls through to v4; the warm-up keeps the PDL classes when the chain
matches the Triton path and drops all of them on an exception.
Run: .venv/bin/python attention/local/test_v5_glue.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
import reference as ref  # noqa: E402

v5 = emulate.load_submission(os.path.join(PROB, "submission_v5.py"))
calls = []
v5.torch.cuda.is_current_stream_capturing = lambda: False
v5.run_pdl = lambda data, tune=None, early=False: (calls.append(("pdl", early)), emulate.run(data, v5))[1]
v5.run = lambda data, tune=None: emulate.run(data, v5)
v5._kernel_fn_v4 = lambda data: (calls.append(("v4", v5.shape_class(data["config"]))), emulate.run(data, v5))[1]
v5._warm_up_v4 = lambda: v5._WARM.append(True)
v5._WARM_DEVICE = "cpu"
v5._self_check_inputs = lambda bclass, length, full, device: ref.generate_input(
    seed=77, device="cpu", batch=1, ctx=600 if full else 2300, heads="tp8", layer="full" if full else "sliding",
    q_len=length, page=1, dist="soft")
bad = 0
v5._WARM.clear()
v5._warm_up()
ok = len(v5.PDL_ROUTE) == 6
print(("PASS " if ok else "FAIL ") + f"warm-up keeps the 6 PDL classes: {sorted(v5.PDL_ROUTE)}")
bad += not ok
for cls in [(False, False, 16), (False, True, 1), (True, False, 1), (False, False, 1), (True, True, 1), (True, False, 16)]:
    full, verify, bclass = cls
    case = dict(batch=1, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
                q_len=8 if verify else 1, page=1, dist="soft")
    data = ref.generate_input(seed=880 + bclass, device="cpu", **case)
    calls.clear()
    sc = v5.shape_class
    v5.shape_class = lambda cfg, c=cls: c
    out = v5.kernel_fn(data)
    v5.shape_class = sc
    good, msg = ref.check_implementation(data, ref.collect(data, out))
    want = "v4" if cls in v5.CUDA_ROUTE or cls not in v5.PDL_ROUTE else "pdl"
    ok = good and calls and calls[0][0] == want and (want != "pdl" or calls[0][1] is False)
    print(("PASS " if ok else "FAIL ") + f"{cls}: {calls} {msg[:40]}")
    bad += not ok
v5.run_pdl = lambda data, tune=None, early=False: (_ for _ in ()).throw(RuntimeError("launch_pdl unsupported"))
v5._WARM.clear()
v5._warm_up()
ok = not v5.PDL_ROUTE
print(("PASS " if ok else "FAIL ") + "a PDL failure at warm-up drops every PDL class")
bad += not ok
print("test_v5_glue:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)

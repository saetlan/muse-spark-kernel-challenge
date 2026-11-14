"""CPU dry run of proj/lab_e48.py (base v13): v11b's modules mocked as in check_v11b.py; okc_proj_f12mq mocked from its source
(namespace renamed, padded part[], exactly F12MQ_INSTANCES) with each function checking its instance against the call (tokens
<= 8 NT, rows % RB, K = 64 NS KSW) and computing x @ W from the decoded F12M planes. Planted: o m16 NT2 RTW4 RG1 KSW8 D1 drops
its last 64 k (rejected as wrong). Fake timer: f12mq 5% faster than the incumbent, so every other shape moves with its
incumbent as fallback; the judge's check passes on every shape.

Run: .venv/bin/python proj/local/dryrun_lab_e48.py
"""

import io
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_f12m  # noqa: E402
import check_make_dgrs as cmd  # noqa: E402
import check_v11b  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import make_e48  # noqa: E402

PLANT = (1, 1, 1, 8, 16, 1, True)


def mock_mq(seen):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_f12mh2":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "__shared__ float part[KSW][RB][MT + 3];" in cuda_sources and "namespace f12mh2 {" in cuda_sources and "corr[row][tok] = c;" in cuda_sources and ", 256, 0, stream>>>" not in cuda_sources
        assert "namespace f12m {" not in cuda_sources and " f12mh::" not in cuda_sources
        insts = {tuple(int(v) if v not in ("true", "false") else v == "true" for v in t.split(", "))
                 for t in re.findall(r"f12mh2::f12m_kernel<([^>]*)><<<", cuda_sources)}
        seen["insts"] = insts
        mod = types.SimpleNamespace()
        for f in functions:
            if f.endswith("_info"):
                setattr(mod, f, lambda: 96 * 1000000)
                continue
            parts = [int(v) for v in f[len("f12mh2_run_"):].split("_") if v.isdigit()]
            nt, rtw, rg, ksw, ns, dd, f12 = parts[:7]

            def run(wp, rec, x, out, m, n, inst=(nt, rtw, rg, ksw, ns, dd, f12)):
                nt_, rtw_, rg_, ksw_, ns_ = inst[:5]
                k = x.shape[1]
                assert 1 <= m <= 8 * nt_ and n % (16 * rg_ * rtw_) == 0 and k == 64 * ns_ * ksw_, (inst, m, n, k)
                if id(wp) not in cache:
                    cache[id(wp)] = check_f12m.decode_weights({"p": wp, "rec": rec}, k, dec)
                xd = x.double()
                if inst[:6] == PLANT[:6]:
                    xd = xd.clone()
                    xd[:, -64:] = 0
                out.copy_((xd @ cache[id(wp)][:n].T).to(torch.bfloat16))
            setattr(mod, f, run)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    make_e48.build()
    cmd.setup()
    check_v11b.mock_v10({})
    import check_v13
    check_v13.mock_mh({})
    seen = {}
    mock_mq(seen)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e48.py", "lab_e48_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 9.5 if c is not None and c["kind"] == "f12mh2" else 10.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    assert {i[:7] for i in seen["insts"]} == set(lab.E48_INSTANCES) and all(i[7:] in ((1, 1), (1, i[5])) for i in seen["insts"]), seen
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] mh2 ") and "(sglang" in ln]
    assert len(lines) == len(lab.E48_TARGETS), (lines, out[-2500:])
    wrong = [ln for ln in lines if " 1 wrong" in ln]
    assert len(wrong) == 1 and wrong[0].startswith("[lab] mh2 qkv_gate m8"), lines
    moved = {k: c for k, c in lab.CHOICE.items() if c.get("kind") == "f12mh2"}
    assert set(moved) == set(lab.E48_TARGETS), (sorted(moved), out[-2000:])
    for k, c in moved.items():
        assert c["fallback"] == base_choice.get(k), (k, c)
        assert not (k == ("qkv_gate", 8) and c["D"] == 1), c
    assert out.count("MH2 CHOICE") == len(moved)
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 9700 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    print(f"ok: lab_e48.py dry run; {len(seen['insts'])} helper instances in okc_proj_f12mh2; planted wrong qkv m8 D1 config "
          f"rejected; {len(moved)} shapes moved with fallbacks; judge's check on {len(lab.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main()

"""CPU check of proj/submission_v13.py (v12 + F12MH helper warp on MH_CHOICE shapes + extra padded F12M configs on MQ_CHOICE
shapes): v11b's modules mocked as in check_v11b.py (F12M mocked from the instance names, so extra instances work);
okc_proj_f12mh mocked from its source (helper edits present, 288-thread launches, exactly F12MH_INSTANCES) and computing x @ W
from the decoded planes. prepare(); CHOICE = v12's except the MH / MQ shapes (kinds f12mh / f12m, v12's entries as fallbacks);
the judge's check on every shape x 2 inputs; the MH shapes really call the helper module.

Run: .venv/bin/python proj/local/check_v13.py [proj/submission_v13.py]
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


def mock_mh(calls):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_f12mh":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        assert "corr[row][tok] = c;" in cuda_sources and "sum = fmaf(sum, f12m_scale(base), corr[row][tok]);" in cuda_sources
        assert "__shared__ float part[KSW][RB][MT + 3];" in cuda_sources and "rc0" not in cuda_sources
        assert ", 256, 0, stream>>>" not in cuda_sources and "namespace f12mh {" in cuda_sources
        calls["insts"] = {tuple(int(v) if v not in ("true", "false") else v == "true" for v in t.split(", "))[:7]
                          for t in re.findall(r"f12mh::f12m_kernel<([^>]*)><<<", cuda_sources)}
        mod = types.SimpleNamespace()
        for f in functions:
            if f.endswith("_info"):
                setattr(mod, f, lambda: 90 * 1000000)
                continue

            def run(wp, rec, x, out, m, n, f=f):
                calls[f] = calls.get(f, 0) + 1
                k = x.shape[1]
                if id(wp) not in cache:
                    cache[id(wp)] = check_f12m.decode_weights({"p": wp, "rec": rec}, k, dec)
                out.copy_((x.double() @ cache[id(wp)][:n].T).to(torch.bfloat16))
            setattr(mod, f, run)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main(path):
    cmd.setup()
    check_v11b.mock_v10({})
    calls = {}
    mock_mh(calls)
    w = d.weights()
    base = d.load(d.PROJ / "submission_v12.py", "v12_ref_cpu")
    mod = d.load(Path(path), "v13_cpu")
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    assert mod.F12MH_MOD is not None and calls["insts"] == set(mod.F12MH_INSTANCES), (calls, mod.F12MH_INSTANCES)
    assert set(mod.MQ_INSTANCES) <= set(mod.F12M_INSTANCES)
    for key, c in mod.CHOICE.items():
        if key in mod.MH_CHOICE:
            assert c["kind"] == "f12mh" and c["fallback"] == base.CHOICE.get(key), (key, c)
        elif key in mod.MQ_CHOICE:
            assert c["kind"] == "f12m" and c["fallback"] == base.CHOICE.get(key), (key, c)
        else:
            assert c == base.CHOICE[key], key
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(mod.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 9900 + m, dist=dist, device="cpu")
            before = sum(v for k, v in calls.items() if k != "insts")
            ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
            ran_mh = sum(v for k, v in calls.items() if k != "insts") > before
            assert ran_mh == ((layer, m) in mod.MH_CHOICE), (layer, m, ran_mh)
    print(f"ok: {Path(path).name} CPU check; helper warp on {sorted(mod.MH_CHOICE)}, extra F12M on {sorted(mod.MQ_CHOICE)}, the "
          f"rest = v12; judge's check on all {len(mod.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(d.PROJ / "submission_v13.py"))

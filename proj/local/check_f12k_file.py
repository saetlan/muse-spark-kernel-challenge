"""CPU check of a clean F12K file built by make_f12k.py (default proj/submission_v12.py): v11b's modules mocked as in
check_v11b.py, "okc_proj_f12k" mocked from its source (dryrun_lab_e42.mock_f12k without plants: each call checks its
arguments like the C++ wrapper, counters at 0 on entry); prepare(); every F12K_CHOICE shape in CHOICE as kind f12k with v11b's
entry as fallback; the judge's check_implementation on every chosen shape x 2 input distributions; every F12K shape really
ran F12K; counters back at 0; instances built = the chosen configs' instances.

Run: .venv/bin/python proj/local/check_f12k_file.py [proj/submission_v12.py]
"""

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_make_dgrs as cmd  # noqa: E402
import check_v11b  # noqa: E402
import dryrun_lab_e1 as d  # noqa: E402
import dryrun_lab_e42 as dr  # noqa: E402


def main(path):
    cmd.setup()
    check_v11b.mock_v10({})
    calls, built = {}, {}
    dr.mock_f12k(calls, built, plant=False)
    torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(multi_processor_count=132)
    w = d.weights()
    base = d.load(d.PROJ / "scored" / "submission_v11b.py", "v11b_ref_cpu")
    mod = d.load(Path(path), "f12k_file_cpu")
    with redirect_stdout(io.StringIO()):
        state = mod.prepare(w)
    assert mod.F12K_MOD is not None and built["insts"] == set(mod.F12K_INSTANCES), (built, mod.F12K_INSTANCES)
    assert set(mod.F12K_INSTANCES) == {(c["NT"], c["KSW"], c["D"]) for c in mod.F12K_CHOICE.values()}
    for key, c in mod.F12K_CHOICE.items():
        got = mod.CHOICE[key]
        assert got["kind"] == "f12k" and {k: v for k, v in got.items() if k != "fallback"} == c, (key, got)
        assert got["fallback"] == base.CHOICE.get(key), (key, got["fallback"], base.CHOICE.get(key))
    others = {k: c for k, c in mod.CHOICE.items() if k not in mod.F12K_CHOICE}
    assert others == {k: c for k, c in base.CHOICE.items() if k not in mod.F12K_CHOICE}, "non-F12K shapes changed"
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(mod.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 4900 + m, dist=dist, device="cpu")
            before = sum(calls.values())
            ok, msg = reference.check_implementation(data, mod.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
            assert ((layer, m) in mod.F12K_CHOICE) == (sum(calls.values()) > before), (layer, m, "F12K ran / didn't run")
    assert all(int(b["cnt"].abs().sum()) == 0 for b in mod.F12K_STATE["bufs"].values()), "counters not at 0"
    print(f"ok: {Path(path).name} CPU check; F12K on {sorted(mod.F12K_CHOICE)} with v11b's entries as fallbacks, the other "
          f"{len(others)} shapes unchanged; judge's check on all {len(mod.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(d.PROJ / "submission_v12.py"))

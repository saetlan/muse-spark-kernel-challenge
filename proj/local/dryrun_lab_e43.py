"""CPU dry run of proj/lab_e43.py (v11b + F12K v2 section + sweep + f12k_lab + e43 with PE42's winners): the base mocked as in
check_v11b.py, "okc_proj_f12k" mocked from its v2 source (F12 planes or the router's hi + lo fragments, FP32 output for HL).
Planted (when no real winners are filled in, a sample winner set is used): router m16 NT2 KSW4 D2 G512 drops its last 64 k
(rejected as wrong). Checks: the instances built = the router targets' and the winners'; every winner in CHOICE with v11b's
entry as fallback; the router shapes moved with fallbacks; the judge's check on every shape x 2 inputs, F12K running on the
moved ones; counters at 0.

Run: .venv/bin/python proj/local/dryrun_lab_e43.py
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
import make_e43  # noqa: E402

SAMPLE = {("qkv_gate", 16): {"kind": "f12k", "NT": 2, "KSW": 8, "D": 2, "G": 264},
          ("unpooler", 8): {"kind": "f12k", "NT": 1, "KSW": 8, "D": 2, "G": 256}}


def main():
    winners = SAMPLE if "--sample" in sys.argv else make_e43.PE42_WINNERS
    out_path = d.PROJ / "lab_e43_dry.py"
    make_e43.build(winners, out_path)
    cmd.setup()
    check_v11b.mock_v10({})
    calls, built = {}, {}
    dr.WRONG = ((2, 4, 2, True), 512, 256)
    dr.DIRTY = ((9, 9, 9), 0, 0)
    dr.mock_f12k(calls, built, plant=True)
    torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(multi_processor_count=132)
    w = d.weights()
    base = d.load(d.PROJ / "scored" / "submission_v11b.py", "v11b_ref_cpu43")
    lab = d.load(out_path, "lab_e43_cpu")

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        return 4.0 + 0.001 * c["G"] if c is not None and c["kind"] == "f12k" else 20.0
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    targets = lab.e43_targets(132)
    want_insts = {lab.f12k_inst(c) for cs in targets.values() for c in cs} | {lab.f12k_inst(c) for c in winners.values()}
    assert built["insts"] == want_insts, (built, want_insts)
    assert out.count("PE42 winner ") == len(winners) and "WRONG" not in "".join(l for l in out.splitlines() if "PE42 winner" in l), out[-2500:]
    for key, c in winners.items():
        got = lab.CHOICE[key]
        assert got["kind"] == "f12k" and {k: v for k, v in got.items() if k != "fallback"} == c and got["fallback"] == base.CHOICE.get(key)
    for key, cs in targets.items():
        got = lab.CHOICE[key]
        assert got["kind"] == "f12k" and bool(got.get("HL")) == (key[0] == "router") and got["fallback"] == base.CHOICE.get(key), (key, got)
        assert {k: v for k, v in got.items() if k != "fallback"} in cs, (key, got)
        assert not (key == ("router", 16) and (got["NT"], got["KSW"], got["D"], got["G"]) == (2, 4, 2, 512)), got
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] f12k router") and "(sglang" in ln]
    assert len(lines) == 4 and sum(" 1 wrong" in ln for ln in lines) == 1 and "router m16" in [ln for ln in lines if " 1 wrong" in ln][0], lines
    assert "f12k counters: " in out and "DIRTY" not in out, out[-1500:]
    sys.path.insert(0, str(d.PROJ))
    import reference
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 5400 + m, dist=dist, device="cpu")
            before = sum(calls.values())
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
            is_f12k = lab.CHOICE[(layer, m)]["kind"] == "f12k"
            assert is_f12k == (sum(calls.values()) > before), (layer, m)
    out_path.unlink()
    print(f"ok: lab_e43 dry run ({'sample' if '--sample' in sys.argv else 'real'} PE42 winners: {sorted(winners)}); "
          f"{len(built['insts'])} instances; planted wrong router config rejected; {len(targets)} shapes moved with fallbacks; "
          f"judge's check on all {len(lab.CHOICE)} shapes x 2 inputs")


if __name__ == "__main__":
    main()

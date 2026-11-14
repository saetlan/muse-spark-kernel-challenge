"""CPU dry run of proj/lab_e42.py (v11b + F12K section + lab/sweep.py + lab/f12k_lab.py + lab/e42.py): the base mocked as in
check_v11b.py, "okc_proj_f12k" mocked from its source text (instances parsed, each wrapper checks its arguments like the
C++ one: tokens, rows, maxp >= the pieces per tile, scratch sizes, counters at 0 on entry; x @ W from the decoded F12M
planes). Planted: unpooler m8 NT1 KSW4 D2 G512 drops its last 64 k (must be rejected as wrong); qkv m16 NT2 KSW8 D2 G396 leaves
a counter at 1 and times fastest (chosen by the sweep, then put back to v11b's entry by the counters check). Then the judge's
check_implementation on every chosen shape, two input distributions, with the moved shapes really running F12K.

Run: .venv/bin/python proj/local/dryrun_lab_e42.py
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

sys.path.insert(0, str(d.PROJ / "f12m"))
import f12k  # noqa: E402

WRONG = ((1, 4, 2), 512, 1024)   # (inst, grid, n_rows)
DIRTY = ((2, 8, 2), 396, 2304)


def unpack_hilo(frag, k):
    """FP64 hi + lo [N, K] from p9_pack_hilo's [N/16, K/16, 2, 32, 8] fragments (inverse of p9_pack, then k_perm undone)."""
    from p9_layout import k_perm
    rt, ks = frag.shape[0], frag.shape[1]
    w = []
    for h in range(2):
        v = frag[:, :, h].reshape(rt, ks, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(16 * rt, k)
        nat = torch.empty_like(v)
        nat[:, k_perm(k)] = v
        w.append(nat.double())
    return w[0] + w[1]


def mock_f12k(calls, built, plant=True):
    prev = sys.modules["torch.utils.cpp_extension"].load_inline
    dec = check_f12m.helpers()["decode"]
    cache = {}

    def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kw):
        if name != "okc_proj_f12k":
            return prev(name, cpp_sources, cuda_sources, functions, **kw)
        insts = {tuple(int(v) for v in t) for t in re.findall(r"f12k::f12k_kernel<(\d+), (\d+), (\d+)><<<", cuda_sources)}
        insts |= {(int(a), int(b_), int(c_)) + ((True,) if hl == "true" else ()) for a, b_, c_, hl in
                  re.findall(r"f12k::f12k_kernel<(\d+), (\d+), (\d+), (true|false)><<<", cuda_sources)}
        assert insts and "@@" not in cuda_sources, insts
        assert sorted(functions) == sorted([f12k.f12k_name(i) for i in insts] + [f12k.f12k_name(i) + "_info" for i in insts])
        built["insts"] = insts
        mod = types.SimpleNamespace()
        for inst in insts:
            def run(wp, rec, x, out, slots, cnt, m, n_rows, grid, maxp, inst=inst):
                nt, ksw, dd = inst[:3]
                hl = len(inst) > 3
                k = x.shape[1]
                n_tiles, chs = n_rows // 16, k // 64
                assert x.shape == (m, k) and x.is_contiguous() and k % 64 == 0 and 1 <= m <= 8 * nt, (inst, x.shape, m)
                assert n_rows % 16 == 0 and out.shape == (m, n_rows) and out.dtype == (torch.float32 if hl else torch.bfloat16)
                assert 1 <= grid <= n_tiles * chs and maxp >= max(f12k.f12k_pieces(n_tiles, chs, grid)), (grid, maxp)
                assert slots.dtype == torch.float32 and slots.numel() >= n_tiles * maxp * 8 * nt * 16
                assert cnt.dtype == torch.int32 and cnt.numel() >= n_tiles
                assert wp.numel() * wp.element_size() >= n_tiles * chs * (4096 if hl else 1536)
                assert hl or rec.numel() * rec.element_size() >= n_rows * 32
                key = (inst, grid, n_rows)
                assert (plant and key == DIRTY) or int(cnt.abs().sum()) == 0, ("counters dirty on entry", key)
                calls[key] = calls.get(key, 0) + 1
                if id(wp) not in cache:
                    cache[id(wp)] = unpack_hilo(wp, k) if hl else check_f12m.decode_weights({"p": wp, "rec": rec}, k, dec)
                xd = x.double()
                if plant and key == WRONG:
                    xd = xd.clone()
                    xd[:, -64:] = 0
                out.copy_((xd @ cache[id(wp)][:n_rows].T).to(out.dtype))
                if plant and key == DIRTY:
                    cnt[0] = 1
            setattr(mod, f12k.f12k_name(inst), run)
            setattr(mod, f12k.f12k_name(inst) + "_info", lambda: 110 * 1000000 + 0 * 100 + 2)
        return mod
    sys.modules["torch.utils.cpp_extension"].load_inline = load_inline


def main():
    cmd.setup()
    check_v11b.mock_v10({})
    calls, built = {}, {}
    mock_f12k(calls, built)
    torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(multi_processor_count=132)
    w = d.weights()
    lab = d.load(d.PROJ / "lab_e42.py", "lab_e42_cpu")
    base_choice = dict(lab.CHOICE)

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        cells = [cl.cell_contents for cl in fn.__closure__ or ()]
        c = next((v for v in cells if isinstance(v, dict) and "kind" in v), None)
        if c is None or c["kind"] != "f12k":
            return 20.0
        return 3.0 if (c["NT"], c["KSW"], c["D"]) == DIRTY[0] and c["G"] == DIRTY[1] else 4.0 + 0.01 * c["G"] / 132
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = lab.prepare(w)
    out = buf.getvalue()
    assert built["insts"] == set(lab.F12K_LAB_INSTANCES), built
    for key in ("f12k build ok in", "f12k scratch:", "f12k counters: ", "DIRTY", "chosen config left its counters dirty"):
        assert key in out, (key, out[-3000:])
    targets = lab.f12k_targets(132)
    new = {k: c for k, c in lab.CHOICE.items() if base_choice.get(k) != c}
    assert set(new) == set(targets) - {("qkv_gate", 16)}, (sorted(new), out[-2500:])
    assert lab.CHOICE[("qkv_gate", 16)] == base_choice[("qkv_gate", 16)], lab.CHOICE[("qkv_gate", 16)]
    assert all(c["kind"] == "f12k" and c["fallback"] == base_choice.get(k) for k, c in new.items()), new
    assert not (new[("unpooler", 8)]["NT"], new[("unpooler", 8)]["KSW"], new[("unpooler", 8)]["D"], new[("unpooler", 8)]["G"]) \
        == WRONG[0] + (WRONG[1],), new[("unpooler", 8)]
    sk = [ln for ln in out.splitlines() if "SK CHOICE" in ln]
    assert len(sk) == len(new), sk
    lines = [ln for ln in out.splitlines() if ln.startswith("[lab] f12k ") and "(sglang" in ln]
    wrong = [ln for ln in lines if " 1 wrong" in ln]   # the planted config runs on both unpooler-slice shapes
    assert len(lines) == 5 and len(wrong) == 2 and all("unpooler m" in ln for ln in wrong), lines
    sys.path.insert(0, str(d.PROJ))
    import reference
    before = dict(calls)
    for (layer, m) in sorted(lab.CHOICE):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(m, layer, 4800 + m, dist=dist, device="cpu")
            ok, msg = reference.check_implementation(data, lab.kernel_fn(data, state), w)
            assert ok, (layer, m, dist, msg)
    ran = {k for k in calls if calls[k] > before.get(k, 0)}
    assert {k[2] for k in ran} == {2304, 1024} and len(ran) >= 3, ran
    print(f"ok: lab_e42.py dry run; {len(built['insts'])} F12K instances in one worker-thread module; planted wrong config "
          f"rejected, planted dirty-counter winner put back to v11b's entry; {len(new)} shapes moved with fallbacks; judge's "
          f"check passes on all {len(lab.CHOICE)} shapes x 2 inputs (F12K ran on {len(ran)} configs)")


if __name__ == "__main__":
    main()

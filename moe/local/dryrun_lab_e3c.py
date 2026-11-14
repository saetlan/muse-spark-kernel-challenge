"""CPU dry run of lab_e3c.prepare(): every lab code path except nvcc and the CUDA launches, which are
replaced by FP32 stand-ins (one instance made to fail on purpose), and time_graph by a fake timer.
Catches Python mistakes in the lab before they cost a check. Uses real dims and 32 experts.

Run: .venv/bin/python moe/local/dryrun_lab_e3c.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_v0_lab as base  # noqa: E402


def load_cpu_lab():
    src = (base.MOE / "lab_e3c.py").read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    path = Path("/tmp") / "lab_e3c_cpu.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("lab_e3c_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    check_m2.install_stubs()
    ext_mod = types.ModuleType("torch.utils.cpp_extension")
    ext_mod.load_inline = None
    sys.modules["torch.utils.cpp_extension"] = ext_mod
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=132)
    lab = load_cpu_lab()
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(1)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=gen, device="cpu", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=gen, device="cpu", leading=(32,))
    weights = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
               "w2_gs": gs2.expand(32).contiguous(),
               "post_norm_w": (0.1 * torch.randn((4096,), generator=gen)).to(torch.bfloat16)}
    ref = lab._Ref(weights)
    calls = {"g1": 0, "g2": 0, "pn": 0, "run": 0, "v1": 0, "warm": 0}

    class Ext:
        def m11_attrs(self, gated, tt, nt, d, ks):
            return torch.tensor([168, 0, 2432])

    def fail(cfg):
        if cfg["nt"] == 4 and cfg["d"] == 2 and cfg["c"] == 4:
            raise RuntimeError("planted launch failure")

    def gemm1(x, ids, state, cfg):
        fail(cfg)
        calls["g1"] += 1
        return ref.act(x, ids).to(torch.bfloat16)

    def gemm2(act, ids, state, cfg):
        fail(cfg)
        calls["g2"] += 1
        return act  # stands in for y; post_norm below ignores it

    def post_norm(y, ids, tw, state, tokens):
        calls["pn"] += 1
        return ref.stage2(y, ids.view(tokens, 8), tw).to(torch.bfloat16)

    def run(x, ids, w, state, cfg):
        calls["run"] += 1
        return ref.full(x, ids, w).to(torch.bfloat16)

    def v1_kernel_fn(data, state):
        calls["v1"] += 1
        return ref.full(data["hidden_states"], data["topk_ids"], data["topk_weights"]).to(torch.bfloat16)

    def warm(state, cfg):
        assert set(cfg) == {"s1", "s2"} and set(cfg["s1"]) == {"nt", "d", "ks", "c"} and \
            set(cfg["s2"]) == {"tt", "nt", "d", "ks", "c"}, cfg
        calls["warm"] += 1

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        return 30.0 + (hash(fn.__code__.co_firstlineno) % 7)

    def submission_prepare(w):
        return {"norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16), "m11_configs": ()}

    lab.m11_build = lambda: Ext()
    lab.m11_gemm1, lab.m11_gemm2, lab.m11_post_norm, lab.m11_run = gemm1, gemm2, post_norm, run
    lab._v1_kernel_fn, lab.m11_warm, lab._submission_prepare = v1_kernel_fn, warm, submission_prepare
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(weights)
    cfgs = state["m11_configs"]
    assert [c[0] for c in cfgs] == [5, 9, 17, 33, 192, 321], cfgs
    assert state["m11"]["w13"].shape == (32, 512, 64, 32, 4) and state["m11"]["w2_s"].shape == (32, 256, 64, 32)
    assert calls["g1"] > 100 and calls["g2"] > 100 and calls["run"] > 8 and calls["v1"] == 6 * 3, calls
    print(f"ok: lab_e3c.prepare dry run on the CPU; calls {calls}; m11_configs classes {[c[0] for c in cfgs]}")


if __name__ == "__main__":
    main()

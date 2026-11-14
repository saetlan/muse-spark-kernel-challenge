"""CPU dry run of lab_e2.prepare(): every lab code path except the Triton launches, which are
replaced by FP32 stand-ins (one variant made to fail on purpose), and time_graph by a fake timer.
Catches Python mistakes in the lab before they cost a check. Uses real dims and 32 experts.

Run: .venv/bin/python moe/local/dryrun_lab_e2.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_v0_lab as base  # noqa: E402


def load_cpu_lab():
    src = (base.MOE / "lab_e2.py").read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    path = Path("/tmp") / "lab_e2_cpu.py"
    path.write_text(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("lab_e2_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    check_m2.install_stubs()
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    lab = load_cpu_lab()
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(1)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=gen, device="cpu", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=gen, device="cpu", leading=(32,))
    weights = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
               "w2_gs": gs2.expand(32).contiguous(),
               "post_norm_w": (0.1 * torch.randn((4096,), generator=gen)).to(torch.bfloat16)}
    ref = lab._Ref(weights)
    calls = {"stage1": 0, "stage2": 0, "run": 0, "v1": 0, "warm": 0}

    def fail_if_scale2d(cfg):
        if cfg.get("scale2d") and not cfg["swap"]:
            raise RuntimeError("planted compile failure")

    def stage1(x, ids, bank, cfg, sms):
        fail_if_scale2d(cfg)
        calls["stage1"] += 1
        return ref.act(x, ids).to(torch.bfloat16)

    def stage2(act, ids, tw, norm_w, bank, cfg, sms, tokens):
        fail_if_scale2d(cfg)
        calls["stage2"] += 1
        return ref.stage2(act, ids.view(tokens, 8), tw).to(torch.bfloat16)

    def run(x, ids, w, state, cfg1, cfg2):
        calls["run"] += 1
        return ref.full(x, ids, w).to(torch.bfloat16)

    def gemm(*a, **k):
        return types.SimpleNamespace(asm={"ttgir": "tt.dot  ttg.convert_layout", "ptx": "mma.sync"}, n_regs=128,
                                     n_spills=0, metadata=types.SimpleNamespace(shared=1024))

    def v1_kernel_fn(data, state):
        calls["v1"] += 1
        return ref.full(data["hidden_states"], data["topk_ids"], data["topk_weights"]).to(torch.bfloat16)

    def warm(state, cfg1, cfg2):
        calls["warm"] += 1

    def fake_time(self, fn, reps=20, cold=True):
        fn()
        return 50.0 + (hash(fn.__code__.co_firstlineno) % 7)

    def submission_prepare(w):
        state = {"sms": 132, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
        state["m2"] = lab.m2_bank(w, 0, 32, (256,))
        state["m2_configs"] = ()
        return state

    lab.m2_stage1, lab.m2_stage2, lab.m2_run, lab.m2_gemm = stage1, stage2, run, gemm
    lab._v1_kernel_fn, lab.m2_warm, lab._submission_prepare = v1_kernel_fn, warm, submission_prepare
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(weights)
    cfgs = state["m2_configs"]
    assert len(cfgs) == 4 and [c[0] for c in cfgs] == [5, 13, 25, 33], cfgs
    assert calls["stage1"] > 10 and calls["stage2"] > 10 and calls["run"] > 8 and calls["v1"] == 4, calls
    print(f"ok: lab_e2.prepare dry run on the CPU; calls {calls}; m2_configs classes {[c[0] for c in cfgs]}")


if __name__ == "__main__":
    main()

"""CPU dry run of lab_e6.prepare() (M14 at 32-320 tokens): the real M13 Python glue (m13_setup, m13_bank, m13_gemm1/2,
m13_post_norm, m13_run, the lab) with nvcc and the CUDA launches replaced by FP32 stand-ins that check
the C++ argument lists and shapes; one M13 config made to fail and one to leave its scratch dirty
(the lab must drop both); time_graph by a fake timer under which M13 wins, so the chosen-config and
warm-up paths run too. Real dims and 32 experts for the references; M11's bank is faked small.

Run: .venv/bin/python moe/local/dryrun_lab_e6.py
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
    src = (base.MOE / "lab_e6.py").read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    path = Path("/tmp") / "lab_e6_cpu.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("lab_e6_cpu", path)
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
    calls = {"m13_g1": 0, "m13_g2": 0, "m11": 0, "post": 0, "v1": 0, "m13_warm": 0, "m11_warm": 0, "timed_m13": 0}
    state_box = {}

    def local_pairs(ids):
        flat = ids.reshape(-1).long()
        return flat, [e for e in torch.unique(flat).tolist() if e < 32]

    class Ext:
        def m13_attrs(self, gated, tt, nt):
            return torch.tensor([128 if nt == 1 else 168, 0, 7000, 4 if nt == 1 else 3])

        def m14_setup(self):
            return torch.tensor([[g, tt, nt, 160, 0, 21000, 3 * 8 * nt * 128, 3 if nt == 8 else 4]
                                 for g, tt, nt in lab.M14_INSTANCES])

        def m14_gemm(self, *args):
            return self.m13_gemm(*args, kind="m14")

        def m13_gemm(self, *args, kind="m13"):
            assert len(args) == 19, len(args)
            (w, sf, c, ids, a, out, scr, ctr, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated, tt, nt,
             grid, pdl, pf) = args
            assert w.dtype == torch.int32 and w.shape[2:] == (64, 32, 4) and sf.dtype == torch.int16 and sf.shape[2:] == (16, 32, 4)
            limit = lab.M13_MAX_TOKENS if kind == "m13" else lab.M14_MAX_TOKENS
            assert ids.dtype == torch.int32 and ids.dim() == 1 and n_pairs == ids.numel() <= 8 * limit, (kind, n_pairs)
            assert out.shape == (n_pairs, out_cols) and out.dtype == torch.bfloat16
            assert scr.shape[1] == 16 * rt_total and ctr.dtype == torch.int32
            occ = state_box["s"]["m13_occ"].get((gated, 2 if gated else tt, nt), 4) if "m13_occ" in state_box["s"] else 4
            assert grid % 132 == 0 and 1 <= grid // 132 <= occ, (grid, occ)
            assert pdl in (0, 1) and pf >= 0 and (gated == 0 or pdl == 0)
            if kind == "m14" and nt == 4 and grid == 2 * 132 and gated:
                raise RuntimeError("planted launch failure")
            flat, experts = local_pairs(ids)
            if gated:
                assert a_per_pair == 0 and rt_total == 512
                calls["m13_g1"] += 1
                out.copy_(torch.zeros_like(out))
                act = ref.act(a, ids.view(-1, 8))
                rows = (flat < 32).nonzero().flatten()
                out[rows] = act[rows].to(torch.bfloat16)
            else:
                assert a_per_pair == 1 and rt_total == 256
                calls["m13_g2"] += 1
                for e in experts:
                    p = (flat == e).nonzero().flatten()
                    out[p] = (a[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
                if kind == "m14" and tt == 2 and nt == 8 and grid == 132:
                    scr[0, 0] = 1.0  # planted: leaves the scratch dirty
            return out

        def m11_gemm(self, *args):
            assert len(args) == 17, len(args)
            w, sf, c, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated = args[:12]
            calls["m11"] += 1
            flat, experts = local_pairs(ids)
            if gated:
                act = ref.act(a, ids.view(-1, 8))
                rows = (flat < 32).nonzero().flatten()
                out[rows] = act[rows].to(torch.bfloat16)
            else:
                for e in experts:
                    p = (flat == e).nonzero().flatten()
                    out[p] = (a[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
            return out

        def m13_post_norm(self, y, nw, tw, ids, out, tokens, local_start, eps, pdl):
            calls["post"] += 1
            assert out.shape == (tokens, 4096) and pdl in (0, 1) and nw.dtype == torch.bfloat16
            flat = ids.reshape(-1).long()
            res = torch.zeros((tokens, 4096))
            for p in range(flat.numel()):
                e = int(flat[p]) - local_start
                if 0 <= e < 32:
                    v = y[p].float() * nw.float()
                    v = v * torch.rsqrt(v.pow(2).mean() + eps)
                    res[p // 8] += tw.reshape(-1)[p] * v
            out.copy_(res.to(torch.bfloat16))
            return out

    def fake_bank(weights, start, count):
        bank = {"start": start, "count": count, "inter": 4096, "hidden": 4096,
                "c13": torch.ones(count), "c2": torch.ones(count), "rt_w13": 512, "rt_w2": 256}
        for name, rt in (("w13", 512), ("w2", 256)):
            bank[name] = torch.zeros((count, 2, 64, 32, 4), dtype=torch.int32)  # small stand-ins (2 tiles)
            bank[name + "_s"] = torch.zeros((count, 2, 64, 32), dtype=torch.int16)
        return bank

    def v1_kernel_fn(data, state):
        calls["v1"] += 1
        return ref.full(data["hidden_states"], data["topk_ids"], data["topk_weights"]).to(torch.bfloat16)

    real_m13_warm, real_m11_warm = lab.m13_warm, lab.m11_warm

    def m13_warm(state, cfg, tokens=8):
        calls["m13_warm"] += 1
        assert cfg["s1"]["kind"] in ("m13", "m14") and cfg["s2"]["kind"] in ("m13", "m14") and cfg["post"]["kind"] == "m13"
        assert {"pdl", "pf"} <= set(cfg["s2"]), cfg
        return real_m13_warm(state, cfg, tokens)

    def m11_warm(state, cfg):
        calls["m11_warm"] += 1
        return real_m11_warm(state, cfg)

    def fake_time(self, fn, reps=20, cold=True):
        before = calls["m13_g1"] + calls["m13_g2"]
        fn()
        used = calls["m13_g1"] + calls["m13_g2"] > before
        calls["timed_m13"] += used
        return (20.0 if used else 40.0) + (hash(fn.__code__.co_firstlineno) % 3)

    def submission_prepare(w):
        # the M13 file's prepare() without v1's Marlin repack (needs the GPU): its own setup and warm-ups
        state = {"norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16),
                 "m11_configs": lab.M11_CONFIGS, "m13_configs": lab.M13_CONFIGS}
        state_box["s"] = state
        lab.m13_setup(state, w)
        for _, cfg in lab.M11_CONFIGS:
            if cfg is not None:
                lab.m11_warm(state, cfg)
        return state

    def m11_post_norm(y, ids, tw, state, tokens):
        out = torch.empty((tokens, 4096), dtype=torch.bfloat16)
        return Ext().m13_post_norm(y, state["norm_w"], tw, ids, out, tokens, 0, lab.POST_NORM_EPS, 0)

    lab.m11_post_norm = m11_post_norm
    lab.m13_build = lambda: Ext()
    lab.m11_bank = fake_bank
    lab._v1_kernel_fn, lab._submission_prepare = v1_kernel_fn, submission_prepare
    lab.m13_warm, lab.m11_warm = m13_warm, m11_warm
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(weights)
    cfgs = state["m13_configs"]
    assert [c[0] for c in cfgs] == [5, 9, 17, 33, 65, 192, 321], cfgs
    for below, cfg in cfgs:
        if below in (33, 65, 192, 321):
            assert cfg is not None and cfg["s1"]["kind"] == "m14", (below, cfg)
            assert not (cfg["s1"]["nt"] == 4 and cfg["s1"]["c"] == 2), "the failing config was chosen"
            assert not (cfg["s2"]["tt"] == 2 and cfg["s2"]["nt"] == 8 and cfg["s2"]["c"] == 1), "the dirty config was chosen"
    assert state["m13"]["w13"].shape == (32, 2, 64, 32, 4) and state["m13"]["w13_s"].shape == (32, 2, 16, 32, 4)
    assert state["m13"]["scr13"].shape == (8 * lab.M14_MAX_TOKENS, 16 * 512)
    assert calls["m13_warm"] == 4 and calls["v1"] == 4 * 3 and calls["m11_warm"] == 4, calls
    # kernel_fn dispatch with the lab's configs: M14 at 128 tokens
    x, ids, w = lab._case(128, 5)
    data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w, "config": {"bs": 128, "local_start": 0, "local_count": 32}}
    before = calls["m13_g1"]
    out = lab.kernel_fn(data, state)
    assert calls["m13_g1"] == before + 1 and out.shape == (128, 4096)
    err, row = lab._errors(out, ref.full(x, ids, w))
    assert err < 1e-2, err
    print(f"ok: lab_e6.prepare dry run on the CPU; calls {calls}; m13_configs classes {[c[0] for c in cfgs]}; "
          f"kernel_fn bs128 through M14 err {err:.1e}")


if __name__ == "__main__":
    main()

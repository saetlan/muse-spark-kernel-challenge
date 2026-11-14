"""CPU dry run of lab_re1.py's lab sections (not the GPU kernels): GPU modules stubbed, 'cuda' mapped to
the CPU, tiny sizes, _experts and graph timing faked, torch._grouped_mm replaced by a loop. Catches
Python-level bugs (names, shapes, closures, budget logic) before a check is spent.

Run: .venv/bin/python prefill_layer/local/dryrun_lab_re1.py
"""

import sys
import types
from pathlib import Path

import torch

DIR = Path(__file__).resolve().parents[1]


class _Any(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})


for mod in ("sgl_kernel", "sgl_kernel.flash_attn", "sgl_kernel.scalar_type", "sglang", "sglang.kernels",
            "sglang.kernels.ops", "sglang.kernels.ops.activation", "sglang.kernels.ops.activation.activation",
            "sglang.kernels.ops.kvcache", "sglang.kernels.ops.kvcache.kvcache", "sglang.kernels.ops.moe",
            "sglang.kernels.ops.moe.moe_wna16_marlin", "sglang.kernels.ops.muse_spark_v12",
            "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing",
            "sglang.kernels.ops.muse_spark_v12.fused_norm_gate",
            "sglang.kernels.ops.muse_spark_v12.fused_norm_residual",
            "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
            "sglang.kernels.ops.muse_spark_v12.fused_qk_norm", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.moe", "sglang.srt.layers.moe.fused_moe_triton", "sglang.srt.layers.muse_spark_layers",
            "sglang.srt.layers.quantization", "sglang.srt.layers.quantization.marlin_utils",
            "sglang.srt.layers.quantization.marlin_utils_fp4", "sglang.srt.layers.quantization.unquant"):
    sys.modules[mod] = _Any(mod)

src = (DIR / "lab_re1.py").read_text().replace('"cuda"', '"cpu"')
ns = {"__name__": "lab_re1"}
exec(compile(src, "lab_re1.py", "exec"), ns)

torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.cuda.get_device_name = lambda *a, **k: "cpu"
torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=0)


def fake_grouped_mm(a, b, offs, out_dtype=None):
    out, start = [], 0
    for g, end in enumerate(offs.tolist()):
        out.append(a[start:end].float() @ b[g].float())
        start = end
    return torch.cat(out).to(out_dtype or a.dtype)


torch._grouped_mm = fake_grouped_mm
ns.update(HIDDEN=64, D_EXPERT=32, RE1_TOKENS=(40, 75))
calls = []
ns["_experts"] = lambda x, w, ids, state, block_size_m=None: (calls.append(block_size_m), x * w.sum())[1]
ns["Lab"].time_graph = lambda self, fn, reps=20, cold=True: (fn(), 1.0)[1]
ns["_submission_prepare"] = lambda weights: {"bank": None}
weights = {"router": torch.randn(256, 64), "router_bias": torch.randn(256) * 0.01}
lines = []
ns["log"] = lambda msg: lines.append(msg)
state = ns["prepare"](weights)
text = "\n".join(lines)
print(text)
assert "FAILED" not in text, "a lab section failed in the dry run"
assert state == {"bank": None}
assert sorted(set(c for c in calls if c)) == [8, 16, 32, 48, 64], calls
for t in (40, 75):
    assert f"marlin T={t} real" in text and f"marlin T={t} synth" in text
    for pad in (1, 16, 128):
        assert f"grouped_mm T={t} pad{pad}:" in text
print("dryrun_lab_re1: ok")

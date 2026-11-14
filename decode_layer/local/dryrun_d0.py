"""CPU dry run of lab_d0.prepare(): every lab code path runs with SGLang, FA3 and Triton replaced
by shape-correct stand-ins and time_graph by a fake timer. Catches Python mistakes (names, shapes,
dtypes, control flow, budget handling) before they cost a check. Real layer dims; contexts cut
to 8,000 tokens to keep it fast. One stand-in fails on purpose (tiny_gemm) to exercise the
failure path.

Run: .venv/bin/python decode_layer/local/dryrun_d0.py [decode_layer/lab_d0.py]
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

DECODE = Path(__file__).resolve().parents[1]
CALLS = {}


def hit(name):
    CALLS[name] = CALLS.get(name, 0) + 1


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    return m


class _Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        name = self.fn.__name__

        def launch(*args, **kw):
            hit(name)
            if name == "_post_expert_norm_weighted_sum_kernel":
                y, w, tw, out = args[:4]
                out.copy_((y.float() * tw[..., None]).sum(1).to(out.dtype))
            elif name == "_lab_skinny_gemm":
                x, w, out = args[:3]
                res = (x.float() @ w.float().T)
                if kw["OUT_BF16"]:
                    out.copy_(res.to(out.dtype))
                else:
                    out.zero_()
                    out[0].copy_(res)
            elif name == "_lab_split_reduce":
                part, out = args[:2]
                out.view(-1).copy_(part.sum(0).view(-1).to(out.dtype))
            else:
                raise RuntimeError(f"unexpected kernel {name}")
        return launch


def install_stubs():
    tl = mod("triton.language", constexpr=lambda x=None: x, float32="f32", bfloat16="bf16")
    mod("triton", jit=lambda fn=None, **_: _Kernel(fn) if fn else _Kernel, language=tl,
        cdiv=lambda a, b: -(-a // b), next_power_of_2=lambda n: 1 << (n - 1).bit_length())

    def fa3(q=None, k_cache=None, v_cache=None, num_splits=0, pack_gqa=None, scheduler_metadata=None, **kw):
        hit("fa3")
        assert kw["window_size"] in ((2047, 0), (-1, -1))
        return torch.zeros(q.shape, dtype=torch.bfloat16) + num_splits * 1e-3

    mod("sgl_kernel.flash_attn", flash_attn_with_kvcache=fa3,
        get_scheduler_metadata=lambda **kw: torch.zeros(8, dtype=torch.int32))
    mod("sgl_kernel.scalar_type", scalar_types=types.SimpleNamespace(float4_e2m1f="fp4"))
    mod("sglang.kernels.ops.activation.activation", silu_and_mul=lambda x, out, **k: hit("silu"))

    def store_cache(k, v, kc, vc, loc, reserved_skip_index):
        hit("store")
        assert k.dtype == torch.uint8 and kc.shape[1] == 128 and loc.dtype == torch.int64
        kc[loc] = k

    mod("sglang.kernels.ops.kvcache.kvcache", store_cache=store_cache)

    def marlin(a, c, *args, size_m, size_n, size_k, top_k, moe_block_size, **kw):
        hit(f"marlin bsm{moe_block_size}")
        assert a.shape == (size_m, size_k) and c.shape == (size_m * top_k, size_n), (a.shape, c.shape)
        return c.zero_()

    mod("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=marlin)

    class Routing:
        def __init__(self, expert_bias, **kw):
            self.bias = expert_bias

        def __call__(self, _, logits, k, renorm):
            s = torch.sigmoid(logits * 0.1767766952966369)
            ids = (s + self.bias).topk(k, dim=-1).indices
            w = s.gather(1, ids)
            return w / w.sum(-1, keepdim=True), ids.to(torch.int32)

    mod("sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing", FusedMuseSparkRouting=Routing)

    class NormGate:
        def __init__(self, *a):
            pass

        def __call__(self, x, gate):
            assert x.dtype == torch.bfloat16 and x.shape == gate.shape
            return x

    mod("sglang.kernels.ops.muse_spark_v12.fused_norm_gate", FusedRMSNormGate=NormGate)

    class Boundary:
        def __init__(self, post, res, pre):
            pass

        def __call__(self, stream, branch, branch_dtype, router_dtype=None):
            assert stream.dtype == torch.float32 and branch.shape == stream.shape, (stream.shape, branch.shape)
            s = stream + branch.float()
            return s.to(torch.bfloat16), s, (s if router_dtype is not None else None)

    mod("sglang.kernels.ops.muse_spark_v12.fused_norm_residual", FusedPostNormResidualPreNorm=Boundary)

    def old_post(y, tw, norm_weight, eps, weight_first, slot_expert_ids):
        hit("sglang post")
        return (y.float() * tw[..., None]).sum(1).to(y.dtype)

    mod("sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
        _post_expert_norm_weighted_sum_kernel=_Kernel(type("k", (), {"__name__": "_post_expert_norm_weighted_sum_kernel"})),
        fused_post_expert_norm_weighted_sum=old_post)
    sys.modules["sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton"] \
        ._post_expert_norm_weighted_sum_kernel.fn.__name__ = "_post_expert_norm_weighted_sum_kernel"

    class QK:
        def __init__(self, **kw):
            pass

        def __call__(self, qkv, positions=None, cos_sin=None):
            assert qkv.shape[1] == 2304
            return qkv[:, :1024].contiguous(), qkv[:, 1024:1152].contiguous()

    mod("sglang.kernels.ops.muse_spark_v12.fused_qk_norm", FusedQKNorm=QK)

    def align(ids, bsm, e):
        hit("align")
        assert ids.dtype == torch.int32
        return torch.zeros(64, dtype=torch.int32), torch.zeros(8, dtype=torch.int32), torch.tensor([64])

    mod("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=align)

    class Norm(torch.nn.Module):
        def __init__(self, size, eps=1e-5, has_weight=True, zero_centered_gamma=True, **kw):
            super().__init__()
            self.weight = None

        def precompute_effective_weight(self):
            pass

        def _get_effective_weight(self):
            return self.weight.float() + 1

        def forward(self, x):
            return x

    class Residual(torch.nn.Module):
        def __init__(self, h, temperature):
            super().__init__()

        def precompute_coefficients(self):
            pass

    mod("sglang.srt.layers.muse_spark_layers", MuseSpark1xNormPreservingResidualConnection=Residual,
        MuseSpark1xRMSNorm=Norm, MuseSpark1xRMSNormWithInputScale=Norm)
    mod("sglang.srt.layers.quantization.marlin_utils", marlin_make_workspace=lambda d, max_blocks_per_sm: torch.zeros(4))
    mod("sglang.srt.layers.quantization.marlin_utils_fp4", prepare_moe_nvfp4_layer_for_marlin=lambda layer: None)
    mod("sglang.srt.layers.quantization.unquant", _bf16_gemm_dispatch_impl=lambda x, w, b: F.linear(x, w))

    def gemv(x, w):
        hit("hopper_gemv")
        return F.linear(x, w)

    mod("sglang.kernels.ops.gemm.hopper_bf16_gemv", hopper_bf16_gemv=gemv,
        use_hopper_bf16_gemv=lambda m, n, k: m == 1 and n < 12288)

    def tiny(x, w):
        raise RuntimeError("planted tiny_gemm failure")

    mod("sglang.kernels.ops.gemm.tiny_gemm", tiny_gemm_bf16=tiny, can_use_tiny_gemm=lambda n, k, m: True)


def patch_torch():
    class Stream:
        def __init__(self, *a, **k):
            pass

        def wait_stream(self, other):
            pass

    import contextlib
    torch.cuda.Stream = Stream
    torch.cuda.current_stream = lambda *a: Stream()
    torch.cuda.stream = lambda s: contextlib.nullcontext()
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.mem_get_info = lambda *a: (100 << 30, 141 << 30)
    torch.cuda.get_device_properties = lambda *a: types.SimpleNamespace(name="stub H200", multi_processor_count=132)
    torch.Tensor.record_stream = lambda self, s: None
    real_mm = torch.mm

    def mm(a, b, out_dtype=None):
        return real_mm(a.float(), b.float()) if out_dtype is not None else real_mm(a, b)

    torch.mm = mm


def load_lab(path):
    src = path.read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    src = src.replace("LAB_CTX = 80000", "LAB_CTX = 8000")
    out = Path("/tmp") / "lab_d0_cpu.py"
    out.write_text(src)
    spec = importlib.util.spec_from_file_location("lab_d0_cpu", out)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def weights():
    g = torch.Generator().manual_seed(0)

    def dense(n, k):
        return (torch.randn((n, k), generator=g) / k ** 0.5).to(torch.bfloat16)

    def vec(n):
        return (0.1 * torch.randn((n,), generator=g)).to(torch.bfloat16)

    return {
        "qkv_gate": dense(2304, 8192), "o": dense(8192, 1024), "attn_residual_gate": vec(8192),
        "pre_ffn_norm": vec(8192), "router": 0.045 * torch.randn((256, 8192), generator=g),
        "router_bias": 0.05 * torch.randn((256,), generator=g), "pooler": dense(4096, 8192),
        "pre_expert_norm": vec(4096), "w13": torch.zeros((32, 8192, 2048), dtype=torch.uint8),
        "w13_sf": torch.zeros((32, 8192, 256), dtype=torch.uint8), "w13_gs": torch.ones(32),
        "w2": torch.zeros((32, 4096, 2048), dtype=torch.uint8),
        "w2_sf": torch.zeros((32, 4096, 256), dtype=torch.uint8), "w2_gs": torch.ones(32),
        "post_expert_norm": vec(4096), "unpooler": dense(8192, 4096), "post_ffn_norm": vec(8192),
        "ffn_residual_gate": vec(8192), "next_pre_attn_norm": vec(8192),
    }


def main():
    install_stubs()
    patch_torch()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_d0.py"
    lab = load_lab(path)
    lab.ROPE_POSITIONS = 1 << 10  # the cos/sin table; values unused by the stubs
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

    calls_t = [0]

    def fake_time(self, fn, reps=15, cold=True):
        fn()
        calls_t[0] += 1
        return 10.0 if calls_t[0] % 5 == 1 else 10.0 + calls_t[0] % 7  # some times equal the fixed cost

    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    state = lab.prepare(weights())
    assert isinstance(state, dict) and "bank" in state
    text = "\n".join(lines)
    fails = [ln for ln in lines if "FAILED" in ln]
    print(f"\n--- {len(lines)} lab lines, {len(fails)} FAILED lines; calls {CALLS}")
    expected_fail = [ln for ln in fails if "planted tiny_gemm failure" in ln]
    unexpected = [ln for ln in fails if ln not in expected_fail]
    for ln in unexpected:
        print("UNEXPECTED:", ln)
    checks = {
        "all 12 shapes in ops": sum(f"ops {n[0]}:" in text for n in lab.SCORED) == 12,
        "all 12 shapes in fa3": sum(f"fa3 {n[0]} " in text for n in lab.SCORED) == 12,
        "gemm covers 6 weights": all(f"gemm {w} [" in text for w in ("qkv", "o", "pooler", "unpool slice",
                                                                       "unpooler", "router hi")),
        "router fp32 line": "gemm router fp32 T=8" in text,
        "SGLang heights used": any(k.startswith("marlin bsm") and k != "marlin bsm8" for k in CALLS),
        "fused post-norm used": CALLS.get("_post_expert_norm_weighted_sum_kernel", 0) > 0,
        "sglang post-norm used": CALLS.get("sglang post", 0) > 0,
        "triton gemm used": CALLS.get("_lab_skinny_gemm", 0) > 0 and CALLS.get("_lab_split_reduce", 0) > 0,
        "planted failure caught": bool(expected_fail),
        "no unexpected failures": not unexpected,
        "summary line": "ops summary" in text,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run: all checks passed")


if __name__ == "__main__":
    main()

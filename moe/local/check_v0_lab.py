"""Local (CPU) checks for submission.py (v1) and the generated lab files: they import with the
GPU-only libraries stubbed out, and the parts that don't need a GPU behave as intended.

Run: .venv/bin/python moe/local/check_v0_lab.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))
PROBLEM_LINE = "# okc: problem=muse_moe_w4a16"


def stub_modules():
    """Empty stand-ins for sglang, sgl_kernel and triton so the files import on a Mac."""
    def make(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    for name in ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.activation",
                 "sglang.kernels.ops.moe", "sglang.kernels.ops.muse_spark_v12", "sglang.srt",
                 "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.quantization",
                 "sgl_kernel"):
        make(name)
    make("sgl_kernel.scalar_type", scalar_types=types.SimpleNamespace(float4_e2m1f=None))
    make("sglang.kernels.ops.activation.activation", silu_and_mul=None)
    make("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=None)
    make("sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
         fused_post_expert_norm_weighted_sum=None, _post_expert_norm_weighted_sum_kernel=None)
    make("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=None)
    make("sglang.srt.layers.quantization.marlin_utils", marlin_make_workspace=None)
    make("sglang.srt.layers.quantization.marlin_utils_fp4", prepare_moe_nvfp4_layer_for_marlin=None)

    class _Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            raise RuntimeError("Triton kernels can't run locally")

    tl = make("triton.language", constexpr=int, int32="int32", int64="int64", float32="float32")
    make("triton", jit=_Kernel, cdiv=lambda a, b: -(-a // b), language=tl, __version__="stub",
         next_power_of_2=lambda n: 1 << (n - 1).bit_length())


def load(name):
    spec = importlib.util.spec_from_file_location(name, MOE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_first_lines(names):
    for name in names:
        first = (MOE / f"{name}.py").read_text().splitlines()[0]
        assert first == PROBLEM_LINE, (name, first)
    print(f"ok: {', '.join(names)} start with the okc problem line")


def check_v1_block_sizes(v1):
    got = {bs: v1.block_size_m(bs) for bs in (1, 8, 16, 32, 128, 256, 4, 33, 257, 1000)}
    want = {1: 8, 8: 8, 16: 8, 32: 16, 128: 16, 256: 32, 4: 8, 33: 16, 257: 32, 1000: 32}
    assert got == want, got
    print(f"ok: v1 tile heights {got}")


def check_lab_wraps_submission(lab_name):
    """The lab's timed code must be the submission's, byte for byte, up to the lab sections."""
    sub = (MOE / "submission.py").read_text().replace("\ndef prepare(weights", "\ndef _submission_prepare(weights", 1)
    lab = (MOE / f"{lab_name}.py").read_text()
    assert lab.startswith(sub.rstrip()), f"{lab_name} doesn't start with submission.py"
    mod = load(lab_name)
    assert callable(mod.prepare) and callable(mod._submission_prepare) and callable(mod.kernel_fn)
    print(f"ok: {lab_name} = submission.py + lab sections; prepare wraps _submission_prepare")
    return mod


def check_routing(lab):
    for bs in (1, 8, 16, 32, 128, 256):
        ids, w = lab.synthetic_routing(bs, 1000 * bs + 11, device="cpu")
        assert ids.shape == (bs, 8) and ids.dtype == torch.int32 and int(ids.min()) >= 0 and int(ids.max()) < 256
        assert torch.allclose(w.sum(-1), torch.ones(bs))
        groups = torch.bincount((ids.long() // 32).flatten(), minlength=8)
        assert int(groups.argmax()) == 0, "experts 0-31 must be the busiest GPU"
    print("ok: synthetic routing is well formed, experts 0-31 busiest")


def emulate_decode_asm(words):
    """CPU emulation of the decode asm: int32 words -> int32 [n, 4] bf16x2 words."""
    q = words.to(torch.int64) & 0xFFFFFFFF
    outs = []
    for _ in range(4):
        outs.append((q & 0x80008000) | ((q & 0x70007000) >> 6))
        q = (q << 4) & 0xFFFFFFFF
    out = torch.stack(outs, dim=-1)
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


def check_decode_postprocessing(lab):
    from nvfp4 import random_weights, unpack_e2m1
    gen = torch.Generator().manual_seed(7)
    rows, k = 128, 4096  # full rows (checklist 9); every E4M3 scale meets every code
    packed, _, gs = random_weights(rows, k, generator=gen, device="cpu")
    sf = lab._all_e4m3_scales((rows, k // 16), "cpu")
    assert sf.view(torch.uint8).unique().numel() == 127
    words = packed.contiguous().view(torch.int32)
    n = words.numel()
    halves = lab._bf16_halves(emulate_decode_asm(words.reshape(-1)))
    order = torch.tensor([[3, 7], [2, 6], [1, 5], [0, 4]])
    vals = torch.empty((n, 8), dtype=torch.float32)
    vals[:, order.flatten()] = halves.reshape(n, 8)
    want = unpack_e2m1(packed).reshape(n, 8)
    assert torch.equal(vals * 2.0**126, want), "element order or unpacking is wrong"
    # The scale-multiply bookkeeping of sec_decode, with bf16 math on the CPU.
    shift = 64
    col = torch.arange(k) // 16
    a_words = vals.reshape(-1).to(torch.bfloat16).view(torch.int32)
    s_words = (sf.float() * 2.0**shift).to(torch.bfloat16)[:, col].reshape(-1).contiguous().view(torch.int32)
    a = lab._bf16_halves(a_words).reshape(-1).to(torch.bfloat16)
    s = lab._bf16_halves(s_words).reshape(-1).to(torch.bfloat16)
    got = (a * s).float() * 2.0 ** (126 - shift)
    assert torch.equal(got, want.reshape(-1) * sf.float()[:, col].reshape(-1))
    print("ok: decode post-processing and scale bookkeeping recover every value exactly, all 127 scales")


def check_headers_section(lab):
    """sec_headers on this machine: find_spec on missing, namespace and regular packages."""
    lab.sec_headers(None)
    print("ok: sec_headers runs (output above is this Mac's, not the judge's)")


if __name__ == "__main__":
    stub_modules()
    check_first_lines(["submission", "lab_e1b"])
    v1 = load("submission")
    check_v1_block_sizes(v1)
    lab = check_lab_wraps_submission("lab_e1b")
    check_routing(lab)
    check_decode_postprocessing(lab)
    check_headers_section(lab)

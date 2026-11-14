"""v1 must call SGLang's kernels exactly as baseline.py does, apart from its two intended changes:
the Marlin tile height, and the fused post-norm kernel below 32 tokens.

Runs prepare() and kernel_fn() of both files on the CPU with every SGLang op replaced by a
recorder. SGLang's real post-norm wrapper (from the serving checkout) runs with only its Triton
launch recorded. Each recorded argument is described by role (input, state entry or scratch
buffer), shape, dtype, strides and, for small integer tensors, contents.

Per shape, two runs:
  natural  baseline picks its own tile height and keeps SGLang's small-batch fallback; v1's
           height must be the measured table's, and baseline's must follow SGLang's rule
  aligned  baseline is forced to v1's height and its wrapper to the fused kernel at every
           token count; then both call logs must be identical, launch for launch

Run: .venv/bin/python moe/local/check_v1_vs_baseline.py [submission.py]
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

MOE = Path(__file__).resolve().parents[1]
ROOT = MOE.parent
POSTNORM_SRC = (ROOT / "serving/sglang/python/sglang/kernels/ops/muse_spark_v12/"
                "fused_post_expert_norm_weighted_sum_triton.py")
POSTNORM_MOD = "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton"
H = I = 4096
LOCAL = 32
# (bs, label): the six scored shapes, then the judge's five tests (TASK.md).
SHAPES = [(1, "decode 1"), (16, "decode 16"), (32, "decode 32"), (8, "verify 1"),
          (128, "verify 16"), (256, "verify 32"), (4, "test"), (33, "test"), (257, "test"),
          (128, "test skewed"), (1, "test")]
V1_HEIGHT = {1: 8, 4: 8, 8: 8, 16: 8, 32: 16, 33: 16, 128: 16, 256: 32, 257: 32}

LOG = []


def record(op, *args, **kwargs):
    LOG.append((op, args, kwargs))


class _Kernel:
    """Stands in for a @triton.jit function: kernel[grid](...) records the launch."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return lambda *args, **kwargs: record(f"launch {self.fn.__name__}", grid, *args, **kwargs)


def _align(topk_ids, block_size, num_experts):
    record("moe_align_block_size", topk_ids, block_size, num_experts)
    padded = topk_ids.numel() + num_experts * (block_size - 1)
    return (torch.zeros(padded, dtype=torch.int32), torch.zeros(-(-padded // block_size), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32))


def _gemm(*args, **kwargs):
    record("moe_wna16_marlin_gemm", *args, **kwargs)
    return args[1]  # Marlin writes into and returns its output buffer


def stub_modules():
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
    make("sgl_kernel.scalar_type", scalar_types=types.SimpleNamespace(float4_e2m1f="float4_e2m1f"))
    make("sglang.kernels.ops.activation.activation",
         silu_and_mul=lambda *a, **k: record("silu_and_mul", *a, **k))
    make("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=_gemm)
    make("sglang.kernels.ops.moe.moe_align_single_token",
         moe_align_single_token=lambda ids, block: _align(ids, block, LOCAL))
    make("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=_align)

    def workspace(device, max_blocks_per_sm=1):
        record("marlin_make_workspace", device, max_blocks_per_sm=max_blocks_per_sm)
        return torch.zeros(132 * max_blocks_per_sm, dtype=torch.int32)

    make("sglang.srt.layers.quantization.marlin_utils", marlin_make_workspace=workspace)
    make("sglang.srt.layers.quantization.marlin_utils_fp4",
         prepare_moe_nvfp4_layer_for_marlin=lambda layer: record("prepare_moe_nvfp4_layer_for_marlin", layer))
    tl = make("triton.language", constexpr=int, int32="int32", int64="int64", float32="float32")
    make("triton", jit=_Kernel, cdiv=lambda a, b: -(-a // b), language=tl, __version__="stub",
         next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    # SGLang's real post-norm wrapper, so baseline runs its true code path.
    load(POSTNORM_MOD, POSTNORM_SRC)
    # prepare() syncs once after its warm-up launch; there is no GPU here.
    torch.cuda.synchronize = lambda *a, **k: None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_weights():
    # torch.empty never touches the pages; the stubs only read shapes.
    return {
        "w13": torch.empty((256, 2 * I, H // 2), dtype=torch.uint8),
        "w13_sf": torch.empty((256, 2 * I, H // 16), dtype=torch.uint8).view(torch.float8_e4m3fn),
        "w13_gs": torch.ones(256),
        "w2": torch.empty((256, H, I // 2), dtype=torch.uint8),
        "w2_sf": torch.empty((256, H, I // 16), dtype=torch.uint8).view(torch.float8_e4m3fn),
        "w2_gs": torch.ones(256),
        "post_norm_w": (torch.randn(H, generator=torch.Generator().manual_seed(3)) * 0.1).to(torch.bfloat16),
    }


def make_data(bs, seed, skewed=False):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand((bs, 256), generator=gen)
    if skewed:
        scores[:, :LOCAL] += 0.5  # most routes local, as on the busiest GPU
    vals, ids = scores.topk(8, dim=-1)
    return {
        "hidden_states": torch.randn((bs, H), generator=gen).to(torch.bfloat16),
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (vals / vals.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": H, "d_expert": I, "n_experts": 256, "local_start": 0,
                   "local_count": LOCAL, "top_k": 8, "sharding": "ep8"},
    }


class Describer:
    """Names each tensor by role so logs from two different runs compare."""

    def __init__(self, named):
        self.named = {}
        for path, t in named:
            self.named.setdefault((t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype), path)
        self.storages = {}

    def __call__(self, v):
        if isinstance(v, torch.Tensor):
            key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), v.dtype)
            role = self.named.get(key)
            if role is None:
                sid = self.storages.setdefault(v.untyped_storage().data_ptr(), f"scratch{len(self.storages)}")
                role = f"{sid}+{v.storage_offset()}"
            desc = (role, tuple(v.shape), str(v.dtype), v.stride())
            if not v.is_floating_point() and v.numel() <= 1 << 16:
                desc += (tuple(v.flatten().tolist()),)
            return desc
        if isinstance(v, (list, tuple)):
            return type(v)(self(x) for x in v)
        if isinstance(v, dict):
            return {k: self(x) for k, x in v.items()}
        if isinstance(v, types.SimpleNamespace):
            return ("namespace", self(vars(v)))
        return v


def tensors_in(obj, path):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from tensors_in(v, f"{path}[{k!r}]")


def run(mod, data, state):
    LOG.clear()
    out = mod.kernel_fn(data, state)
    d = Describer([*tensors_in(data, "data"), *tensors_in(state, "state"), ("out", out)])
    return [(op, d(args), d(kwargs)) for op, args, kwargs in LOG], out


def logged_prepare(mod, weights):
    LOG.clear()
    state = mod.prepare(weights)
    d = Describer([*tensors_in(weights, "weights"), *tensors_in(state, "state")])
    return state, [(op, d(args), d(kwargs)) for op, args, kwargs in LOG]


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"call {i}: baseline {x[0]} vs v1 {y[0]}\n  baseline {x}\n  v1       {y}"
    return f"baseline made {len(a)} calls, v1 {len(b)}"


def sglang_height(bs):
    """MuseSpark1xFp4Experts' rule: the first candidate with bs * 8 / 32 / height < 0.9, else 64."""
    for height in (8, 16, 32, 48, 64):
        if bs * 8 / LOCAL / height < 0.9:
            return height
    return 64


def main():
    stub_modules()
    postnorm = sys.modules[POSTNORM_MOD]
    assert postnorm._MIN_TOKENS_FOR_FUSED == 32, "SGLang's fallback threshold changed; recheck F17"
    baseline = load("baseline", MOE / "baseline.py")
    v1 = load("submission", Path(sys.argv[1]) if len(sys.argv) > 1 else MOE / "submission.py")

    weights = make_weights()
    b_state, b_prep = logged_prepare(baseline, weights)
    v_state, v_prep = logged_prepare(v1, weights)
    warmups = [c for c in v_prep if c[0].startswith("launch")]
    assert [c for c in v_prep if not c[0].startswith("launch")] == b_prep, first_difference(b_prep, v_prep)
    assert len(warmups) == 1, warmups
    assert torch.equal(v_state["norm_w"], b_state["norm_w"]), "norm weight differs from SGLang's"
    print(f"ok: prepare() makes baseline's {len(b_prep)} calls plus one post-norm warm-up launch")

    candidates = baseline.BLOCK_SIZE_M_CANDIDATES
    for n, (bs, label) in enumerate(SHAPES):
        data = make_data(bs, 100 + n, skewed="skewed" in label)
        # Natural: each file's own choices.
        b_log, _ = run(baseline, data, b_state)
        v_log, out = run(v1, data, v_state)
        b_height = next(c[1][1] for c in b_log if c[0] == "moe_align_block_size")
        v_height = next(c[1][1] for c in v_log if c[0] == "moe_align_block_size")
        assert b_height == sglang_height(bs) and v_height == V1_HEIGHT[bs], (bs, b_height, v_height)
        b_fused = any(c[0].startswith("launch") for c in b_log)
        assert b_fused == (bs >= 32), (bs, b_fused)
        assert tuple(out.shape) == (bs, H) and out.dtype == torch.bfloat16, (out.shape, out.dtype)

        # Aligned: baseline at v1's height, its wrapper on the fused kernel at every size.
        baseline.BLOCK_SIZE_M_CANDIDATES = (v_height,)
        postnorm._MIN_TOKENS_FOR_FUSED = 0
        try:
            b_log, _ = run(baseline, data, b_state)
        finally:
            baseline.BLOCK_SIZE_M_CANDIDATES = candidates
            postnorm._MIN_TOKENS_FOR_FUSED = 32
        v_log, _ = run(v1, data, v_state)
        assert b_log == v_log, f"{label} bs={bs}: {first_difference(b_log, v_log)}"

        # The warm-up must compile the variant kernel_fn launches: same constexprs, same dtypes.
        launch = next(c for c in v_log if c[0].startswith("launch"))
        assert launch[2] == warmups[0][2], (launch[2], warmups[0][2])
        assert [a[2] for a in launch[1][1:]] == [a[2] for a in warmups[0][1][1:]]
        print(f"ok: {label:12s} bs={bs:3d}: height {v_height} (SGLang {b_height}), "
              f"{'fused' if b_fused else 'fallback'} post-norm in SGLang; v1's {len(v_log)} calls match "
              f"baseline's once aligned")


if __name__ == "__main__":
    main()

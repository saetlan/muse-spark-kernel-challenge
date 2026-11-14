"""CPU check of the Marlin-style E2M1 -> BF16 bit trick we plan to run as inline PTX.

Run: .venv/bin/python moe/local/test_dequant_trick.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4 import _E2M1_GRID, dequantize, random_weights  # noqa: E402

MASK_SIGN = 0x80008000
MASK_MAG = 0x70007000
RIGHT_SHIFT = 6  # BF16 exponent bits (8) - E2M1 exponent bits (2)
# Scales are lifted by 2^SCALE_SHIFT before the bf16 multiply so products stay normal;
# the epilogue multiplies by 2^(126 - SCALE_SHIFT) * global_scale.
SCALE_SHIFT = 64


def e2m1_values():
    mags = torch.tensor(_E2M1_GRID, dtype=torch.float32)
    return torch.cat((mags, -mags))  # codes 0..15: sign is bit 3


def marlin_bits(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q: int64 holding a 32-bit word. Returns the two bf16x2 words Marlin builds first
    (from the top nibble of each 16-bit half, then from the next nibble down)."""
    out1 = (q & MASK_SIGN) | ((q & MASK_MAG) >> RIGHT_SHIFT)
    q = (q << 4) & 0xFFFFFFFF
    out2 = (q & MASK_SIGN) | ((q & MASK_MAG) >> RIGHT_SHIFT)
    return out1, out2


def bf16_halves(word: torch.Tensor) -> torch.Tensor:
    """int64 32-bit words -> float32 [..., 2] of the (low, high) bf16 halves."""
    lo = (word & 0xFFFF).to(torch.int32) << 16
    hi = ((word >> 16) & 0xFFFF).to(torch.int32) << 16
    return torch.stack((lo.view(torch.float32), hi.view(torch.float32)), dim=-1)


def test_every_code():
    want = e2m1_values()
    for code in range(16):
        # Put the code in the top nibble of both halves.
        q = torch.tensor([(code << 12) | (code << 28)], dtype=torch.int64)
        out1, _ = marlin_bits(q)
        got = bf16_halves(out1)[0] * 2.0**126
        assert torch.equal(got, torch.full((2,), float(want[code]))), (code, got, want[code])
    print("ok: all 16 E2M1 codes decode exactly (value * 2^-126 in bf16)")


def test_all_nibble_positions():
    """Each 32-bit word holds 8 codes; 4 rounds of (decode, shift by 4) visit all of them."""
    gen = torch.Generator().manual_seed(0)
    words = torch.randint(0, 2**32, (4096,), generator=gen, dtype=torch.int64)
    want = e2m1_values()
    q = words.clone()
    for rnd in range(4):
        out1 = (q & MASK_SIGN) | ((q & MASK_MAG) >> RIGHT_SHIFT)
        got = bf16_halves(out1) * 2.0**126
        nib_lo = (words >> (12 - 4 * rnd)) & 0xF
        nib_hi = (words >> (28 - 4 * rnd)) & 0xF
        assert torch.equal(got[:, 0], want[nib_lo]) and torch.equal(got[:, 1], want[nib_hi]), rnd
        q = (q << 4) & 0xFFFFFFFF
    print("ok: 4 rounds per word decode all 8 nibble positions")


def test_scaled_product_is_exact_and_normal():
    """(e2m1 * 2^-126) * (sf * 2^SCALE_SHIFT) in bf16 must equal e2m1 * sf * 2^(SCALE_SHIFT-126)
    exactly for every code and every finite E4M3 scale, with no subnormal results."""
    vals = e2m1_values() * 2.0**-126  # exact in bf16 (some subnormal)
    sf = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    sf = sf[torch.isfinite(sf) & (sf > 0)]
    lifted = (sf * 2.0**SCALE_SHIFT).to(torch.bfloat16)
    assert torch.equal(lifted.float(), sf * 2.0**SCALE_SHIFT), "lifted scale not exact in bf16"
    a = vals.to(torch.bfloat16)
    assert torch.equal(a.float(), vals), "subnormal code values not exact in bf16"
    prod = (a[:, None] * lifted[None, :])  # bf16 multiply on CPU honors subnormal inputs
    exact = e2m1_values()[:, None] * sf[None, :] * 2.0 ** (SCALE_SHIFT - 126)
    assert torch.equal(prod.float(), exact), "bf16 product rounded"
    nz = exact != 0
    assert (exact[nz].abs() >= 2.0**-126).all(), "product would be subnormal"
    print(f"ok: code x scale products exact and normal in bf16 (scale lifted by 2^{SCALE_SHIFT})")


def test_matches_nvfp4_dequantize():
    """End to end on a real-layout tile: decode via the trick, compare to nvfp4.dequantize."""
    gen = torch.Generator().manual_seed(1)
    packed, sf, gs = random_weights(64, 256, generator=gen, device="cpu")
    ref = dequantize(packed, sf, gs)  # [64, 256]
    # Element 2i is the low nibble of byte i. Decode each nibble with the trick.
    lo = (packed & 0xF).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    def decode(nib):
        q = (nib << 12)  # top nibble of the low half
        out1 = (q & MASK_SIGN) | ((q & MASK_MAG) >> RIGHT_SHIFT)
        return bf16_halves(out1)[..., 0]
    codes = torch.stack((decode(lo), decode(hi)), dim=-1).flatten(-2)  # [64, 256], value * 2^-126
    scale = (sf.float() * 2.0**SCALE_SHIFT).to(torch.bfloat16).repeat_interleave(16, dim=-1)
    w = (codes.to(torch.bfloat16) * scale).float() * (2.0 ** (126 - SCALE_SHIFT) * gs)
    err = (w - ref).abs().max().item()
    assert err <= 1e-6 * ref.abs().max().item(), err
    print(f"ok: tile matches nvfp4.dequantize (max abs diff {err:.2e})")


if __name__ == "__main__":
    test_every_code()
    test_all_nibble_positions()
    test_scaled_product_is_exact_and_normal()
    test_matches_nvfp4_dequantize()

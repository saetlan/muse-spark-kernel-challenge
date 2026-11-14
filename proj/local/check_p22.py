"""Exactness of the P22 (F12) format: encode -> decode (the kernel's integer ops, prmt emulated) ->
escape corrections reproduces every weight bit for bit, on (1) all 65,536 BF16 bit patterns
(NaN, Inf, zeros, subnormals included) and (2) Gaussian weights at the real shapes, whose escape
rate must stay under the fallback threshold (0.5%).

Run: .venv/bin/python proj/local/check_p22.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p9"))
import re  # noqa: E402

from p22_layout import M32, p22_dec_bits, p22_decode_all, p22_decode_step, p22_encode, prmt  # noqa: E402
from p9_layout import k_perm, p9_pack  # noqa: E402


def roundtrip(w):
    n, k = w.shape
    kp = k_perm(k)
    frag = p9_pack(w)
    enc = p22_encode(frag, kp)
    dec = p22_decode_all(enc)                                  # bits per fragment slot
    bits = frag.view(torch.int16).to(torch.int64) & 0xFFFF
    esc_mask = dec != bits
    # every mismatch must be an escape that decodes to exactly w_dec of its tile
    rt = torch.arange(frag.shape[0]).view(-1, 1, 1, 1).expand_as(dec)
    assert torch.equal(dec[esc_mask], p22_dec_bits(enc["tops"])[rt[esc_mask]]), "non-escape decode mismatch"
    assert int(esc_mask.sum()) <= enc["n_esc"]
    # rebuild W in original layout from decoded values + corrections, compare bit for bit
    wbits = (w.view(torch.int16).to(torch.int64) & 0xFFFF).clone()
    recon = torch.full_like(wbits, -1)
    # decoded non-escape values go back through the inverse of p9_pack + k_perm
    v = dec.view(n // 16, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)
    recon[:, kp] = v
    recon[enc["esc_row"], enc["esc_k"]] = enc["esc_bits"]
    assert torch.equal(recon, wbits), "reconstruction differs"
    return enc["n_esc"] / enc["n"]


def cuda_f12_decode():
    """Parse f12_decode's body out of proj/p9/p9.py and turn it into a Python function over uint32
    words, so the CUDA text itself is what gets checked (C operators on uint32 and __byte_perm)."""
    src = (Path(__file__).resolve().parents[1] / "p9" / "p9.py").read_text()
    body = src[src.index("f12_decode(uint32_t q, uint32_t ra, uint32_t rb, uint32_t top4) {"):]
    body = body[body.index("{") + 1:body.index("\n}\n")]
    stmts = [st.strip() for st in body.split(";") if st.strip()]
    py = []
    for st in stmts:
        st = re.sub(r"0x([0-9A-Fa-f]+)u", lambda mo: str(int(mo.group(1), 16)), st)
        st = st.replace("__byte_perm", "prmt")
        if st.startswith("return make_uint4("):
            py.append("return (" + st[len("return make_uint4("):-1] + ")")
            continue
        assert st.startswith("const uint32_t "), st
        depth, parts, cur = 0, [], ""
        for ch in st[len("const uint32_t "):]:
            depth += ch == "("
            depth -= ch == ")"
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        for decl in parts:
            name, expr = decl.split("=", 1)
            py.append(f"{name.strip()} = ({expr.strip()}) & M32")
    code = "def f(q, ra, rb, top4):\n" + "\n".join("    " + line for line in py)
    env = {"prmt": prmt, "M32": M32}
    exec(code, env)
    return env["f"]


def check_cuda_text():
    f = cuda_f12_decode()
    g = torch.Generator().manual_seed(3)
    n = 1 << 16
    q = torch.randint(0, 2**32, (n,), generator=g, dtype=torch.int64)
    ra = torch.randint(0, 2**32, (n,), generator=g, dtype=torch.int64)
    rb = torch.randint(0, 2**32, (n,), generator=g, dtype=torch.int64)
    for top in (15, 100, 127, 140, 255):
        want = p22_decode_step(q, ra, rb, torch.tensor(top))
        got = f(q, ra, rb, (top * 0x01010101) & M32)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), f"CUDA f12_decode differs from the proven decoder at top {top}"
    print("ok: the CUDA text of f12_decode, parsed and executed, matches the proven decoder bit for bit")


def main():
    check_cuda_text()
    allbits = torch.arange(65536, dtype=torch.int64).to(torch.int16)
    perm = torch.randperm(65536, generator=torch.Generator().manual_seed(0))
    w = allbits[perm].view(torch.bfloat16).view(16, 4096)
    rate = roundtrip(w)
    print(f"ok: all 65,536 BF16 bit patterns round-trip exactly (escape rate {rate:.3f} on this adversarial tensor)")
    torch.manual_seed(1)
    for name, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}.items():
        w = (torch.randn(n, k) / math.sqrt(k)).to(torch.bfloat16)
        rate = roundtrip(w)
        assert rate < 0.005, (name, rate)
        print(f"ok: {name:8s} Gaussian weights round-trip exactly; escapes {rate * 100:.4f}% -> {12 + 32 * rate:.3f} bits/weight")


if __name__ == "__main__":
    main()

"""CPU dry run of g2p.py's Python wiring and lab_g2p.py's control flow (no GPU): g2p_install / g2p_ok /
g2p_m11_run / g2p_check / g2p_warm against mocked M11 kernels (a dense reference for GEMM2 + post-norm, the
extension's g2p_gemm emulated with the same math), then sec_gemm2_post with a fake de16_ab whose timings make the
fused path win at some classes and lose at others; a planted wrong fused result must make the section install
nothing.

Run: .venv/bin/python shared/glue/dryrun_g2p.py
"""

import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
H, I, TOP_K = 64, 32, 8  # small widths: wiring only


def post_norm(y, ids, tw, start, eps=1e-8, nw=None):
    tokens = tw.shape[0]
    out = torch.zeros(tokens, H)
    for t in range(tokens):
        for k in range(TOP_K):
            p = t * TOP_K + k
            if 0 <= int(ids[p]) - start < 32:
                v = y[p].float() * nw.float()
                out[t] += tw[t, k] * v * torch.rsqrt((v * v).mean() + eps)
    return out.to(torch.bfloat16)


class FakeExt:
    bad = False

    def g2p_attrs(self, *a):
        return torch.tensor([64, 0, 21000])

    def m11_attrs(self, *a):
        return torch.tensor([62, 0, 20000])

    def g2p_gemm(self, w, sf, c, ids, a, y, n_pairs, start, rt, out_cols, tt, nt, d, ks, grid, pdl, nw, tw, out, cnt,
                 eps):
        assert (tt, nt, d, ks, pdl) in ns["G2P_INSTANCES"] and int(cnt.abs().sum()) == 0
        y.copy_((a.float() @ W2.T).to(torch.bfloat16))
        out.copy_(post_norm(y, ids, tw, start, eps, nw))
        if FakeExt.bad:
            out.mul_(1.5)
        return out


def m11_gemm1(x, ids, state, s1):
    return (x.float()[torch.arange(ids.numel()) // TOP_K] @ W1.T).to(torch.bfloat16)


def m11_gemm2(act, ids, state, s2):
    return (act.float() @ W2.T).to(torch.bfloat16)


def m11_post_norm(y, ids, tw, state, tokens, post=None):
    return post_norm(y, ids, tw, state["m11"]["start"], 1e-8, state["norm_w"])


gen = torch.Generator().manual_seed(0)
W1, W2 = torch.randn(I, H, generator=gen) * 0.1, torch.randn(H, I, generator=gen) * 0.1
ns = {"__name__": "g2p_dry", "torch": torch, "load_inline": None, "tempfile": tempfile, "M11_MAX_TOKENS": 320,
      "POST_NORM_EPS": 1e-8, "m11_gemm1": m11_gemm1, "m11_gemm2": m11_gemm2, "m11_post_norm": m11_post_norm}
exec(compile((HERE / "g2p.py").read_text(), "g2p.py", "exec"), ns)
torch.cuda.synchronize = lambda *a, **k: None
_randn, _rand = torch.randn, torch.rand
torch.randn = lambda *a, device=None, **k: _randn(*a, **k)
torch.rand = lambda *a, device=None, **k: _rand(*a, **k)
_Gen = torch.Generator
torch.Generator = lambda device=None: _Gen()

CONFIGS = ((5, {"s1": {}, "s2": {"tt": 1, "nt": 1, "d": 4, "ks": 4, "c": 4, "pdl": 1}}),
           (9, {"s1": {}, "s2": {"tt": 2, "nt": 1, "d": 4, "ks": 4, "c": 8, "pdl": 1}}),
           (17, {"s1": {}, "s2": {"tt": 2, "nt": 1, "d": 4, "ks": 4, "c": 8}}),
           (33, {"s1": {}, "s2": {"tt": 2, "nt": 2, "d": 2, "ks": 4, "c": 8}}),
           (321, None))


def new_state():
    st = {"m11_configs": CONFIGS, "norm_w": (1 + 0.1 * _randn(H, generator=gen)).to(torch.bfloat16),
          "m11": {"w2": None, "w2_s": None, "c2": None, "start": 0, "rt_w2": 4, "hidden": H, "count": 32},
          "sms": 132, "g2p_ext": FakeExt(), "m11_ext": FakeExt(), "g2p_cnt": torch.zeros(32 + 320, dtype=torch.int32)}
    return st


# --- wiring
st = new_state()
assert all(ns["g2p_ok"](c["s2"]) for _b, c in CONFIGS[:4])
assert not ns["g2p_ok"]({"tt": 2, "nt": 2, "d": 2, "ks": 2, "c": 8})
ns["g2p_install"](st, {9: True, 33: True})
on = {b: (c or {}).get("g2p") for b, c in st["m11_configs"]}
assert on == {5: False, 9: True, 17: False, 33: True, 321: None}, on
assert st["g2p_base_configs"] is CONFIGS and "g2p" not in CONFIGS[1][1]  # base configs untouched
ns["g2p_install"](st, {9: False})
assert not any((c or {}).get("g2p") for _b, c in st["m11_configs"])
err, clean = ns["g2p_check"](st)
assert err < 1e-6 and clean, (err, clean)
ns["g2p_warm"](st)
FakeExt.bad = True
err_bad, _ = ns["g2p_check"](st)
FakeExt.bad = False
assert err_bad > 0.1, err_bad
print(f"ok: g2p_install / g2p_ok / g2p_m11_run / g2p_check / g2p_warm wiring (planted 1.5x error seen as {err_bad:.2f})")

# --- the lab section with a fake de16_ab
LOG = []
lab_ns = dict(ns)
lab_ns.update(log=lambda m: LOG.append(m), LAB_SUMMARY=[], LAB_FAILS=[], _error_text=lambda e, k=200: str(e),
              lab_fail=lambda w, e: LOG.append(f"FAIL {w}"),
              SCORED=(("dec1 sw", 1, 1, "sliding", 46), ("dec16 sw", 16, 1, "sliding", 46),
                      ("dec32 sw", 32, 1, "sliding", 46), ("ver1 sw", 1, 8, "sliding", 46),
                      ("ver16 sw", 16, 8, "sliding", 46)))
SPEED = {5: 0.98, 9: 0.99, 17: 1.01, 33: 0.999}  # fused / two-kernel time per class


def fake_ab(lab, state, shapes, set_variant, seed):
    below = min(b for b in (5, 9, 17, 33) if all(s[1] * s[2] < b for _i, s in shapes))
    tot = {}
    for v in (False, True):
        set_variant(v)
        assert bool(next(c for b, c in state["m11_configs"] if b == below).get("g2p")) == v
        tot[v] = 100.0 * (SPEED[below] if v else 1.0)
    return tot, 1e-4, 1.0


class FakeLab:
    def elapsed(self):
        return 1.0


lab_ns["de16_ab"] = fake_ab
exec(compile((HERE / "lab_g2p.py").read_text(), "lab_g2p.py", "exec"), lab_ns)
st = new_state()
lab_ns["sec_gemm2_post"](FakeLab(), st, 100.0)
text = "\n".join(LOG)
assert "V14_FUSED = {5: True, 9: True, 17: False, 33: False}" in text, text
assert text.count("g2p attrs") == 4 and "fused regs/spill/smem [64, 0, 21000]" in text, text
on = {b: bool((c or {}).get("g2p")) for b, c in st["m11_configs"]}
assert on == {5: True, 9: True, 17: False, 33: False, 321: False}, on
# planted wrong fused result: the correctness gate installs nothing
FakeExt.bad = True
LOG.clear()
st = new_state()
lab_ns["sec_gemm2_post"](FakeLab(), st, 100.0)
FakeExt.bad = False
assert "V14_FUSED = {5: False, 9: False, 17: False, 33: False}" in "\n".join(LOG)
assert not any((c or {}).get("g2p") for _b, c in st["m11_configs"])
print("ok: sec_gemm2_post installs per class only when >= 0.3% faster (5, 9 on; 17 slower, 33 under the bar off) "
      "and installs nothing when the correctness gate fails")

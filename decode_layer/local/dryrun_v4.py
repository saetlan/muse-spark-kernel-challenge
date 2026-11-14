"""CPU dry run of submission_v4 + the DE6 lab (lab_de6.py): v4's prepare (v3's checks, then the projections
and attention drop-in checks), the lab's per-case mode choice and its installation, and v4's kernel_fn under
each mode. The bf16_proj kernels run as stand-ins that compute x @ W^T from their own arguments; the
attention session's kernel_fn is replaced by a stand-in with its contract (qkv [B, L, 2304] in, [B, L, 1024]
out, new cache rows written); everything else as in dryrun_v3.

Run: .venv/bin/python decode_layer/local/dryrun_v4.py [decode_layer/lab_de6.py]
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v2  # noqa: E402
import dryrun_v3  # noqa: E402

DECODE = base.DECODE
WARMING = [False]  # bf16_proj's own warm-up runs its CHOICE with BF16 x (its problem's inputs)


def install_v4():
    real = base._Kernel.__getitem__

    def getitem(self, grid):
        name = self.fn.__name__
        if name in ("pj__fma_kernel", "pj__swap_kernel", "pj__gemm_kernel"):
            def launch(*args, **kw):
                base.hit(name)
                if name == "pj__gemm_kernel":
                    x, w, out = args[:3]
                else:
                    w, x, out = args[:3]
                m = out.shape[0]
                n = out.shape[1]
                if w.dtype == torch.float32 and m <= 8 and not WARMING[0]:  # the router up to 8 tokens: FP32 input
                    assert x.dtype == torch.float32, "router at <= 8 tokens must get the FP32 router input"
                res = x[:m].float() @ w[:n].float().T
                if kw.get("ATOMIC"):
                    out.add_(res.to(out.dtype))
                else:
                    out.copy_(res.to(out.dtype))
            return launch
        return real(self, grid)

    base._Kernel.__getitem__ = getitem
    lib = types.ModuleType("triton.language.extra.libdevice")
    extra = types.ModuleType("triton.language.extra")
    extra.libdevice = lib
    sys.modules["triton.language.extra"] = extra
    sys.modules["triton.language.extra.libdevice"] = lib
    gdc = types.ModuleType("triton.language.extra.cuda")  # attention v5's PDL chain (gdc_wait / launch_dependents)
    gdc.gdc_wait = lambda: None
    gdc.gdc_launch_dependents = lambda: None
    extra.cuda = gdc
    sys.modules["triton.language.extra.cuda"] = gdc
    import contextlib
    fi = types.ModuleType("flashinfer")  # bf16_proj v4's warm-up picks FlashInfer tactics under autotune
    fi.autotune = lambda enabled=True: contextlib.nullcontext()
    sys.modules.setdefault("flashinfer", fi)
    sys.modules["triton.language"].extra = extra
    base.mod("sglang.kernels.ops.muse_spark_v12.norm_gate", rms_norm_gate=lambda *a, **k: None)


def write_cache(adata):
    """The new cache rows both attention paths must write: fp8(k), fp8(v) from qkv (stand-in semantics)."""
    t = adata["qkv"].reshape(-1, 2304)
    adata["k_cache"].view(-1, 128).view(torch.uint8)[adata["cache_loc"]] = t[:, 1024:1152].float().to(torch.float8_e4m3fn).view(torch.uint8)
    adata["v_cache"].view(-1, 128).view(torch.uint8)[adata["cache_loc"]] = t[:, 1152:1280].float().to(torch.float8_e4m3fn).view(torch.uint8)


def att_stub(adata):
    base.hit("att")
    cfg = adata["config"]
    qkv = adata["qkv"]
    b, l = cfg["batch"], cfg["q_len"]
    assert qkv.shape == (b, l, 2304) and qkv.dtype == torch.bfloat16
    for key in ("positions", "cache_loc", "page_table", "cache_seqlens", "k_cache", "v_cache", "k_scale"):
        assert key in adata
    write_cache(adata)
    return qkv.reshape(b * l, 2304)[:, :1024].reshape(b, l, 1024).contiguous()


L9_CALLS = []


class L9Stub:
    """parts/l9.py's staging extension: validates arguments as its TORCH_CHECKs do and records each call."""

    def l9_ranges(self, a, na, b, nb, mode, blocks):
        assert 0 <= na <= a.numel() * a.element_size() and 0 <= nb <= b.numel() * b.element_size()
        assert mode in (0, 1) and blocks > 0
        L9_CALLS.append(("ranges", a.data_ptr(), na, mode))

    def l9_experts(self, ids, local_start, w, sf, rt_total, ks, f16, mode, blocks):
        assert ids.dtype == torch.int32 and ids.is_contiguous() and local_start.dtype == torch.int32
        assert ks in (1, 2, 4) and 1 <= f16 <= 16 and mode in (0, 1) and blocks > 0
        assert w.numel() * w.element_size() == 32 * rt_total * 64 * 32 * 16
        assert sf.numel() * sf.element_size() == 32 * rt_total * 64 * 32 * 2
        L9_CALLS.append(("experts", w.data_ptr(), rt_total, ks, f16, mode))


def load(path):
    """The stubbed CPU world (dryrun_d0 / v2 / v3 / v4 stubs) and the lab or submission at path, with the
    sibling drop-ins replaced by CPU stand-ins."""
    base.install_stubs()
    dryrun_v2.install_more()
    dryrun_v3.install_v3()
    install_v4()
    import dryrun_v12  # v12 / DE14: SGLang's cluster norm-residual module and the boundary's attributes
    dryrun_v12.install_nr()
    dryrun_v12.patch_boundary()
    base.patch_torch()
    import torch.utils.cpp_extension as ce
    real_inline = ce.load_inline
    ce.load_inline = lambda **kw: (base.hit("nvcc l9_ext"), L9Stub())[1] if kw["name"] == "l9_ext" else real_inline(**kw)
    lab = base.load_lab(path)
    if not hasattr(lab, "lab_input"):  # a clean file: borrow the lab's judge-like input generator
        lab.lab_input = base.load_lab(DECODE / "lab_de6.py").lab_input
    if hasattr(lab, "pj_run_p9"):  # bf16_proj v4 (v6): P9 and the library kernels as plain matmuls on the CPU
        def run_p9(c, x, layer, n, st):
            base.hit("pj__p9")
            assert layer != "router" or x.dtype == torch.bfloat16, "P9's router takes BF16 x (hi + lo)"
            out = x.float() @ st[layer][:n].float().T
            return out if layer == "router" else out.to(torch.bfloat16)

        def run_library(c, x, w, n):
            base.hit("pj__lib_" + c["kind"])
            return (x.float() @ w[:n].float().T).to(torch.bfloat16)

        lab.pj_run_p9 = run_p9
        lab.pj_run_library = run_library
        lab.pj_p9_setup = lambda st: base.hit("pj__p9_setup")
    if hasattr(lab, "pj_run_f12v"):  # bf16_proj v6: the 12-bit lossless kernels as plain matmuls on the CPU
        def f12_setup(kind):
            def setup(st):
                base.hit(f"pj__{kind}_setup")
                st[kind] = {name: {"p": None, "rec": None} for name in ("qkv_gate", "o", "pooler", "unpooler")}
                setattr(lab, f"pj_{kind.upper()}_MOD", types.SimpleNamespace())
            return setup

        def run_f12(kind, m_ok):
            def run(c, x, layer, n, st):
                assert layer in st[kind], f"{kind} planes for {layer} must be set up before use"
                assert m_ok(x.shape[0]), f"{kind} at m = {x.shape[0]}"
                assert x.dtype == torch.bfloat16
                base.hit(f"pj__{kind}")
                return (x.float() @ st[layer][:n].float().T).to(torch.bfloat16)
            return run

        lab.pj_f12v_setup = f12_setup("f12v")
        lab.pj_f12m_setup = f12_setup("f12m")
        lab.pj_run_f12v = run_f12("f12v", lambda m: m == 1)
        lab.pj_run_f12m = run_f12("f12m", lambda m: 2 <= m <= 32)
    lab.ROPE_POSITIONS = 1 << 10
    lab.att_kernel_fn = att_stub
    lab.att__warm_up = lambda: base.hit("att warm")
    real_pj_warm = lab.pj__warm

    def pj_warm(st):
        WARMING[0] = True
        try:
            real_pj_warm(st)
        finally:
            WARMING[0] = False

    lab.pj__warm = pj_warm
    # The D1 attention block's store_cache stand-in writes k only; make the stand-in write what att writes
    # (bytes of fp8(k), fp8(v) from qkv) so the self-check's cache comparison is meaningful on the CPU.
    real_d1 = lab._v4_attention_d1

    def d1_block(data, qkv, state):
        out = real_d1(data, qkv, state)
        write_cache({**{k: data[k] for k in lab.V4_ATT_KEYS}, "qkv": qkv})
        return out * 0 + qkv[:, :1024]

    lab._v4_attention_d1 = d1_block
    return lab


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de6.py"
    lab = load(path)
    lines = []
    if hasattr(lab, "log"):  # lab files only
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    n = [0]

    def fake_time(self, fn, reps=15, cold=True):
        before = dict(base.CALLS)
        fn()
        n[0] += 1
        ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
        t = 100.0
        if "att" in ran:
            t -= 10
        if any(k.startswith("pj__") for k in ran):
            t += 3  # projections slower here, so the lab's pick (att only) differs from the loop's last variant
        return t
    if hasattr(lab, "Lab"):  # lab files only
        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    text = "\n".join(lines)
    v4lines = [p for p in printed if p.startswith(("[v4]", "[v6]", "[v9]")) and ("projections" in p or "attention" in p)]
    # P9 must be set up wherever a projection choice uses it (v16 drops P9: then no choice may name it)
    p9_used = hasattr(lab, "pj_CHOICE") and any(c.get("kind") == "p9" for (layer, _m), c in lab.pj_CHOICE.items()
                                                 if not (layer == "router" and _m == 8 and lab.pj_CHOICE[("router", 8)].get("kind") != "p9"))
    p9_ready = not hasattr(lab, "pj_run_p9") or (base.CALLS.get("pj__p9_setup", 0) > 0 if p9_used
                                                  else not getattr(lab, "pj_P9_ENABLED", True))
    f12 = hasattr(lab, "pj_run_f12v")
    f12_ready = not f12 or (base.CALLS.get("pj__f12v_setup", 0) > 0 and base.CALLS.get("pj__f12m_setup", 0) > 0)
    f12_ran = base.CALLS.get("pj__f12v", 0) > 0 and base.CALLS.get("pj__f12m", 0) > 0  # prepare's pj self-check
    v3lines = [p for p in printed if p.startswith("[v3]")]
    v3_on = len(v3lines) >= 5 and all(": on (" in v for v in v3lines)
    mode = state["v4_mode"]
    # paths under the installed modes (24 tokens: a test-only count, not in any V4_MODE)
    paths = {}
    cases = {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"), 24: (3, 8, "sliding"), 256: (32, 8, "full")}
    for t, (b, q, layer) in cases.items():
        base.CALLS.clear()
        data = lab.lab_input(b, q, layer, seed=t, device="cpu")
        out = lab.kernel_fn(data, state)
        assert out[0].shape == (t, 8192) and out[2].dtype == torch.int32
        paths[(t, layer == "sliding")] = sorted(base.CALLS)
    for k, p in paths.items():
        print(f"{k}: mode {mode.get(k)} -> {[c for c in p if c in ('att', 'fa3') or c.startswith('pj__')]}")
    want = lambda k, key: mode.get(k, {}).get(key, False)
    follows = {
        "att path follows the mode": all(("att" in p) == want(k, "att") and ("fa3" in p) == (not want(k, "att"))
                                         for k, p in paths.items()),
        "pj path follows the mode": all(any(c.startswith("pj__") for c in p) == (want(k, "pj") and k[0] in (1, 8, 16, 256))
                                        for k, p in paths.items()),
        "24 tokens (test-only): both off": not any(c == "att" or c.startswith("pj__") for c in paths[(24, True)]),
    }
    if f12:  # bf16_proj v6's CHOICE: F12V at 1 token, F12M at 8 / 16 tokens, wherever pj is on
        follows["F12V at 1 token, F12M at 8 / 16 (pj on)"] = all(
            ("pj__f12v" in p) == (want(k, "pj") and k[0] == 1) and ("pj__f12m" in p) == (want(k, "pj") and k[0] in (8, 16))
            for k, p in paths.items())
        follows["F12 kernels run by prepare's self-check"] = f12_ran
        follows["F12 set up before use"] = f12_ready
        follows["unpooler slice at 8 tokens through bf16_proj's choice (FlashInfer) iff pj"] = (
            ("pj__lib_fi_mm" in paths[(8, False)]) == want((8, False), "pj"))
    if "modes (us" not in text and "modes, routing locked (us" not in text:  # a clean file: its table is installed
        import ast
        import os  # the lab log whose printed V4_MODE the clean file must carry (default: DE6's, for v4)
        de6 = (DECODE / "profiles" / os.environ.get("DRYRUN_PICKS_LOG", "lab_de6.d48630d6950a4589.log")).read_text()
        # the last printed V4_MODE, as make_v4.py's printed_literal copies it (DE12 prints de6's, then the final one)
        picks = ast.literal_eval([ln for ln in de6.splitlines() if "V4_MODE = " in ln][-1].split("V4_MODE = ", 1)[1])
        checks = {
            "v4 self-checks on": len(v4lines) == 2 and all(": on (" in v for v in v4lines),
            "P9 set up before use (bf16_proj v4)": p9_ready,
            "v3 self-checks on (M11 included)": v3_on,
            "state = the file's V4_MODE": mode == lab.V4_MODE and mode is not lab.V4_MODE,
            "V4_MODE = DE6's printed picks": lab.V4_MODE == picks and len(picks) == 12,
            "on and off cases both exercised": {want(k, "att") for k in paths} == {True, False},
            **follows,
            "no lab code in the file": not hasattr(lab, "sec_modes") and not hasattr(lab, "Lab"),
        }
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k)
        if not all(checks.values()):
            raise SystemExit(1)
        print("dry run v4 (clean file): all checks passed")
        return
    modes_line = [ln for ln in lines if ln.startswith(("modes (us", "modes, routing locked (us"))][0]
    decisions = {}
    for seg in modes_line.split("vs off): ", 1)[1].split("; ") if "routing locked" in modes_line else \
            modes_line.split("routing agreement): ", 1)[1].split("; "):
        for nm, b, q, layer, _w in lab.SCORED:
            if seg.startswith(nm + " "):
                decisions[(b * q, layer == "sliding")] = {"att": "-> att1" in seg, "pj": " pj1" in seg}
    checks = {
        "v4 self-checks on": len(v4lines) == 2 and all(": on (" in v for v in v4lines),
        "P9 set up before use (bf16_proj v4)": p9_ready,
        "v3 self-checks on (M11 included)": v3_on,
        "12 cases chosen": len(mode) == 12 and all(set(m) == {"att", "pj"} for m in mode.values()),
        **follows,
        "lab picked att only (fake timings)": all(m == {"att": True, "pj": False} for m in mode.values()),
        "V4_MODE printed = state": ("V4_MODE = " + repr(dict(sorted(mode.items())))) in text,
        "state = the lab's per-case decision": decisions == mode,
        "no failures": "failures 0" in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("V4_MODE"),
        "log under 12K": len(text) < 12000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v4: all checks passed")


if __name__ == "__main__":
    main()

"""Static check of fa3t_block.py against the verified code it copies (Triton does not run on the CPU):
1. _prologue_f8 = submission.py's _prologue except: name, no QK_FACTOR parameter or multiply, query
   stored as tl.float8e4nv instead of tl.bfloat16 (line by line, after those substitutions);
2. run_fa3t's flash_attn_with_kvcache call passes the baseline sglang_path's arguments (cache, page
   table, seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale = sm_scale x qk factor, causal, window
   size, descale) plus num_splits / pack_gqa, the precomputed scheduler_metadata only with FA3's
   defaults, and the output goes through the same rms_norm_gate call.
Run: .venv/bin/python attention/local/check_fa3t.py"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = open(os.path.join(os.path.dirname(HERE), "submission.py")).read()
BLK = open(os.path.join(HERE, "fa3t_block.py")).read()
bad = 0


def check(ok, what):
    global bad
    print(("ok   " if ok else "FAIL ") + what)
    bad += not ok


def body(src, name):
    i = src.index(f"def {name}(")
    j = src.index("\n\n\n", i)
    return src[i:j]


p = body(SUB, "_prologue")
f = body(BLK, "_prologue_f8")
want = (p.replace("def _prologue(", "def _prologue_f8(")
        .replace("              QK_FACTOR: tl.constexpr, EPS: tl.constexpr):", "")
        .replace("HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr,\n", "HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr):\n")
        .replace(" * QK_FACTOR", "").replace("q_e.to(tl.bfloat16)", "q_e.to(tl.float8e4nv)").replace("q_o.to(tl.bfloat16)", "q_o.to(tl.float8e4nv)"))
norm = lambda s: [ln.strip() for ln in s.split("\n") if ln.strip() and not ln.strip().startswith("#")]  # noqa: E731
wl, fl = norm(want), norm(f)
fl = [ln for ln in fl if ln != "k_cache_ptr, v_cache_ptr,"] if False else fl
diff = [(a, b) for a, b in zip(wl, [ln for ln in fl]) if a != b]
check(len(wl) == len(fl) and not diff, f"_prologue_f8 = _prologue with the query in E4M3 and no qk factor ({diff[:2]})")
s = body(SUB, "sglang_path")
r = body(BLK, "run_fa3t")
call = lambda src: re.search(r"flash_attn_with_kvcache\((.*?)\)\n", src, re.S).group(1)  # noqa: E731
args = lambda c: {a.split("=")[0].strip() for a in re.split(r",\s*", " ".join(c.split())) if "=" in a}  # noqa: E731
sa, ra = args(call(s)), args(call(r))
check(sa - {"q"} <= ra and ra - sa == {"num_splits", "pack_gqa"}, f"FA3 call arguments: baseline {sorted(sa)}, fa3t {sorted(ra)}")
check("softmax_scale=cfg[\"sm_scale\"] * cfg[\"qk_scale_factor\"]" in r and "window_size=(window - 1, 0) if window else (-1, -1)" in r
      and "causal=True" in r, "softmax scale, causal flag and window size as the baseline")
check("scheduler_metadata=data[\"scheduler_metadata\"] if default else None" in r and "default = num_splits == 0 and pack_gqa is None" in r,
      "precomputed scheduler metadata only with FA3's default split / packing")
check("rms_norm_gate(out.reshape(tokens, hq * 64).to(torch.bfloat16), gate, None, head_dim=64, eps=RMS_EPS," in r
      and "rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS," in s,
      "norm-gate call as the baseline")
check("HQ=hq, HKV=hkv, ROPE=rope, EPS=RMS_EPS" in r and "q8 = torch.empty((tokens, hq, 64), dtype=FP8" in r,
      "prologue launch: FP8 query buffer, constexprs")
print("check_fa3t:", "all ok" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)

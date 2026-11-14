"""Mutation test of check_v2_cpu.py: plants bugs in v2/gmm.py one at a time, rebuilds submission.py,
expects each to fail the check, then restores the clean files.
Run from the project root: .venv/bin/python prefill_layer/local/mutate_v2_checks.py"""
import subprocess, sys, shutil
G = "prefill_layer/v2/gmm.py"
clean = open(G).read()  # the current, clean gmm.py
MUTS = {
  "offs not cumsum": ("offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)", "offs = counts[:LOCAL_EXPERTS].to(torch.int32)"),
  "act uses w2 global scale": ('_gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w13_gs"]', '_gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w2_gs"]'),
  "inverse perm wrong": ("inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))", "inv.copy_(perm)"),
  "nibble order swapped": ("tl.join(lo, hi)", "tl.join(hi, lo)"),
  "scale group 32": ("(cols // 8)[None, :]", "(cols // 16)[None, :]"),
  "combine skips gs": ("* tl.load(gs_ptr + expert)", ""),
  "non-local sentinel 0": ("keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))", "keys = torch.where(local_ids.view(-1) < 0, 0, local_ids.view(-1))"),
}
results = {}
try:
    for name, (a, b) in MUTS.items():
        assert clean.count(a) == 1, name
        open(G, "w").write(clean.replace(a, b))
        subprocess.run([".venv/bin/python", "prefill_layer/local/make_v2.py"], check=True, capture_output=True)
        r = subprocess.run([".venv/bin/python", "prefill_layer/local/check_v2_cpu.py"], capture_output=True, text=True)
        results[name] = "CAUGHT" if r.returncode else "MISSED"
finally:
    open(G, "w").write(clean)
    subprocess.run([".venv/bin/python", "prefill_layer/local/make_v2.py"], check=True, capture_output=True)
for k, v in results.items(): print(f"{v:7s} {k}")

"""Modal app for H200 experiments outside the judge (TRACKER G5): the judge's base image
(lmsysorg/sglang:dev-muse-spark-open: torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0), our proj/ code mounted
at /root/proj.

  .venv/bin/modal run proj/modal/app.py::probe
      H200 facts, ncu availability, and whether ncu may read GPU counters here (a 1-kernel profile).
  .venv/bin/modal run proj/modal/app.py::bench --file submission_rglab.py --shapes '<json>' [--ncu-kernel rg_kernel]
      bench.py on that file; with --ncu-kernel, Nsight Compute (--set full) on the first launch of the kernels
      matching the name, report printed as text.

Costs: H200 $0.001261/s (~$4.54/h) while a container runs. Credit: $1.00 without a payment method ($30 with one);
timeouts are capped (probe 7 min, bench 10 min) so no single run can exceed it.
"""

import subprocess
from pathlib import Path

import modal

PROJ = Path(__file__).resolve().parents[1]
IMAGE_TAG = "lmsysorg/sglang:dev-muse-spark-open"
# H200 needs a payment method on this account (Nov 4); H100 SXM is the same Hopper sm_90 (50 MB L2, HBM3).
GPU = "H100"

image = modal.Image.from_registry(IMAGE_TAG).add_local_dir(
    str(PROJ), "/root/proj",
    ignore=["**/__pycache__/**", "profiles/**", "scored/profiles/**", "**/*.json", "lab_e*.py"])
app = modal.App("okc-bf16proj", image=image)


def _sh(cmd, timeout=600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return f"$ {cmd}\n{r.stdout[-6000:]}{r.stderr[-3000:]}(exit {r.returncode})\n"


@app.function(gpu=GPU, timeout=420)   # $1.00 of credit: a hang costs at most ~$0.53
def probe():
    out = [_sh("nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,memory.total,driver_version --format=csv"),
           _sh("which ncu nsys nvcc python; ls /usr/local/cuda/bin | grep -i -E 'ncu|nsight' | head"),
           _sh("ncu --version"),
           _sh("python -c \"import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)\""),
           _sh("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed "
               "-c 1 python -c \"import torch; a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); "
               "(a @ a).sum().item()\"", timeout=240)]
    return "\n".join(out)


@app.function(gpu=GPU, timeout=600)   # capped for the $1.00 credit (~$0.76 max)
def bench(file: str, shapes: str, reps: int = 30, ncu_kernel: str = ""):
    run = f"cd /root/proj && python modal/bench.py /root/proj/{file} --shapes '{shapes}' --reps {reps}"
    if not ncu_kernel:
        return _sh(run, timeout=560)
    prof = (f"cd /root/proj && ncu --set full -k regex:{ncu_kernel} -c 2 --target-processes all -f -o /tmp/prof "
            f"python modal/bench.py /root/proj/{file} --shapes '{shapes}' --once")
    return _sh(prof, timeout=480) + _sh("ncu --import /tmp/prof.ncu-rep --page details --print-units base", timeout=60)


@app.local_entrypoint()
def main(task: str = "probe", file: str = "", shapes: str = "", reps: int = 30, ncu_kernel: str = ""):
    if task == "probe":
        print(probe.remote())
    else:
        print(bench.remote(file, shapes, reps, ncu_kernel))

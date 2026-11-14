"""Runs inside a Modal H200 container (proj/modal/app.py): load one of our submission or lab-base files,
prepare it on the judge's weights (reference.generate_weights, seed 5151), then for each requested
(layer, m, config) check the output against FP64 and time it like the judge (CUDA graph replay after a
512 MB L2 flush, median of `reps`). With --once it only launches each config a few times (for ncu,
which captures the kernels it is told to).

Usage: python bench.py FILE --shapes '[{"layer": "router", "m": 128, "cfg": {...}}, ...]' [--reps 30] [--once]
(cfg null = SGLang's path)
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/proj")
from reference import generate_input, generate_weights, out_features  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("okc_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_graph(fn, flush, reps):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        flush.sum()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    mod = load(a.file)
    w = generate_weights(5151)
    t0 = time.time()
    state = mod.prepare(w)
    torch.cuda.synchronize()
    print(json.dumps({"event": "prepare", "s": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name()}), flush=True)
    flush = torch.empty((512 << 20) // 8, dtype=torch.int64, device="cuda").fill_(1)
    for item in json.loads(a.shapes):
        layer, m, cfg = item["layer"], int(item["m"]), item.get("cfg")
        n = out_features(layer, m)
        x = generate_input(m, layer, 1234 + m)

        def fn(cfg=cfg, x=x, layer=layer, n=n):
            return mod.run_choice(cfg, x, layer, n, state) if cfg else mod.sglang_path(x, layer, n, state)
        out = fn()
        torch.cuda.synchronize()
        ref = (x.double() @ w[layer][:n].double().T)
        rel = float((out.double() - ref).norm() / ref.norm())
        if a.once:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            print(json.dumps({"event": "once", "layer": layer, "m": m, "rel": rel}), flush=True)
            continue
        us = time_graph(fn, flush, a.reps)
        print(json.dumps({"event": "time", "layer": layer, "m": m, "cfg": cfg, "us": round(us, 2), "rel": rel}), flush=True)


if __name__ == "__main__":
    main()

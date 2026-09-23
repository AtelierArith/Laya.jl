"""Benchmark the Python laya-mlx runtime (one checkpoint per process).

Run with the reference environment:
    ../extern/laya-mlx/.venv/bin/python bench_python.py --model aac6fef/laya-mlx
Timing mirrors extern/laya-mlx/benchmarks/worker.py: wall clock, synchronized GPU
completion, model loading excluded.
"""

import argparse
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_mlx import Agent
from laya_mlx.agent import collate_items

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "extern" / "laya-mlx" / "examples"


def workload(kind, count):
    spec = json.loads((HERE / "workload.json").read_text())
    state = json.loads((EXAMPLES / "state.json").read_text())
    definitions = list(json.loads((EXAMPLES / "questions.json").read_text()).values())
    if kind == "long":
        state["body"] = spec["long_body"] * spec["long_repeat"]
    return state, {f"q{i}": definitions[i % len(definitions)] for i in range(count)}


def stats(samples):
    return {
        "samples_ms": samples,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
        mx.synchronize()
    samples = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter_ns()
        function()
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return stats(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aac6fef/laya-mlx")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workloads", default=None, help="comma-separated kind:count, e.g. short:1,long:10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / "workload.json").read_text())
    workloads = args.workloads.split(",") if args.workloads else spec["default_workloads"]

    start = time.perf_counter()
    agent = Agent(args.model, dtype=args.dtype, device=args.device, batch_size=args.batch_size)
    mx.synchronize()
    report = {
        "backend": "python-mlx-" + args.device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "device": mx.device_info() if args.device == "gpu" else "cpu",
        },
        "model": args.model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": time.perf_counter() - start,
        "timing": "wall clock, synchronized; model loading excluded",
        "results": [],
    }
    for name in workloads:
        kind, count = name.split(":")
        state, questions = workload(kind, int(count))
        result = agent.predict(state, questions)
        items, _ = agent.prepare(state, questions)
        # Forward timing covers one batch of up to batch_size questions, as upstream.
        raw = collate_items(items[: args.batch_size], agent.tok.pad_token_id)
        batch = {k: mx.array(v) for k, v in raw.items()}
        mx.eval(batch)
        mx.clear_cache()
        mx.reset_peak_memory()
        forward = measure(lambda: mx.eval(agent._inference(**batch)), args.warmup, args.iterations)
        prepare = measure(lambda: agent.prepare(state, questions), args.warmup, args.iterations)
        e2e = measure(lambda: agent.predict(state, questions), args.warmup, args.iterations)
        e2e["questions_per_second"] = int(count) * 1000 / e2e["mean_ms"]
        report["results"].append(
            {
                "workload": name,
                "questions": int(count),
                "sequence_length": int(raw["input_ids"].shape[1]),
                "input_tokens": result["usage"]["input_tokens"],
                "prepare": prepare,
                "forward": forward,
                "e2e": e2e,
                "mlx_peak_bytes": mx.get_peak_memory(),
                "answers": {k: v.get("choice", v.get("score", v.get("noul"))) for k, v in result["answers"].items()},
            }
        )
        print(f"{name:10s} e2e p50 {e2e['p50_ms']:9.2f} ms  forward p50 {forward['p50_ms']:9.2f} ms", flush=True)
    report["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

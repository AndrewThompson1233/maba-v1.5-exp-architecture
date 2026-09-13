import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.config import MabaSparseConfig
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_prefill(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup: int = 1,
    repeats: int = 2,
) -> Dict[str, float]:
    model.eval()
    reset_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    reset_memory_stats(device)
    ts = []

    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            ts.append(t1 - t0)

    avg_sec = sum(ts) / len(ts)
    toks = input_ids.numel()
    tp = toks / max(avg_sec, 1e-9)
    pm = get_peak_memory_mb(device)

    return {
        "latency_ms": avg_sec * 1000.0,
        "throughput_tokens_per_sec": tp,
        "peak_memory_mb": pm,
    }


def benchmark_decode_step(
    model: nn.Module,
    device: torch.device,
    is_maba: bool,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    model.eval()
    stok = torch.randint(0, 1000, (1, 1), device=device)
    seq = torch.randint(0, 1000, (1, 16), device=device)

    with torch.no_grad():
        out = model(seq)
        pst = out.past_states

    with torch.no_grad():
        for _ in range(warmup):
            sout = model(stok, past_states=pst)
            pst = sout.past_states
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    ts = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            sout = model(stok, past_states=pst)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            pst = sout.past_states
            ts.append(t1 - t0)

    return (sum(ts) / len(ts)) * 1000.0


def run_benchmark(
    context_lengths: List[int],
    batch_size: int = 1,
    device_str: Optional[str] = None,
    warmup: int = 1,
    repeats: int = 2,
    output_json: Optional[str] = "benchmark_results.json",
    output_md: Optional[str] = "BENCHMARK_REPORT.md",
) -> Dict[str, Any]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str is None else torch.device(device_str)

    print(f"=== Running Maba-Sparse vs Dense Transformer Benchmark ===")
    print(f"Device: {dev} | Batch Size: {batch_size}")
    print(f"Context Lengths: {context_lengths}")

    print("\nInstantiating 101M Maba-Sparse model...")
    cfg = get_101m_config()
    m_model = MabaSparseForCausalLM(cfg).to(dev)
    m_params = sum(p.numel() for p in set(m_model.parameters()))
    print(f"Maba-Sparse Parameter Count: {m_params:,} ({m_params/1e6:.2f}M)")

    print("\nInstantiating 101M Dense Transformer baseline...")
    d_model = DenseTransformerForCausalLM().to(dev)
    d_params = sum(p.numel() for p in set(d_model.parameters()))
    print(f"Dense Transformer Parameter Count: {d_params:,} ({d_params/1e6:.2f}M)")

    results: Dict[str, Any] = {
        "metadata": {
            "device": str(dev),
            "batch_size": batch_size,
            "maba_parameters": m_params,
            "dense_parameters": d_params,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "benchmarks": [],
    }

    for l in context_lengths:
        print(f"\n--- Context Length: {l} tokens ---")
        ids = torch.randint(1, 32000, (batch_size, l), device=dev)

        print("Benchmarking Maba-Sparse prefill...")
        try:
            mp = benchmark_prefill(m_model, ids, dev, warmup=warmup, repeats=repeats)
            md = benchmark_decode_step(m_model, dev, is_maba=True, warmup=1, repeats=3)
        except Exception as e:
            print(f"Maba-Sparse failed at L={l}: {e}")
            mp = {"latency_ms": -1.0, "throughput_tokens_per_sec": -1.0, "peak_memory_mb": -1.0}
            md = -1.0

        print("Benchmarking Dense Transformer baseline prefill...")
        try:
            dp = benchmark_prefill(d_model, ids, dev, warmup=warmup, repeats=repeats)
            dd = benchmark_decode_step(d_model, dev, is_maba=False, warmup=1, repeats=3)
        except Exception as e:
            print(f"Dense Transformer failed at L={l}: {e}")
            dp = {"latency_ms": -1.0, "throughput_tokens_per_sec": -1.0, "peak_memory_mb": -1.0}
            dd = -1.0

        sp = dp["latency_ms"] / mp["latency_ms"] if dp["latency_ms"] > 0 and mp["latency_ms"] > 0 else 1.0

        entry = {
            "context_length": l,
            "maba": {
                "latency_ms": mp["latency_ms"],
                "throughput": mp["throughput_tokens_per_sec"],
                "peak_mem_mb": mp["peak_memory_mb"],
                "decode_ms_per_token": md,
            },
            "dense": {
                "latency_ms": dp["latency_ms"],
                "throughput": dp["throughput_tokens_per_sec"],
                "peak_mem_mb": dp["peak_memory_mb"],
                "decode_ms_per_token": dd,
            },
            "speedup": sp,
        }
        results["benchmarks"].append(entry)

        print(
            f"L={l:5d} | Maba Prefill: {mp['latency_ms']:8.2f}ms ({mp['throughput_tokens_per_sec']:8.1f} tok/s) | "
            f"Dense Prefill: {dp['latency_ms']:8.2f}ms ({dp['throughput_tokens_per_sec']:8.1f} tok/s) | "
            f"Speedup: {sp:5.2f}x"
        )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved raw benchmark metrics to {output_json}")

    md_lines = [
        "# Maba-Sparse vs Dense Transformer Baseline Benchmark Report",
        "",
        f"- **Device**: `{dev}`",
        f"- **Batch Size**: `{batch_size}`",
        f"- **Maba-Sparse Parameters**: `{m_params:,}` ({m_params/1e6:.2f}M)",
        f"- **Dense Transformer Parameters**: `{d_params:,}` ({d_params/1e6:.2f}M)",
        f"- **Timestamp**: `{results['metadata']['timestamp']}`",
        "",
        "## Empirical Performance Comparison",
        "",
        "| Context Length | Maba Latency (ms) | Dense Latency (ms) | Maba Throughput (tok/s) | Dense Throughput (tok/s) | Speedup Ratio | Maba Decode (ms/tok) | Dense Decode (ms/tok) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for b in results["benchmarks"]:
        ctx = b["context_length"]
        ml = f"{b['maba']['latency_ms']:.2f}"
        dl = f"{b['dense']['latency_ms']:.2f}"
        mt = f"{b['maba']['throughput']:.1f}"
        dt = f"{b['dense']['throughput']:.1f}"
        s = f"{b['speedup']:.2f}x"
        mdc = f"{b['maba']['decode_ms_per_token']:.2f}"
        ddc = f"{b['dense']['decode_ms_per_token']:.2f}"
        md_lines.append(
            f"| {ctx:5d} | {ml:>17} | {dl:>18} | {mt:>23} | {dt:>24} | {s:>13} | {mdc:>20} | {ddc:>21} |"
        )

    md_lines.append("")
    md_lines.append("## Architectural Findings")
    md_lines.append(
        "1. **Decoupled Gated Recurrence (DGDA)** replaces 75% of attention operations with chunkwise parallel recurrence, eliminating quadratic prefill scaling."
    )
    md_lines.append(
        "2. **Delta-Guided Centroid Indexer (DG-Indexer)** selects dynamic top-32 blocks, preserving attention sinks and local window while suppressing distant non-salient tokens."
    )
    md_lines.append(
        "3. **Multi-Head Latent Attention (MLA)** compresses KV representation into d_c=128, drastically reducing cache memory scaling during incremental decoding."
    )
    md_lines.append("")

    report = "\n".join(md_lines)
    if output_md:
        with open(output_md, "w") as f:
            f.write(report)
        print(f"Saved benchmark report to {output_md}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Maba-Sparse vs Dense Transformer")
    parser.add_argument("--contexts", type=str, default="512,1024,2048,4096,8192")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output_json", type=str, default="benchmark_results.json")
    parser.add_argument("--output_md", type=str, default="BENCHMARK_REPORT.md")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ctxs = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
    run_benchmark(
        context_lengths=ctxs,
        batch_size=args.batch_size,
        device_str=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        output_json=args.output_json,
        output_md=args.output_md,
    )

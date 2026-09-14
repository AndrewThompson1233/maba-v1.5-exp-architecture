# Maba-Sparse vs Dense Transformer Baseline Benchmark Report

- **Hardware**: NVIDIA Tesla T4 (16GB VRAM, Turing Architecture)
- **Device**: `cuda:0`
- **Batch Size**: `1`
- **Maba-Sparse Parameters**: `101,282,319` (101.28M)
- **Dense Transformer Parameters**: `103,533,184` (103.53M)
- **Timestamp**: `2026-09-14T06:20:37Z`

## Empirical Performance Comparison

| Context Length | Maba Latency (ms) | Dense Latency (ms) | Maba Throughput (tok/s) | Dense Throughput (tok/s) | Speedup Ratio | Maba Decode (ms/tok) | Dense Decode (ms/tok) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   512 |            597.55 |              45.36 |                   856.8 |                  11286.8 |         0.08x |                52.30 |                 14.17 |
|  1024 |           1145.44 |              78.99 |                   894.0 |                  12963.8 |         0.07x |                52.20 |                 13.83 |
|  2048 |           2325.51 |             157.31 |                   880.7 |                  13018.7 |         0.07x |                50.34 |                 14.24 |
|  4096 |           4798.34 |             401.74 |                   853.6 |                  10195.7 |         0.08x |                51.61 |                 13.92 |

## Analysis and Findings

1. **Linear Scaling of DGDA Prefill**:
   - `Maba-Sparse` maintains constant prefill throughput (~850-890 tokens/sec) across all context lengths from 512 to 4096 tokens.
   - Latency scales strictly linearly (597ms at 512 -> 1145ms at 1024 -> 2325ms at 2048 -> 4798ms at 4096).
2. **Dense Transformer PyTorch SDPA Kernels**:
   - Dense Transformer benefits from PyTorch's native FlashAttention / cuDNN C++ kernels for short sequences.
   - Pure-Python recurrent loops and einsum operations in experimental Maba incur launch overhead without dedicated C++ / CUDA kernels.
3. **Multi-Head Latent Attention (MLA)**:
   - KV representation is compressed into `d_c=128`, reducing cache memory footprint.

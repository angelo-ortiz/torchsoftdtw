"""Benchmark torchsoftdtw against the soft_dtw_cuda.py (Maghoumi) and softdtw-cuda-torch
(BGU-CS-VIL) reference implementations, on CPU and (when available) CUDA.

Run with:
    python -m torchsoftdtw.benchmark
"""

from .run_benchmark import benchmark

__all__ = ["benchmark"]

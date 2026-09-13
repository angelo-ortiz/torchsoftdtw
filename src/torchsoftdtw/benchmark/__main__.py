"""Command-line interface for the soft-DTW benchmark.

Examples:
    python -m torchsoftdtw.benchmark
    python -m torchsoftdtw.benchmark --forward-only --min-run-time 0.5
    python -m torchsoftdtw.benchmark --output benchmark_results.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_benchmark import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m torchsoftdtw.benchmark",
        description=(
            "Benchmark torchsoftdtw's soft-DTW against soft_dtw_cuda.py (Maghoumi) and "
            "softdtw-cuda-torch (BGU-CS-VIL), on CPU and, when available, CUDA, across batches "
            "with no/small/large length spread."
        ),
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="Minimum measurement time per case passed to torch's blocked_autorange (default: %(default)s).",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Time the forward pass only (default: forward + backward).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Markdown file to inject the results table into, between the <!-- benchmark --> markers.",
    )

    args = parser.parse_args()
    benchmark(
        min_run_time=args.min_run_time,
        include_backward=not args.forward_only,
        output=args.output,
    )


if __name__ == "__main__":
    main()

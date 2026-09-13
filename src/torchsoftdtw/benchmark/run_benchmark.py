"""Benchmark torchsoftdtw against soft_dtw_cuda.py (Maghoumi) and softdtw-cuda-torch (BGU-CS-VIL).

Sweeps a range of batch shapes on CPU and, when available, CUDA, each at three length-spread
levels (see generate.py): "none" (every sample padded to the same length), "small" (lengths
within 10% of max_len) and "large" (lengths down to 25% of max_len). Every case is checked for
correctness against torchsoftdtw's own output before it is timed. torchsoftdtw is the only
implementation that is natively length-aware: it masks padded positions directly inside its
batched kernel. The two reference implementations have no notion of per-sample length, so
whenever a batch's lengths are non-uniform ("small"/"large"), they are timed one sample at a
time, each sliced to its true length, rather than batched over the padded tensor (see
implementations.py).
"""

from __future__ import annotations

import contextlib
import io
import platform
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.benchmark import Compare, Measurement, Timer

from .generate import SPREADS, Batch, make_batch
from .implementations import HAS_SDTW_CUDA_TORCH, maghoumi, ours, sdtw_cuda_torch

GAMMA = 1.0
SEED = 0

# (batch_size, max_len, feat_dim) -- spanning short/large-batch to long/small-batch workloads.
CONFIGS = [
    (64, 16, 8),
    (64, 64, 8),
    (32, 256, 16),
    (16, 512, 16),
]

BENCHMARK_MARKERS = ("<!-- benchmark -->", "<!-- /benchmark -->")


def _build_impls(device: torch.device) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    impls: list[tuple[str, Callable[..., torch.Tensor]]] = [
        ("torchsoftdtw (ours)", lambda X, Y, lx, ly: ours(X, Y, lx, ly, GAMMA)),
        (
            "soft_dtw_cuda (Maghoumi)",
            lambda X, Y, lx, ly: maghoumi(
                X, Y, lx, ly, GAMMA, use_cuda=device.type == "cuda"
            ),
        ),
    ]
    if HAS_SDTW_CUDA_TORCH:
        impls.append(
            (
                "softdtw-cuda-torch",
                lambda X, Y, lx, ly: sdtw_cuda_torch(X, Y, lx, ly, GAMMA),
            )
        )
    return impls


def _check_correctness(
    outputs: dict[str, torch.Tensor], reference_name: str = "torchsoftdtw (ours)"
) -> None:
    reference = outputs[reference_name]
    for name, out in outputs.items():
        if name == reference_name:
            continue
        try:
            torch.testing.assert_close(out, reference, atol=1e-3, rtol=1e-3)
        except AssertionError as exc:
            warnings.warn(
                f"{name} disagrees with {reference_name}: {exc}", stacklevel=2
            )


def measurements(
    config: tuple[int, int, int],
    spread: str,
    device: torch.device,
    include_backward: bool,
    min_run_time: float = 0.2,
) -> list[Measurement]:
    """Time every implementation on one (batch, spread, device) configuration."""
    batch_size, max_len, feat_dim = config
    num_threads = torch.get_num_threads()
    batch: Batch = make_batch(batch_size, max_len, feat_dim, spread, device, seed=SEED)
    impls = _build_impls(device)

    # Warm up (JIT compilation, allocator caching), then sanity-check agreement.
    outputs = {}
    for name, fn in impls:
        for _ in range(2):
            out = fn(batch.X, batch.Y, batch.lengths_x, batch.lengths_y)
        outputs[name] = out
    _check_correctness(outputs)

    label = f"{device.type}{' (+backward)' if include_backward else ''}"
    description = f"B={batch_size} L={max_len} D={feat_dim} spread={spread}"

    def measure(name: str, fn: Callable[..., torch.Tensor]) -> Measurement:
        if include_backward:
            X = batch.X.clone().requires_grad_(True)

            def stmt_fn() -> None:
                X.grad = None
                out = fn(X, batch.Y, batch.lengths_x, batch.lengths_y)
                out.sum().backward()

        else:
            X = batch.X

            def stmt_fn() -> None:
                fn(X, batch.Y, batch.lengths_x, batch.lengths_y)

        return Timer(
            stmt="stmt_fn()",
            globals={"stmt_fn": stmt_fn},
            num_threads=num_threads,
            label=label,
            sub_label=name,
            description=description,
        ).blocked_autorange(min_run_time=min_run_time)

    return [measure(name, fn) for name, fn in impls]


def _device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine() or "CPU"


def _render(results: list[Measurement]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        Compare(results).print()
    return buffer.getvalue().strip("\n")


def _inject(output: Path, results: list[Measurement]) -> None:
    begin, end = BENCHMARK_MARKERS
    block = f"## Benchmark results on {_device_name()}\n\n```\n{_render(results)}\n```"
    content = output.read_text(encoding="utf-8")
    pattern = re.escape(begin) + r".*?" + re.escape(end)
    new_content, count = re.subn(
        pattern, lambda _: f"{begin}\n{block}\n{end}", content, flags=re.DOTALL
    )
    if count == 0:
        msg = f"Markers not found in {output}. Add:\n{begin}\n{end}"
        raise RuntimeError(msg)
    output.write_text(new_content, encoding="utf-8")


def benchmark(
    min_run_time: float = 0.2,
    include_backward: bool = True,
    output: Path | None = None,
) -> None:
    """Run the full sweep (CONFIGS x SPREADS x devices) and print a timing comparison table."""
    devices = [
        torch.device(t)
        for t in ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ]

    results = [
        m
        for device in devices
        for config in CONFIGS
        for spread in SPREADS
        for m in measurements(config, spread, device, include_backward, min_run_time)
    ]

    compare = Compare(results)
    compare.colorize()
    compare.print()

    if output is not None:
        _inject(output, results)

"""Uniform (X, Y, lengths_x, lengths_y) -> costs wrappers around each soft-DTW implementation.

Only torchsoftdtw's own `soft_dtw` is length-aware: it uses lengths_x/lengths_y to mask the
DP recursion directly inside its batched kernel, so padded positions never enter the alignment
no matter how uneven the batch's lengths are. soft_dtw_cuda.py (Maghoumi) and softdtw-cuda-torch
(BGU-CS-VIL) both take rectangular (B,N,D)/(B,M,D) batches with no length argument -- a single
batched call would treat the padded tail of every sequence as real, zero-valued data. To keep
the comparison fair when lengths are non-uniform, both wrappers fall back to slicing each sample
down to its true length and calling the implementation once per sample (batch size 1), then
stacking the results.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from torchsoftdtw.distances import pairwise_l2_squared
from torchsoftdtw.soft_dtw_cuda import SoftDTW as _MaghoumiSoftDTW
from torchsoftdtw.softdtw import soft_dtw as _ours

try:
    from softdtw_cuda import SoftDTW as _RefSoftDTW

    HAS_SDTW_CUDA_TORCH = True
except ImportError:
    HAS_SDTW_CUDA_TORCH = False


def _is_uniform(lengths: torch.Tensor, size: int) -> bool:
    return bool((lengths == size).all())


def _batched_or_per_sample(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    Y: torch.Tensor,
    lengths_x: torch.Tensor,
    lengths_y: torch.Tensor,
) -> torch.Tensor:
    """Call `fn` on the whole batch if lengths are uniform, else one sample at a time.

    `fn` has no notion of per-sample length, so a non-uniform batch is split into
    single-sample calls, each sliced down to that sample's true length, to avoid treating
    padding as real data.
    """
    if _is_uniform(lengths_x, X.shape[1]) and _is_uniform(lengths_y, Y.shape[1]):
        return fn(X, Y)
    outs = [
        fn(
            X[b : b + 1, : int(lengths_x[b])],
            Y[b : b + 1, : int(lengths_y[b])],
        )
        for b in range(X.shape[0])
    ]
    return torch.cat(outs, dim=0)


def ours(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengths_x: torch.Tensor,
    lengths_y: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """torchsoftdtw's native (C++/CUDA) implementation, using the real per-sample lengths."""
    D = pairwise_l2_squared(X, Y)
    return _ours(D, lengths_x, lengths_y, gamma=gamma)


def maghoumi(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengths_x: torch.Tensor,
    lengths_y: torch.Tensor,
    gamma: float,
    use_cuda: bool,
) -> torch.Tensor:
    """Maghoumi's soft_dtw_cuda.py (numba jit on CPU, numba cuda kernel on GPU)."""
    sdtw = _MaghoumiSoftDTW(use_cuda=use_cuda, gamma=gamma)
    return _batched_or_per_sample(sdtw, X, Y, lengths_x, lengths_y)


def sdtw_cuda_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengths_x: torch.Tensor,
    lengths_y: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """BGU-CS-VIL's softdtw-cuda-torch package (C++/CUDA extension, CPU and GPU)."""
    if not HAS_SDTW_CUDA_TORCH:
        raise ImportError("softdtw-cuda-torch is not installed")
    sdtw = _RefSoftDTW(gamma=gamma)
    return _batched_or_per_sample(sdtw, X, Y, lengths_x, lengths_y)

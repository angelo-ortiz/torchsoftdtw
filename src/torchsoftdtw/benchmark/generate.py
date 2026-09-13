"""Random batch generation for the soft-DTW benchmark, with controllable length spread.

A batch is a pair of padded tensors X: (B, N, D), Y: (B, M, D) plus per-sample lengths
lengths_x: (B,), lengths_y: (B,), where N = M = max_len is the padded length and each
sample's true length is drawn according to a "spread" level:

  - "none":  every sample has length == max_len (the classic, unpadded case).
  - "small": lengths are drawn uniformly from [0.9 * max_len, max_len].
  - "large": lengths are drawn uniformly from [0.25 * max_len, max_len].

Positions beyond a sample's true length are zeroed out, mimicking how a real padded
batch (e.g. from a DataLoader with variable-length sequences) would look.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SPREADS = ("none", "small", "large")

# Fraction of max_len below which lengths are sampled, per spread level.
_SPREAD_MIN_FRAC = {"none": 1.0, "small": 0.9, "large": 0.25}


@dataclass
class Batch:
    X: torch.Tensor
    Y: torch.Tensor
    lengths_x: torch.Tensor
    lengths_y: torch.Tensor


def make_lengths(
    batch_size: int, max_len: int, spread: str, generator: torch.Generator
) -> torch.Tensor:
    """Draw a (batch_size,) tensor of sequence lengths in [1, max_len] for the given spread."""
    min_frac = _SPREAD_MIN_FRAC[spread]
    if min_frac >= 1.0:
        return torch.full((batch_size,), max_len, dtype=torch.long)
    min_len = max(1, round(min_frac * max_len))
    return torch.randint(
        min_len, max_len + 1, (batch_size,), generator=generator, dtype=torch.long
    )


def make_batch(
    batch_size: int,
    max_len: int,
    feat_dim: int,
    spread: str,
    device: torch.device,
    seed: int = 0,
) -> Batch:
    """Build a random padded batch with the requested length-spread level."""
    if spread not in SPREADS:
        raise ValueError(f"Unknown spread={spread!r}. Expected one of {SPREADS}.")

    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(batch_size, max_len, feat_dim, generator=g, dtype=torch.float32)
    Y = torch.randn(batch_size, max_len, feat_dim, generator=g, dtype=torch.float32)
    lengths_x = make_lengths(batch_size, max_len, spread, g)
    lengths_y = make_lengths(batch_size, max_len, spread, g)

    for t, lengths in ((X, lengths_x), (Y, lengths_y)):
        for b in range(batch_size):
            t[b, int(lengths[b]) :] = 0.0

    return Batch(
        X=X.to(device),
        Y=Y.to(device),
        lengths_x=lengths_x.to(device),
        lengths_y=lengths_y.to(device),
    )

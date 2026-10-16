import torch
from torch import nn
from torch.autograd import Function

from . import _C  # noqa: F401 # ty: ignore[unresolved-import]
from .distances import PAIRWISE_DISTANCES


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


# ---- FakeTensor kernels for torch.compile ----


@torch.library.register_fake("torchsoftdtw::forward")
def _forward_fake(D, lengths_x, lengths_y, gamma, bandwidth):
    B, N, M = D.shape
    acc_dt = _acc_dtype(D.dtype)
    costs = D.new_empty((B,), dtype=acc_dt)
    R = D.new_empty((B, N + 2, M + 2), dtype=acc_dt)
    return costs, R


@torch.library.register_fake("torchsoftdtw::backward")
def _backward_fake(D, R, lengths_x, lengths_y, gamma, bandwidth):
    B, N, M = D.shape
    return D.new_empty((B, N, M))


class SoftDTWAutograd(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, D, lengths_x, lengths_y, gamma, bandwidth):
        costs, R = torch.ops.torchsoftdtw.forward(D, lengths_x, lengths_y, gamma, bandwidth)
        ctx.save_for_backward(D, R, lengths_x, lengths_y)
        ctx.gamma = gamma
        ctx.bandwidth = bandwidth
        return costs

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        D, R, lengths_x, lengths_y = ctx.saved_tensors
        E = torch.ops.torchsoftdtw.backward(D, R, lengths_x, lengths_y, ctx.gamma, ctx.bandwidth)
        return grad_output.unsqueeze(-1).unsqueeze(-1) * E, None, None, None, None


def soft_dtw(
    D: torch.Tensor,
    lengths_x: torch.Tensor | None = None,
    lengths_y: torch.Tensor | None = None,
    gamma: float = 1.0,
    bandwidth: int = -1,
) -> torch.Tensor:
    """Compute soft-DTW on a pre-computed cost matrix.

    Args:
        D: (B, N, M) pairwise cost matrix.
        lengths_x: (B,) actual lengths along dim 1, int32 or int64. Defaults to N for all.
        lengths_y: (B,) actual lengths along dim 2, int32 or int64. Defaults to M for all.
        gamma: Smoothing parameter (> 0).
        bandwidth: Sakoe-Chiba bandwidth around the diagonal joining (0, 0) to (lengths_x - 1, lengths_y - 1),
            in steps along the longer axis. Floored at half the diagonal's slope so that a warping path always
            exists. -1 means no constraint.

    Returns:
        costs: (B,) soft-DTW values.
    """
    B, N, M = D.shape
    if lengths_x is None:
        lengths_x = torch.full((B,), N, dtype=torch.long, device=D.device)
    if lengths_y is None:
        lengths_y = torch.full((B,), M, dtype=torch.long, device=D.device)
    return SoftDTWAutograd.apply(D, lengths_x, lengths_y, gamma, bandwidth)


class SoftDTW(nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        bandwidth: int = -1,
        normalize: bool = False,
        distance: str = "l2_squared",
    ):
        """
        Args:
            gamma: Smoothing parameter (> 0).
            bandwidth: Sakoe-Chiba bandwidth around the diagonal joining the first and last frames, in steps
                along the longer sequence. Floored at half the diagonal's slope so that a warping path always
                exists. -1 means no constraint.
            normalize: Divide the cost by path length.
            distance: Pairwise distance used to build the cost matrix from X, Y.
                One of "l2_squared", "l1", "cosine".
        """
        super().__init__()
        if distance not in PAIRWISE_DISTANCES:
            raise ValueError(f"Unknown distance {distance!r}, expected one of {sorted(PAIRWISE_DISTANCES)}")
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.distance = distance

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        lengths_x: torch.Tensor | None = None,
        lengths_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft-DTW between sequences X and Y.

        Args:
            X: (B, N, D) or (N, D) input sequences.
            Y: (B, M, D) or (M, D) target sequences.
            lengths_x: (B,) actual sequence lengths for X.
            lengths_y: (B,) actual sequence lengths for Y.

        Returns:
            costs: (B,) soft-DTW distances.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)

        B = X.size(0)

        D = PAIRWISE_DISTANCES[self.distance](X, Y)
        costs = soft_dtw(D, lengths_x, lengths_y, self.gamma, self.bandwidth)

        if not self.normalize:
            return costs

        if lengths_x is None:
            lengths_x = torch.full((B,), X.size(1), dtype=torch.long, device=X.device)
        if lengths_y is None:
            lengths_y = torch.full((B,), Y.size(1), dtype=torch.long, device=Y.device)

        return costs / (lengths_x + lengths_y - 1).to(costs.dtype)

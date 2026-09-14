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


# ---- Reference implementations (for testing) ----


def _naive_forward(D, lengths_x, lengths_y, gamma, bandwidth):
    """Pure Python reference implementation for when C extension is unavailable."""
    B, N, M = D.shape
    INF = float("inf")
    R = torch.full((B, N + 2, M + 2), INF, dtype=D.dtype, device=D.device)
    R[:, 0, 0] = 0.0

    for b in range(B):
        nx = int(lengths_x[b].item())
        ny = int(lengths_y[b].item())
        for i in range(nx):
            for j in range(ny):
                if bandwidth >= 0 and abs(i - j) > bandwidth:
                    continue
                ri, rj = i + 1, j + 1
                predecessors = torch.tensor(
                    [R[b, ri - 1, rj - 1], R[b, ri - 1, rj], R[b, ri, rj - 1]],
                    dtype=D.dtype,
                    device=D.device,
                )
                neg_pred = -predecessors / gamma
                rmax = neg_pred.max()
                if torch.isinf(rmax) and rmax < 0:
                    softmin = INF
                else:
                    softmin = -gamma * (
                        torch.log(torch.exp(neg_pred - rmax).sum()) + rmax
                    )
                R[b, ri, rj] = D[b, i, j] + softmin

    costs = torch.stack(
        [R[b, int(lengths_x[b].item()), int(lengths_y[b].item())] for b in range(B)]
    )
    return costs, R


def _naive_backward(D, R, lengths_x, lengths_y, gamma, bandwidth):
    """Pure Python reference backward (log-space stable)."""
    B, N, M = D.shape
    NEG_INF = float("-inf")
    INF = float("inf")

    R_bw = R.clone()

    logE = torch.full((B, N + 2, M + 2), NEG_INF, dtype=D.dtype, device=D.device)

    for b in range(B):
        nx = int(lengths_x[b].item())
        ny = int(lengths_y[b].item())
        R_bw[b, nx + 1, ny + 1] = R[b, nx, ny]
        logE[b, nx + 1, ny + 1] = 0.0

        def safe_R(row, col):
            v = R_bw[b, row, col].item()
            return NEG_INF if v == INF else v

        def D_pad(row, col):
            if 1 <= row <= nx and 1 <= col <= ny:
                return D[b, row - 1, col - 1].item()
            return 0.0

        for i in range(nx - 1, -1, -1):
            for j in range(ny - 1, -1, -1):
                if bandwidth >= 0 and abs(i - j) > bandwidth:
                    continue
                ri, rj = i + 1, j + 1
                R_ij = safe_R(ri, rj)

                la = (safe_R(ri + 1, rj) - R_ij - D_pad(ri + 1, rj)) / gamma
                lb = (safe_R(ri, rj + 1) - R_ij - D_pad(ri, rj + 1)) / gamma
                lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma

                t1 = logE[b, ri + 1, rj].item() + la
                t2 = logE[b, ri, rj + 1].item() + lb
                t3 = logE[b, ri + 1, rj + 1].item() + lc

                m = max(t1, t2, t3)
                if m == NEG_INF:
                    logE[b, ri, rj] = NEG_INF
                else:
                    import math

                    logE[b, ri, rj] = m + math.log(
                        math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m)
                    )

    E = torch.exp(logE[:, 1 : N + 1, 1 : M + 1]).contiguous()

    for b in range(B):
        nx = int(lengths_x[b].item())
        ny = int(lengths_y[b].item())
        if nx < N:
            E[b, nx:, :] = 0
        if ny < M:
            E[b, :, ny:] = 0

    return E


# ---- Autograd Function ----


class SoftDTWAutograd(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, D, lengths_x, lengths_y, gamma, bandwidth):
        costs, R = torch.ops.torchsoftdtw.forward(
            D, lengths_x, lengths_y, gamma, bandwidth
        )
        ctx.save_for_backward(D, R, lengths_x, lengths_y)
        ctx.gamma = gamma
        ctx.bandwidth = bandwidth
        return costs

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        D, R, lengths_x, lengths_y = ctx.saved_tensors
        E = torch.ops.torchsoftdtw.backward(
            D, R, lengths_x, lengths_y, ctx.gamma, ctx.bandwidth
        )
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
        bandwidth: Sakoe-Chiba bandwidth. -1 means no constraint.

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
            bandwidth: Sakoe-Chiba bandwidth. -1 means no constraint.
            normalize: Divide the cost by lengths_y (or Y's sequence length).
            distance: Pairwise distance used to build the cost matrix from X, Y.
                One of "l2_squared", "l1", "cosine".
        """
        super().__init__()
        if distance not in PAIRWISE_DISTANCES:
            raise ValueError(
                f"Unknown distance {distance!r}, expected one of "
                f"{sorted(PAIRWISE_DISTANCES)}"
            )
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

        if lengths_y is None:
            lengths_y = torch.full((B,), Y.size(1), dtype=torch.long, device=Y.device)

        return costs / lengths_y.to(costs.dtype)

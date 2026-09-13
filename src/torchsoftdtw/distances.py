import torch


def pairwise_l2_squared(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        X: (B, N, D)
        Y: (B, M, D)

    Returns:
        D: (B, N, M) where D[b,i,j] = ||X[b,i] - Y[b,j]||^2
    """
    x2 = (X * X).sum(dim=-1)  # (B, N)
    y2 = (Y * Y).sum(dim=-1)  # (B, M)
    xy = torch.bmm(X, Y.transpose(1, 2))  # (B, N, M)
    D = x2.unsqueeze(2) + y2.unsqueeze(1) - 2 * xy
    return D.clamp_min(0.0)

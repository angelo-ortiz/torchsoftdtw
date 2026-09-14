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


def pairwise_l1(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L1 (Manhattan) distances.

    Args:
        X: (B, N, D)
        Y: (B, M, D)

    Returns:
        D: (B, N, M) where D[b,i,j] = sum_d |X[b,i,d] - Y[b,j,d]|
    """
    return (X.unsqueeze(2) - Y.unsqueeze(1)).abs().sum(dim=-1)


def pairwise_cosine(
    X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute pairwise cosine distances (1 - cosine similarity).

    Args:
        X: (B, N, D)
        Y: (B, M, D)
        eps: Numerical stability floor for the norms.

    Returns:
        D: (B, N, M) where D[b,i,j] = 1 - cos_sim(X[b,i], Y[b,j])
    """
    x_norm = X.norm(dim=-1, keepdim=True).clamp_min(eps)  # (B, N, 1)
    y_norm = Y.norm(dim=-1, keepdim=True).clamp_min(eps)  # (B, M, 1)
    Xn = X / x_norm
    Yn = Y / y_norm
    sim = torch.bmm(Xn, Yn.transpose(1, 2))  # (B, N, M)
    return 1.0 - sim


PAIRWISE_DISTANCES = {
    "l2_squared": pairwise_l2_squared,
    "l1": pairwise_l1,
    "cosine": pairwise_cosine,
}

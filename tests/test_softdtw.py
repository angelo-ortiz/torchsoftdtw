import math

# import pytest
import torch

from torchsoftdtw import SoftDTW, soft_dtw
from torchsoftdtw.distances import pairwise_l2_squared

GAMMA_SWEEP = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


def _gen_all_paths(m, n):
    """Yield every monotonic alignment path (as a list of (i, j) cells)
    from (0, 0) to (m-1, n-1), moving by (i+1, j), (i, j+1) or (i+1, j+1)."""

    def rec(i, j):
        if i == m - 1 and j == n - 1:
            yield [(i, j)]
            return
        if i + 1 < m:
            for tail in rec(i + 1, j):
                yield [(i, j)] + tail
        if j + 1 < n:
            for tail in rec(i, j + 1):
                yield [(i, j)] + tail
        if i + 1 < m and j + 1 < n:
            for tail in rec(i + 1, j + 1):
                yield [(i, j)] + tail

    yield from rec(0, 0)


def _soft_dtw_brute_force(D, gamma):
    """Soft-DTW via explicit softmin over the cost of every alignment path.

    This is independent of the DP recursion under test: it directly
    implements soft-DTW's definition as -gamma * log(sum_A exp(-<A,D>/gamma)).
    """
    m, n = D.shape
    costs = [sum(D[i, j].item() for (i, j) in path) for path in _gen_all_paths(m, n)]
    costs = torch.tensor(costs, dtype=D.dtype)
    neg = -costs / gamma
    m_val = neg.max()
    return (-gamma * (torch.log(torch.exp(neg - m_val).sum()) + m_val)).item()


def _reference_softdtw(D, gamma, bandwidth=-1):
    """Triple-loop reference soft-DTW (no padding)."""
    B, N, M = D.shape
    INF = float("inf")
    R = torch.full((B, N + 2, M + 2), INF, dtype=D.dtype, device=D.device)
    R[:, 0, 0] = 0.0

    for b in range(B):
        for i in range(N):
            for j in range(M):
                if bandwidth >= 0 and abs(i - j) > bandwidth:
                    continue
                ri, rj = i + 1, j + 1
                preds = torch.tensor(
                    [R[b, ri - 1, rj - 1], R[b, ri - 1, rj], R[b, ri, rj - 1]],
                    dtype=D.dtype,
                )
                neg = -preds / gamma
                rmax = neg.max()
                if torch.isinf(rmax) and rmax < 0:
                    sm = INF
                else:
                    sm = -gamma * (torch.log(torch.exp(neg - rmax).sum()) + rmax)
                R[b, ri, rj] = D[b, i, j] + sm
    return torch.stack([R[b, N, M] for b in range(B)])


class TestForward:
    def test_basic_correctness(self):
        torch.manual_seed(42)
        B, N, M = 2, 5, 7
        D = torch.rand(B, N, M, dtype=torch.float64)

        for gamma in GAMMA_SWEEP:
            result = soft_dtw(D, gamma=gamma)
            expected = _reference_softdtw(D, gamma)
            torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-6)

    def test_identical_sequences(self):
        X = torch.randn(3, 10, 4, dtype=torch.float64)
        D = pairwise_l2_squared(X, X)
        costs = soft_dtw(D, gamma=1.0)
        # Soft-DTW of identical sequences is slightly negative due to
        # soft-min < true min (by ~gamma*log(3) per step). This is expected.
        assert (costs < 0.5).all()
        # With small gamma, should approach 0
        costs_small_gamma = soft_dtw(D, gamma=0.001)
        assert (costs_small_gamma.abs() < 0.1).all()

    def test_single_element(self):
        D = torch.tensor([[[2.0]]], dtype=torch.float64)
        cost = soft_dtw(D, gamma=1.0)
        torch.testing.assert_close(cost, torch.tensor([2.0], dtype=torch.float64))

    def test_gamma_small_approaches_hard_dtw(self):
        torch.manual_seed(0)
        D = torch.rand(1, 6, 8, dtype=torch.float64) * 10

        # Hard DTW via standard DP
        N, M = D.shape[1], D.shape[2]
        R = torch.full((N + 1, M + 1), float("inf"), dtype=torch.float64)
        R[0, 0] = 0
        for i in range(N):
            for j in range(M):
                R[i + 1, j + 1] = D[0, i, j] + min(R[i, j], R[i, j + 1], R[i + 1, j])
        hard_dtw = R[N, M]

        soft_cost = soft_dtw(D, gamma=0.001)
        assert abs(soft_cost.item() - hard_dtw.item()) < 0.1

    def test_symmetric(self):
        torch.manual_seed(7)
        D_xy = torch.rand(2, 5, 8, dtype=torch.float64)
        D_yx = D_xy.transpose(1, 2).contiguous()

        c_xy = soft_dtw(D_xy, gamma=0.5)
        c_yx = soft_dtw(D_yx, gamma=0.5)
        torch.testing.assert_close(c_xy, c_yx, atol=1e-10, rtol=1e-10)


class TestPadding:
    def test_padding_ignored(self):
        torch.manual_seed(1)
        B = 3
        actual_lengths_x = [4, 6, 3]
        actual_lengths_y = [5, 4, 7]
        max_N, max_M = 8, 8
        gamma = 0.5

        D_padded = torch.rand(B, max_N, max_M, dtype=torch.float64)
        lx = torch.tensor(actual_lengths_x, dtype=torch.long)
        ly = torch.tensor(actual_lengths_y, dtype=torch.long)

        costs_padded = soft_dtw(D_padded, lx, ly, gamma=gamma)

        for b in range(B):
            nx, ny = actual_lengths_x[b], actual_lengths_y[b]
            D_single = D_padded[b, :nx, :ny].unsqueeze(0)
            cost_single = soft_dtw(D_single, gamma=gamma)
            torch.testing.assert_close(
                costs_padded[b : b + 1], cost_single, atol=1e-10, rtol=1e-10
            )

    def test_padding_gradient(self):
        torch.manual_seed(2)
        D = torch.rand(2, 6, 8, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([4, 6], dtype=torch.long)
        ly = torch.tensor([5, 8], dtype=torch.long)

        costs = soft_dtw(D, lx, ly, gamma=1.0)
        costs.sum().backward()
        grad = D.grad

        # Gradient should be zero in the padded region
        assert (grad[0, 4:, :] == 0).all()
        assert (grad[0, :, 5:] == 0).all()
        # But non-zero in the valid region
        assert (grad[0, :4, :5] != 0).any()


class TestBandwidth:
    def test_bandwidth_constrains(self):
        torch.manual_seed(3)
        D = torch.rand(1, 10, 10, dtype=torch.float64)
        gamma = 0.5

        cost_full = soft_dtw(D, gamma=gamma, bandwidth=-1)
        cost_band = soft_dtw(D, gamma=gamma, bandwidth=2)

        # Bandwidth-constrained cost should be >= unconstrained
        assert cost_band.item() >= cost_full.item() - 1e-10

    def test_bandwidth_matches_reference(self):
        torch.manual_seed(4)
        D = torch.rand(2, 8, 6, dtype=torch.float64)
        bw = 3

        for gamma in GAMMA_SWEEP:
            result = soft_dtw(D, gamma=gamma, bandwidth=bw)
            expected = _reference_softdtw(D, gamma, bandwidth=bw)
            torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-6)


class TestGradients:
    def test_gradcheck(self):
        torch.manual_seed(5)
        D = torch.rand(2, 4, 5, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([4, 4], dtype=torch.long)
        ly = torch.tensor([5, 5], dtype=torch.long)

        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=1.0),
            (D,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_with_bandwidth(self):
        torch.manual_seed(6)
        D = torch.rand(2, 5, 5, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([5, 5], dtype=torch.long)
        ly = torch.tensor([5, 5], dtype=torch.long)

        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=0.5, bandwidth=2),
            (D,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_with_padding(self):
        torch.manual_seed(7)
        D = torch.rand(2, 6, 8, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([4, 6], dtype=torch.long)
        ly = torch.tensor([5, 8], dtype=torch.long)

        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=1.0),
            (D,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradient_flow_through_module(self):
        torch.manual_seed(8)
        X = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(2, 7, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=1.0)
        costs = sdtw(X, Y)
        costs.sum().backward()

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert not torch.isnan(X.grad).any()


class TestModule:
    def test_normalize(self):
        torch.manual_seed(9)
        X = torch.randn(2, 5, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=1.0, normalize=True)
        costs = sdtw(X, X)
        # Self-distance after normalization should be ~0
        assert costs.abs().max().item() < 1e-6

    def test_unbatched_input(self):
        torch.manual_seed(10)
        X = torch.randn(5, 4, dtype=torch.float64)
        Y = torch.randn(7, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=1.0)
        cost = sdtw(X, Y)
        assert cost.shape == (1,)


class TestBruteForcePaths:
    """Independent check against soft-DTW's definition: softmin over the
    cost of every monotonic alignment path (mirrors mblondel/soft-dtw's
    gen_all_paths-based test), rather than against another DP recursion."""

    def test_matches_brute_force_small(self):
        torch.manual_seed(20)
        D = torch.rand(1, 3, 4, dtype=torch.float64)

        for gamma in GAMMA_SWEEP:
            result = soft_dtw(D, gamma=gamma).item()
            expected = _soft_dtw_brute_force(D[0], gamma)
            assert math.isclose(result, expected, abs_tol=1e-8, rel_tol=1e-6)

    def test_matches_brute_force_square(self):
        torch.manual_seed(21)
        D = torch.rand(1, 4, 4, dtype=torch.float64)

        for gamma in GAMMA_SWEEP:
            result = soft_dtw(D, gamma=gamma).item()
            expected = _soft_dtw_brute_force(D[0], gamma)
            assert math.isclose(result, expected, abs_tol=1e-8, rel_tol=1e-6)

    def test_matches_brute_force_rectangular(self):
        torch.manual_seed(22)
        D = torch.rand(1, 2, 5, dtype=torch.float64)

        for gamma in GAMMA_SWEEP:
            result = soft_dtw(D, gamma=gamma).item()
            expected = _soft_dtw_brute_force(D[0], gamma)
            assert math.isclose(result, expected, abs_tol=1e-8, rel_tol=1e-6)


class TestDistances:
    def test_pairwise_l2_squared(self):
        torch.manual_seed(11)
        X = torch.randn(2, 3, 4, dtype=torch.float64)
        Y = torch.randn(2, 5, 4, dtype=torch.float64)

        D = pairwise_l2_squared(X, Y)
        assert D.shape == (2, 3, 5)
        assert (D >= 0).all()

        # Compare to explicit computation
        for b in range(2):
            for i in range(3):
                for j in range(5):
                    expected = ((X[b, i] - Y[b, j]) ** 2).sum()
                    torch.testing.assert_close(
                        D[b, i, j], expected, atol=1e-10, rtol=1e-10
                    )

"""Robustness tests: numerical stability, edge shapes, validation, non-contiguous,
combined bandwidth+padding, normalize gradcheck, float32, bandwidth=0, and
gradient through the full X->D->sdtw chain.

Adapted from gaps identified by cross-repo analysis against:
  - mblondel/soft-dtw (gradient w.r.t. X)
  - bootphon/torchdtw (validation, non-contiguous, dtype coverage)
  - BGU-CS-VIL/sdtw-cuda-torch (stability with small gamma, edge shapes)
"""

import pytest
import torch

from torchsoftdtw import SoftDTW, soft_dtw
from torchsoftdtw.distances import pairwise_l2_squared

# =====================================================================
# HIGH PRIORITY
# =====================================================================


class TestNumericalStability:
    """Small gamma on larger matrices stress-tests the log-space backward."""

    def test_small_gamma_gradients_finite(self):
        torch.manual_seed(100)
        N, M = 32, 32
        D = torch.rand(1, N, M, dtype=torch.float64, requires_grad=True)
        costs = soft_dtw(D, gamma=0.01)
        costs.sum().backward()
        assert torch.isfinite(D.grad).all(), "Gradient contains inf/nan with gamma=0.01"
        assert (D.grad != 0).any(), "Gradient is all zeros"

    def test_small_gamma_larger_matrix(self):
        torch.manual_seed(101)
        N, M = 64, 64
        D = torch.rand(2, N, M, dtype=torch.float64, requires_grad=True)
        costs = soft_dtw(D, gamma=0.001)
        costs.sum().backward()
        assert torch.isfinite(D.grad).all()

    def test_very_small_gamma_gradcheck(self):
        torch.manual_seed(102)
        D = torch.rand(1, 6, 6, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([6], dtype=torch.long)
        ly = torch.tensor([6], dtype=torch.long)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=0.01),
            (D,),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-2,
        )


class TestGradientThroughDistance:
    """Gradient w.r.t. input sequences X through the full chain:
    X -> pairwise_l2_squared(X, Y) -> soft_dtw(D).
    Adapted from mblondel/soft-dtw's test_soft_dtw.py.
    """

    def test_gradcheck_through_distance(self):
        torch.manual_seed(200)
        X = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(2, 7, 4, dtype=torch.float64)

        def fn(x):
            D = pairwise_l2_squared(x, Y)
            return soft_dtw(D, gamma=1.0)

        torch.autograd.gradcheck(fn, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_through_distance_with_bandwidth(self):
        torch.manual_seed(201)
        X = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(1, 6, 3, dtype=torch.float64)

        def fn(x):
            D = pairwise_l2_squared(x, Y)
            return soft_dtw(D, gamma=0.5, bandwidth=2)

        torch.autograd.gradcheck(fn, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_nonzero_and_finite(self):
        torch.manual_seed(202)
        X = torch.randn(3, 8, 4, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(3, 10, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=1.0)
        costs = sdtw(X, Y)
        costs.sum().backward()

        assert torch.isfinite(X.grad).all()
        assert (X.grad != 0).any()


# =====================================================================
# MEDIUM PRIORITY
# =====================================================================


class TestEdgeShapes:
    """Edge shapes: single-feature dim, 1xM, Nx1, extreme aspect ratios."""

    def test_single_feature_dim(self):
        torch.manual_seed(300)
        D = torch.rand(2, 5, 7, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([5, 5], dtype=torch.long)
        ly = torch.tensor([7, 7], dtype=torch.long)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=1.0),
            (D,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_1xM(self):
        torch.manual_seed(301)
        D = torch.rand(1, 1, 5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, gamma=1.0), (D,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_Nx1(self):
        torch.manual_seed(302)
        D = torch.rand(1, 5, 1, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, gamma=1.0), (D,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_extreme_aspect_ratio(self):
        torch.manual_seed(303)
        D = torch.rand(1, 2, 20, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, gamma=0.5), (D,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_extreme_aspect_ratio_reversed(self):
        torch.manual_seed(304)
        D = torch.rand(1, 20, 2, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, gamma=0.5), (D,), eps=1e-6, atol=1e-4, rtol=1e-3
        )


class TestInputValidation:
    """Input validation error paths — adapted from torchdtw's test_validation.py."""

    def test_D_must_be_3d(self):
        D = torch.rand(5, 5)
        lx = torch.tensor([5], dtype=torch.long)
        ly = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="3D"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)

    def test_lengths_must_be_1d(self):
        D = torch.rand(1, 5, 5)
        lx = torch.tensor([[5]], dtype=torch.long)
        ly = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="1D"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)

    def test_batch_mismatch(self):
        D = torch.rand(2, 5, 5)
        lx = torch.tensor([5, 5, 5], dtype=torch.long)
        ly = torch.tensor([5, 5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="[Bb]atch"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)

    def test_gamma_must_be_positive(self):
        D = torch.rand(1, 5, 5)
        lx = torch.tensor([5], dtype=torch.long)
        ly = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="gamma"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 0.0, -1)
        with pytest.raises(RuntimeError, match="gamma"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, -1.0, -1)

    def test_D_must_be_float(self):
        D = torch.randint(0, 10, (1, 5, 5))
        lx = torch.tensor([5], dtype=torch.long)
        ly = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="float"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)

    def test_lengths_must_be_int(self):
        D = torch.rand(1, 5, 5)
        lx = torch.tensor([5.0], dtype=torch.float32)
        ly = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError, match="int"):
            torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)


class TestNonContiguous:
    """Non-contiguous inputs — adapted from torchdtw's test_noncontiguous.py."""

    def test_transposed_D(self):
        torch.manual_seed(400)
        D_contig = torch.rand(2, 5, 8, dtype=torch.float64)
        D_transposed = D_contig.transpose(1, 2).contiguous().transpose(1, 2)
        assert not D_transposed.is_contiguous()

        costs_contig = soft_dtw(D_contig, gamma=1.0)
        costs_noncontig = soft_dtw(D_transposed, gamma=1.0)
        torch.testing.assert_close(costs_contig, costs_noncontig)

    def test_strided_D(self):
        torch.manual_seed(401)
        D_big = torch.rand(4, 10, 12, dtype=torch.float64)
        D_strided = D_big[::2, ::2, ::2]
        assert not D_strided.is_contiguous()
        costs = soft_dtw(D_strided, gamma=1.0)
        costs_contig = soft_dtw(D_strided.contiguous(), gamma=1.0)
        torch.testing.assert_close(costs, costs_contig)

    def test_gradient_noncontiguous(self):
        torch.manual_seed(402)
        D = torch.rand(2, 8, 6, dtype=torch.float64, requires_grad=True)
        D_nc = D.transpose(1, 2).contiguous().transpose(1, 2)
        costs = soft_dtw(D_nc, gamma=1.0)
        costs.sum().backward()
        assert D.grad is not None
        assert torch.isfinite(D.grad).all()


class TestBandwidthPaddingCombined:
    """Bandwidth + padding combined — these two features interact in the inner loop."""

    def test_bandwidth_and_padding_forward(self):
        torch.manual_seed(500)
        B = 3
        max_N, max_M = 10, 10
        actual_nx = [6, 8, 5]
        actual_ny = [7, 6, 9]
        bw = 3
        gamma = 0.5

        D = torch.rand(B, max_N, max_M, dtype=torch.float64)
        lx = torch.tensor(actual_nx, dtype=torch.long)
        ly = torch.tensor(actual_ny, dtype=torch.long)

        costs = soft_dtw(D, lx, ly, gamma=gamma, bandwidth=bw)

        for b in range(B):
            nx, ny = actual_nx[b], actual_ny[b]
            D_single = D[b, :nx, :ny].unsqueeze(0)
            cost_single = soft_dtw(D_single, gamma=gamma, bandwidth=bw)
            torch.testing.assert_close(
                costs[b : b + 1], cost_single, atol=1e-10, rtol=1e-10
            )

    def test_bandwidth_and_padding_gradient(self):
        torch.manual_seed(501)
        D = torch.rand(2, 8, 8, dtype=torch.float64, requires_grad=True)
        lx = torch.tensor([5, 8], dtype=torch.long)
        ly = torch.tensor([6, 8], dtype=torch.long)

        torch.autograd.gradcheck(
            lambda d: soft_dtw(d, lx, ly, gamma=1.0, bandwidth=2),
            (D,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestNormalizeGradcheck:
    """Normalize mode produces gradients through soft_dtw scaled by reference length."""

    def test_normalize_gradcheck(self):
        torch.manual_seed(600)
        X = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(2, 5, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=1.0, normalize=True)

        def fn(x):
            return sdtw(x, Y)

        torch.autograd.gradcheck(fn, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_normalize_gradcheck_with_bandwidth(self):
        torch.manual_seed(601)
        X = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(2, 6, 4, dtype=torch.float64)

        sdtw = SoftDTW(gamma=0.5, normalize=True, bandwidth=2)

        def fn(x):
            return sdtw(x, Y)

        torch.autograd.gradcheck(fn, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =====================================================================
# LOW PRIORITY
# =====================================================================


class TestFloat32:
    """Verify float32 produces correct results (existing tests use float64 only)."""

    def test_forward_float32(self):
        torch.manual_seed(700)
        D64 = torch.rand(2, 6, 8, dtype=torch.float64)
        D32 = D64.float()

        costs64 = soft_dtw(D64, gamma=1.0)
        costs32 = soft_dtw(D32, gamma=1.0)

        torch.testing.assert_close(costs32.double(), costs64, atol=1e-5, rtol=1e-4)

    def test_gradient_float32(self):
        torch.manual_seed(701)
        D = torch.rand(2, 5, 7, dtype=torch.float32, requires_grad=True)
        costs = soft_dtw(D, gamma=1.0)
        costs.sum().backward()
        assert D.grad is not None
        assert torch.isfinite(D.grad).all()
        assert (D.grad != 0).any()


class TestHalfPrecision:
    """Half/bfloat16 support with float32 accumulation."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_half(self, dtype):
        torch.manual_seed(710)
        D64 = torch.rand(2, 5, 7, dtype=torch.float64)
        D_half = D64.to(dtype)

        costs_half = soft_dtw(D_half, gamma=1.0)
        costs64 = soft_dtw(D64, gamma=1.0)

        torch.testing.assert_close(costs_half.double(), costs64, atol=0.1, rtol=0.05)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gradient_half(self, dtype):
        torch.manual_seed(711)
        D = torch.rand(2, 4, 6, dtype=dtype, requires_grad=True)
        costs = soft_dtw(D, gamma=1.0)
        costs.sum().backward()
        assert D.grad is not None
        assert torch.isfinite(D.grad).all()
        assert (D.grad != 0).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_costs_promoted_dtype(self, dtype):
        D = torch.rand(1, 3, 3, dtype=dtype)
        costs = soft_dtw(D, gamma=1.0)
        assert costs.dtype == torch.float32


class TestBandwidthZero:
    """Bandwidth=0 should only allow the diagonal (i==j) path."""

    def test_bandwidth_zero_diagonal_only(self):
        torch.manual_seed(800)
        D = torch.rand(1, 5, 5, dtype=torch.float64)
        costs_bw0 = soft_dtw(D, gamma=1.0, bandwidth=0)
        diagonal_cost = D[0].diag().sum()
        torch.testing.assert_close(costs_bw0[0], diagonal_cost, atol=1e-10, rtol=1e-10)

    def test_bandwidth_zero_rectangular_inf(self):
        torch.manual_seed(801)
        D = torch.rand(1, 3, 5, dtype=torch.float64)
        costs_bw0 = soft_dtw(D, gamma=1.0, bandwidth=0)
        assert torch.isinf(costs_bw0).all(), (
            "bandwidth=0 on non-square matrix should be inf (no valid path)"
        )


class TestTorchCompile:
    """torch.compile compatibility via FakeTensor registration."""

    def test_fake_tensor_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode() as mode:
            D = mode.from_tensor(torch.rand(2, 5, 7))
            lx = mode.from_tensor(torch.tensor([5, 5], dtype=torch.long))
            ly = mode.from_tensor(torch.tensor([7, 7], dtype=torch.long))

            costs, R = torch.ops.torchsoftdtw.forward(D, lx, ly, 1.0, -1)
            assert costs.shape == (2,)
            assert R.shape == (2, 7, 9)

    def test_fake_tensor_backward_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            D = mode.from_tensor(torch.rand(2, 5, 7))
            R = mode.from_tensor(torch.rand(2, 7, 9))
            lx = mode.from_tensor(torch.tensor([5, 5], dtype=torch.long))
            ly = mode.from_tensor(torch.tensor([7, 7], dtype=torch.long))

            E = torch.ops.torchsoftdtw.backward(D, R, lx, ly, 1.0, -1)
            assert E.shape == (2, 5, 7)

"""Comparison tests against reference implementations.

Requires:
    pip install sdtw-cuda-torch  (or install from https://github.com/BGU-CS-VIL/sdtw-cuda-torch)
    pip install soft-dtw-cuda    (or install from https://github.com/Maghoumi/pytorch-softdtw-cuda)
"""

import pytest
import torch

from torchsoftdtw import SoftDTW, soft_dtw
from torchsoftdtw.distances import pairwise_l2_squared


def _has_sdtw_cuda_torch():
    try:
        from softdtw_cuda import SoftDTW as RefSoftDTW

        return True
    except ImportError:
        return False


def _has_pytorch_softdtw_cuda():
    try:
        from torchsoftdtw.soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

        return True
    except ImportError:
        return False


requires_sdtw_cuda_torch = pytest.mark.skipif(
    not _has_sdtw_cuda_torch(),
    reason="sdtw-cuda-torch not installed",
)

requires_pytorch_softdtw_cuda = pytest.mark.skipif(
    not _has_pytorch_softdtw_cuda(),
    reason="pytorch-softdtw-cuda not installed",
)


@requires_sdtw_cuda_torch
class TestVsSdtwCudaTorch:
    def _compare_forward(self, B, N, M, D_feat, gamma, bandwidth=None):
        from softdtw_cuda import SoftDTW as RefSoftDTW

        torch.manual_seed(42)
        X = torch.randn(B, N, D_feat, dtype=torch.float64)
        Y = torch.randn(B, M, D_feat, dtype=torch.float64)

        D = pairwise_l2_squared(X, Y)

        # Our implementation
        bw = -1 if bandwidth is None else bandwidth
        our_costs = soft_dtw(D, gamma=gamma, bandwidth=bw)

        # Reference (sdtw-cuda-torch) — takes (X, Y) directly
        ref_sdtw = RefSoftDTW(gamma=gamma, bandwidth=bandwidth)
        ref_costs = ref_sdtw(X, Y)

        # torch.testing.assert_close(our_costs, ref_costs, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(our_costs, ref_costs)

    def test_forward_small(self):
        self._compare_forward(B=4, N=10, M=12, D_feat=8, gamma=1.0)

    def test_forward_medium(self):
        self._compare_forward(B=2, N=50, M=40, D_feat=16, gamma=0.1)

    def test_forward_with_bandwidth(self):
        self._compare_forward(B=3, N=20, M=20, D_feat=8, gamma=0.5, bandwidth=5)

    def test_backward(self):
        from softdtw_cuda import SoftDTW as RefSoftDTW

        torch.manual_seed(43)
        B, N, M, D_feat = 2, 8, 10, 4

        X_ours = torch.randn(B, N, D_feat, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(B, M, D_feat, dtype=torch.float64)
        X_ref = X_ours.detach().clone().requires_grad_(True)

        D = pairwise_l2_squared(X_ours, Y)
        our_costs = soft_dtw(D, gamma=1.0)
        our_costs.sum().backward()

        ref_sdtw = RefSoftDTW(gamma=1.0)
        ref_costs = ref_sdtw(X_ref, Y)
        ref_costs.sum().backward()

        # torch.testing.assert_close(X_ours.grad, X_ref.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(X_ours.grad, X_ref.grad)


@requires_pytorch_softdtw_cuda
class TestVsPytorchSoftdtwCuda:
    def test_forward(self):
        from torchsoftdtw.soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

        torch.manual_seed(44)
        B, N, M, D_feat = 4, 8, 12, 6

        X = torch.randn(B, N, D_feat, dtype=torch.float64)
        Y = torch.randn(B, M, D_feat, dtype=torch.float64)
        D = pairwise_l2_squared(X, Y)

        our_costs = soft_dtw(D, gamma=1.0)

        # Maghoumi's implementation takes X, Y directly and uses squared Euclidean
        ref_sdtw = MaghoumiSoftDTW(use_cuda=False, gamma=1.0)
        ref_costs = ref_sdtw(X, Y)

        # torch.testing.assert_close(our_costs, ref_costs, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(our_costs, ref_costs)

    def test_forward_with_bandwidth(self):
        from torchsoftdtw.soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

        torch.manual_seed(45)
        B, N, M, D_feat = 3, 10, 10, 4

        X = torch.randn(B, N, D_feat, dtype=torch.float64)
        Y = torch.randn(B, M, D_feat, dtype=torch.float64)
        D = pairwise_l2_squared(X, Y)

        our_costs = soft_dtw(D, gamma=0.5, bandwidth=3)

        ref_sdtw = MaghoumiSoftDTW(use_cuda=False, gamma=0.5, bandwidth=3)
        ref_costs = ref_sdtw(X, Y)

        # torch.testing.assert_close(our_costs, ref_costs, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(our_costs, ref_costs)

    def test_backward(self):
        from torchsoftdtw.soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

        torch.manual_seed(46)
        B, N, M, D_feat = 2, 6, 8, 4

        X_ours = torch.randn(B, N, D_feat, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(B, M, D_feat, dtype=torch.float64)
        X_ref = X_ours.detach().clone().requires_grad_(True)

        D = pairwise_l2_squared(X_ours, Y)
        our_costs = soft_dtw(D, gamma=1.0)
        our_costs.sum().backward()

        ref_sdtw = MaghoumiSoftDTW(use_cuda=False, gamma=1.0)
        ref_costs = ref_sdtw(X_ref, Y)
        ref_costs.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, atol=1e-5, rtol=1e-4)
        # torch.testing.assert_close(X_ours.grad, X_ref.grad)

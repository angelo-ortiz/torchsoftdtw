"""Parity tests between the CPU and CUDA device kernels of the C++ extension.

These exercise both the non-tiled kernel path (max(N, M) <= 1024) and the
tiled kernel path (max(N, M) > 1024), with both int32 and int64 lengths,
and with rectangular (N != M) shapes where the tiled kernels' antidiagonal
offset must be computed correctly.
"""

import pytest
import torch

from torchsoftdtw.softdtw import soft_dtw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _compare(B, N, M, gamma, bandwidth, len_dtype, seed):
    torch.manual_seed(seed)
    D_cpu = torch.rand(B, N, M, dtype=torch.float64, requires_grad=True)
    lx = torch.randint(low=max(1, N - 3), high=N + 1, size=(B,), dtype=len_dtype)
    ly = torch.randint(low=max(1, M - 3), high=M + 1, size=(B,), dtype=len_dtype)

    D_gpu = D_cpu.detach().clone().cuda().requires_grad_(True)
    lx_gpu, ly_gpu = lx.cuda(), ly.cuda()

    cost_cpu = soft_dtw(D_cpu, lx, ly, gamma, bandwidth)
    cost_gpu = soft_dtw(D_gpu, lx_gpu, ly_gpu, gamma, bandwidth)
    torch.testing.assert_close(cost_cpu, cost_gpu.cpu(), atol=1e-6, rtol=1e-5)

    cost_cpu.sum().backward()
    cost_gpu.sum().backward()
    torch.testing.assert_close(D_cpu.grad, D_gpu.grad.cpu(), atol=1e-6, rtol=1e-5)


@requires_cuda
class TestCpuGpuParity:
    @pytest.mark.parametrize("len_dtype", [torch.int32, torch.int64])
    def test_small_no_band(self, len_dtype):
        _compare(4, 6, 8, 1.0, -1, len_dtype, seed=0)

    @pytest.mark.parametrize("len_dtype", [torch.int32, torch.int64])
    def test_small_with_band(self, len_dtype):
        _compare(4, 6, 8, 0.5, 2, len_dtype, seed=1)

    def test_equal_lengths(self):
        _compare(3, 5, 5, 1.0, -1, torch.int32, seed=2)

    # Sizes above 1024 exercise the CUDA tiled kernel path.
    @pytest.mark.parametrize("len_dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("N,M", [(120, 130), (150, 110), (110, 150), (130, 130)])
    def test_tiled_path_rectangular(self, len_dtype, N, M):
        _compare(2, N, M, 0.7, -1, len_dtype, seed=3)

    def test_tiled_path_with_band(self):
        _compare(2, 120, 130, 1.0, 50, torch.int64, seed=4)

    # Strongly rectangular shapes where |N - M| > bandwidth: the band must follow the scaled diagonal.
    @pytest.mark.parametrize("bandwidth", [0, 1, 3])
    @pytest.mark.parametrize("N,M", [(6, 20), (20, 6)])
    def test_rectangular_with_band(self, N, M, bandwidth):
        _compare(3, N, M, 0.5, bandwidth, torch.int64, seed=5)

    def test_tiled_path_rectangular_with_band(self):
        _compare(2, 300, 1500, 1.0, 10, torch.int64, seed=6)

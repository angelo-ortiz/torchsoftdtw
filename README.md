# PyTorch soft-DTW C++ extension

A fast Soft-DTW implementation for PyTorch, with native CPU and CUDA kernels exposed through a custom `torch.ops`
extension.

```python
import torch
from torchsoftdtw import SoftDTW

sdtw = SoftDTW(gamma=1.0, normalize=True)
X = torch.randn(8, 50, 16, device="cuda")
Y = torch.randn(8, 40, 16, device="cuda")
loss = sdtw(X, Y)  # (8,)
loss.sum().backward()
```

## Features

- **C++/CUDA extension** built against PyTorch's stable ABI (`torch/csrc/stable`), registered as proper `torch.ops`
  custom ops with `FakeTensor` kernels so forward and backward both work under `torch.compile`.
- **Variable-length batches**: `lengths_x` / `lengths_y` let you pack ragged sequences into a single padded batch
  without wasting computation or leaking gradient through padding.
- **Sakoe-Chiba band** support via a `bandwidth` argument.
- **Length-normalized Soft-DTW** (`sdtw(X,Y) / len(Y)`), dividing the cost by the reference sequence length.
- Pure-Python reference forward/backward included for correctness testing when the compiled extension isn't available.

## Testing

`tests/test_comparison.py` and the benchmark suite optionally compare against `sdtw-cuda-torch` and
`pytorch-softdtw-cuda` for numerical parity and performance; see their docstrings for the extra dependencies required.

## Acknowledgements

This project builds on ideas from three earlier Soft-DTW/PyTorch implementations:

- **[Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda)** — the original CUDA Soft-DTW
  for PyTorch, using Numba to JIT-compile the diagonal wavefront kernel. `torchsoftdtw` keeps its vendored copy around
  (`torchsoftdtw/soft_dtw_cuda.py`) purely as a correctness baseline in tests.
- **[BGU-CS-VIL/sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch)** — a further-optimized fork of the
  above, also used here as a reference implementation for parity testing.
- **[bootphon/torchdtw](https://github.com/bootphon/torchdtw)** — a batched PyTorch DTW package, whose API shape (a
  thin `nn.Module` wrapping a differentiable batched distance) informed the interface design of `SoftDTW`.

`torchsoftdtw` is a from-scratch reimplementation, not a fork of any of the above. The main differences:

- Kernels are written directly in C++/CUDA against PyTorch's stable ABI and dispatched as registered `torch.ops` custom
  operators, rather than JIT-compiled with Numba. This avoids Numba as a runtime dependency and gives correct behavior
  under `torch.compile`/`FakeTensor` tracing and CUDA graphs.
- Native support for per-sample sequence lengths within a padded batch, rather than requiring uniform-length inputs.
- A Sakoe-Chiba band constraint implemented directly in the kernel (not a Python-side mask).
- The normalized variant divides `sdtw(X,Y)` by the reference sequence length instead of computing a symmetric
  divergence, so it works for batches with mixed sequence lengths.

## License

MIT. See [LICENSE](LICENSE). Portions of `soft_dtw_cuda.py` retain their original copyright notice (Mehran Maghoumi,
2020).

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>

// Not part of the public stable ABI surface, but the only way (pre PyTorch 2.13's
// Stream::nativeHandle()) to get the raw cudaStream_t backing the current stream,
// so that our raw kernel launches enqueue on the same stream PyTorch itself uses.
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace torchsoftdtw {

namespace stbl = torch::stable;

template <typename T, size_t N>
using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;

// See softdtw.cpp for why these take a single template argument (T) rather
// than the more natural accessor<T, N>: a literal comma inside "<...>"
// confuses THO_DISPATCH_V2's internal argument counting when it appears
// inside a dispatch body.
template <typename T>
inline TensorAccessor<T, 1> accessor1(stbl::Tensor t) {
    return TensorAccessor<T, 1>(
        reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename T>
inline TensorAccessor<T, 3> accessor3(stbl::Tensor t) {
    return TensorAccessor<T, 3>(
        reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

// lengths_x/lengths_y may arrive as int32 or int64 (checked in
// softdtw_forward_checks, softdtw.cpp). The kernels below only ever deal
// with int64_t lengths, so normalize once, up front, instead of templating
// every kernel over a second integer type.
inline stbl::Tensor lengths_to_int64(stbl::Tensor t) {
    if (t.scalar_type() == torch::headeronly::ScalarType::Long) return t;
    return stbl::to(t, torch::headeronly::ScalarType::Long);
}

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t logsumexp3(scalar_t a, scalar_t b, scalar_t c) {
    scalar_t m = fmax(fmax(a, b), c);
    if (isinf(m) && m < 0) return m;
    return m + log(exp(a - m) + exp(b - m) + exp(c - m));
}

// Forward kernel: one block per batch element, anti-diagonal wavefront
// Uses shared memory rotating buffers for fast predecessor reads
template <typename scalar_t>
__global__ void softdtw_forward_kernel(
    const scalar_t* __restrict__ D,  // (B, N, M)
    scalar_t* __restrict__ R,        // (B, N+2, M+2)
    scalar_t* __restrict__ costs,    // (B,)
    const int64_t* __restrict__ lengths_x,  // (B,)
    const int64_t* __restrict__ lengths_y,  // (B,)
    int N, int M,
    scalar_t gamma, int bandwidth)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nx = lengths_x[b];
    const int ny = lengths_y[b];
    const int R_M = M + 2;

    scalar_t* R_b = R + b * (N + 2) * R_M;
    const scalar_t* D_b = D + b * N * M;

    const scalar_t INF = INFINITY;

    // Initialize R to +inf, R[0][0] = 0
    for (int idx = tid; idx < (N + 2) * R_M; idx += blockDim.x) {
        R_b[idx] = INF;
    }
    __syncthreads();
    if (tid == 0) {
        R_b[0] = 0;
    }
    __syncthreads();

    // Anti-diagonal wavefront
    int n_passes = nx + ny - 1;
    for (int p = 0; p < n_passes; p++) {
        // Each thread handles one cell on this anti-diagonal
        // i + j == p, where i is the row (0-indexed into D), j is the column
        int i = tid;
        int j = p - tid;

        if (i >= 0 && i < nx && j >= 0 && j < ny) {
            bool in_band = (bandwidth < 0) || (abs(i - j) <= bandwidth);
            if (in_band) {
                int ri = i + 1;  // 1-indexed into R (padded)
                int rj = j + 1;

                scalar_t r0 = -R_b[(ri - 1) * R_M + (rj - 1)] / gamma;  // diag
                scalar_t r1 = -R_b[(ri - 1) * R_M + rj] / gamma;        // above
                scalar_t r2 = -R_b[ri * R_M + (rj - 1)] / gamma;        // left

                scalar_t rmax = fmax(fmax(r0, r1), r2);
                scalar_t softmin;
                if (isinf(rmax) && rmax < 0) {
                    softmin = INF;
                } else {
                    softmin = -gamma * (log(exp(r0 - rmax) + exp(r1 - rmax) + exp(r2 - rmax)) + rmax);
                }

                R_b[ri * R_M + rj] = D_b[i * M + j] + softmin;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        costs[b] = R_b[nx * R_M + ny];
    }
}

// Tiled forward kernel: one launch per anti-diagonal, for sequences > 1024
template <typename scalar_t>
__global__ void softdtw_forward_tiled_kernel(
    const scalar_t* __restrict__ D,
    scalar_t* __restrict__ R,
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M, int B,
    scalar_t gamma, int bandwidth,
    int p)  // current anti-diagonal index
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B) return;

    const int nx = lengths_x[b];
    const int ny = lengths_y[b];
    const int R_M = M + 2;

    const int i_lo = max(0, p - M + 1);
    const int i = i_lo + tid;
    const int j = p - i;

    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    if (bandwidth >= 0 && abs(i - j) > bandwidth) return;

    const scalar_t INF = INFINITY;
    scalar_t* R_b = R + b * (N + 2) * R_M;
    const scalar_t* D_b = D + b * N * M;

    int ri = i + 1;
    int rj = j + 1;

    scalar_t r0 = -R_b[(ri - 1) * R_M + (rj - 1)] / gamma;
    scalar_t r1 = -R_b[(ri - 1) * R_M + rj] / gamma;
    scalar_t r2 = -R_b[ri * R_M + (rj - 1)] / gamma;

    scalar_t rmax = fmax(fmax(r0, r1), r2);
    scalar_t softmin;
    if (isinf(rmax) && rmax < 0) {
        softmin = INF;
    } else {
        softmin = -gamma * (log(exp(r0 - rmax) + exp(r1 - rmax) + exp(r2 - rmax)) + rmax);
    }

    R_b[ri * R_M + rj] = D_b[i * M + j] + softmin;
}

// Backward kernel: log-space stable, one block per batch element
template <typename scalar_t>
__global__ void softdtw_backward_kernel(
    const scalar_t* __restrict__ D,  // (B, N, M)
    const scalar_t* __restrict__ R,  // (B, N+2, M+2) — from forward, with modified boundaries
    scalar_t* __restrict__ E,        // (B, N+2, M+2) — logE, then exponentiated at the end
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M,
    scalar_t gamma, int bandwidth)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nx = lengths_x[b];
    const int ny = lengths_y[b];
    const int R_M = M + 2;
    const int R_size = (N + 2) * R_M;

    const scalar_t* D_b = D + b * N * M;
    const scalar_t* R_b = R + b * R_size;
    scalar_t* E_b = E + b * R_size;

    const scalar_t NEG_INF = -INFINITY;
    const scalar_t INF = INFINITY;

    // E is pre-initialized by host: -inf everywhere, E[b, nx+1, ny+1] = 0
    // R is pre-initialized by host: R[b, nx+1, ny+1] = R_orig[b, nx, ny]

    // D padded: for indices outside [1..nx, 1..ny], cost is 0
    // We use a lambda-like inline to read D_pad
    // D_pad[ri][rj] = D[ri-1][rj-1] if 1<=ri<=nx and 1<=rj<=ny, else 0

    // Anti-diagonal wavefront, reversed
    int n_passes = nx + ny - 1;
    for (int p = n_passes - 1; p >= 0; p--) {
        int i = tid;
        int j = p - tid;

        if (i >= 0 && i < nx && j >= 0 && j < ny) {
            bool in_band = (bandwidth < 0) || (abs(i - j) <= bandwidth);
            if (in_band) {
                int ri = i + 1;
                int rj = j + 1;

                // Read R values; demote +inf to -inf for log-space stability
                auto safe_R = [&](int row, int col) -> scalar_t {
                    scalar_t v = R_b[row * R_M + col];
                    return (v == INF) ? NEG_INF : v;
                };

                scalar_t R_ij = safe_R(ri, rj);

                // Read D_pad values
                auto D_pad = [&](int row, int col) -> scalar_t {
                    if (row >= 1 && row <= nx && col >= 1 && col <= ny)
                        return D_b[(row - 1) * M + (col - 1)];
                    return 0;
                };

                // Transition weights in log space
                scalar_t la = (safe_R(ri + 1, rj) - R_ij - D_pad(ri + 1, rj)) / gamma;
                scalar_t lb = (safe_R(ri, rj + 1) - R_ij - D_pad(ri, rj + 1)) / gamma;
                scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma;

                scalar_t t1 = E_b[(ri + 1) * R_M + rj] + la;
                scalar_t t2 = E_b[ri * R_M + (rj + 1)] + lb;
                scalar_t t3 = E_b[(ri + 1) * R_M + (rj + 1)] + lc;

                E_b[ri * R_M + rj] = logsumexp3(t1, t2, t3);
            }
        }
        __syncthreads();
    }

    // Exponentiate logE -> E for the valid region, zero out the rest
    __syncthreads();
    for (int idx = tid; idx < R_size; idx += blockDim.x) {
        int row = idx / R_M;
        int col = idx % R_M;
        if (row >= 1 && row <= nx && col >= 1 && col <= ny) {
            E_b[idx] = exp(E_b[idx]);
        } else {
            E_b[idx] = 0;
        }
    }
}

// Tiled backward kernel: one launch per anti-diagonal
template <typename scalar_t>
__global__ void softdtw_backward_tiled_kernel(
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ R,
    scalar_t* __restrict__ E,
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M, int B,
    scalar_t gamma, int bandwidth,
    int p)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B) return;

    const int nx = lengths_x[b];
    const int ny = lengths_y[b];
    const int R_M = M + 2;

    const int i_lo = max(0, p - M + 1);
    const int i = i_lo + tid;
    const int j = p - i;

    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    if (bandwidth >= 0 && abs(i - j) > bandwidth) return;

    const scalar_t NEG_INF = -INFINITY;
    const scalar_t INF = INFINITY;

    const scalar_t* D_b = D + b * N * M;
    const scalar_t* R_b = R + b * (N + 2) * R_M;
    scalar_t* E_b = E + b * (N + 2) * R_M;

    int ri = i + 1;
    int rj = j + 1;

    auto safe_R = [&](int row, int col) -> scalar_t {
        scalar_t v = R_b[row * R_M + col];
        return (v == INF) ? NEG_INF : v;
    };

    auto D_pad = [&](int row, int col) -> scalar_t {
        if (row >= 1 && row <= nx && col >= 1 && col <= ny)
            return D_b[(row - 1) * M + (col - 1)];
        return 0;
    };

    scalar_t R_ij = safe_R(ri, rj);

    scalar_t la = (safe_R(ri + 1, rj) - R_ij - D_pad(ri + 1, rj)) / gamma;
    scalar_t lb = (safe_R(ri, rj + 1) - R_ij - D_pad(ri, rj + 1)) / gamma;
    scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma;

    scalar_t t1 = E_b[(ri + 1) * R_M + rj] + la;
    scalar_t t2 = E_b[ri * R_M + (rj + 1)] + lb;
    scalar_t t3 = E_b[(ri + 1) * R_M + (rj + 1)] + lc;

    E_b[ri * R_M + rj] = logsumexp3(t1, t2, t3);
}

}  // anonymous namespace


// Host-side launcher: forward
std::tuple<stbl::Tensor, stbl::Tensor> softdtw_cuda_forward(
    stbl::Tensor D,
    stbl::Tensor lengths_x,
    stbl::Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    const stbl::accelerator::DeviceGuard device_guard(D.get_device());

    D = stbl::contiguous(D);
    lengths_x = stbl::contiguous(lengths_to_int64(lengths_x));
    lengths_y = stbl::contiguous(lengths_to_int64(lengths_y));

    const int B = static_cast<int>(D.size(0));
    const int N = static_cast<int>(D.size(1));
    const int M = static_cast<int>(D.size(2));
    const int bandwidth_i = static_cast<int>(bandwidth);

    auto R = stbl::full(
        {B, N + 2, M + 2}, static_cast<double>(INFINITY),
        D.scalar_type(), std::nullopt, D.device());
    {
        auto R_row0 = stbl::select(R, 1, 0);
        auto R_00 = stbl::select(R_row0, 1, 0);
        stbl::fill_(R_00, 0.0);
    }
    auto costs = stbl::empty({B}, D.scalar_type(), std::nullopt, D.device());

    const int max_len = std::max(N, M);
    const stbl::Device cpu_device(stbl::DeviceType::CPU);

    stbl::accelerator::DeviceIndex device_idx = stbl::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(D.scalar_type(), "softdtw_cuda_forward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        if (max_len <= 1024) {
            int threads = std::min(1024, max_len);
            softdtw_forward_kernel<scalar_t><<<B, threads, 0, stream>>>(
                D.const_data_ptr<scalar_t>(),
                R.mutable_data_ptr<scalar_t>(),
                costs.mutable_data_ptr<scalar_t>(),
                lengths_x.const_data_ptr<int64_t>(),
                lengths_y.const_data_ptr<int64_t>(),
                N, M, gamma_val, bandwidth_i);
        } else {
            // Tiled path: one launch per anti-diagonal
            int n_passes = N + M - 1;
            for (int p = 0; p < n_passes; p++) {
                int max_threads = std::min({N, M, p + 1, N + M - 1 - p});
                if (max_threads <= 0) continue;
                int tpb = 256;
                int blocks_x = (max_threads + tpb - 1) / tpb;
                dim3 grid(blocks_x, B);
                softdtw_forward_tiled_kernel<scalar_t><<<grid, tpb, 0, stream>>>(
                    D.const_data_ptr<scalar_t>(),
                    R.mutable_data_ptr<scalar_t>(),
                    lengths_x.const_data_ptr<int64_t>(),
                    lengths_y.const_data_ptr<int64_t>(),
                    N, M, B, gamma_val, bandwidth_i, p);
            }
            // Extract costs — done on CPU for simplicity
            auto R_cpu = stbl::to(R, cpu_device);
            auto lx_cpu = stbl::to(lengths_x, cpu_device);
            auto ly_cpu = stbl::to(lengths_y, cpu_device);
            auto costs_cpu = stbl::empty({B}, D.scalar_type(), std::nullopt, cpu_device);

            auto R_a = accessor3<const scalar_t>(R_cpu);
            auto lx_a = accessor1<const int64_t>(lx_cpu);
            auto ly_a = accessor1<const int64_t>(ly_cpu);
            auto c_a = accessor1<scalar_t>(costs_cpu);
            for (int64_t b = 0; b < B; b++) {
                c_a[b] = R_a[b][lx_a[b]][ly_a[b]];
            }
            costs = stbl::to(costs_cpu, D.device());
        }
    }), AT_EXPAND(AT_FLOATING_TYPES));

    return std::make_tuple(costs, R);
}

// Host-side launcher: backward
stbl::Tensor softdtw_cuda_backward(
    stbl::Tensor D,
    stbl::Tensor R,
    stbl::Tensor lengths_x,
    stbl::Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    const stbl::accelerator::DeviceGuard device_guard(D.get_device());

    D = stbl::contiguous(D);
    R = stbl::contiguous(R);
    lengths_x = stbl::contiguous(lengths_to_int64(lengths_x));
    lengths_y = stbl::contiguous(lengths_to_int64(lengths_y));

    const int B = static_cast<int>(D.size(0));
    const int N = static_cast<int>(D.size(1));
    const int M = static_cast<int>(D.size(2));
    const int bandwidth_i = static_cast<int>(bandwidth);
    const stbl::Device cpu_device(stbl::DeviceType::CPU);

    // Set up R boundary for backward:
    // R_bw[b, nx+1, ny+1] = R[b, nx, ny] (sentinel one step beyond terminal)
    auto R_bw = stbl::clone(R);

    // logE: initialized to -inf, logE[b, nx+1, ny+1] = 0 (sentinel)
    auto E = stbl::full(
        {B, N + 2, M + 2}, -static_cast<double>(INFINITY),
        D.scalar_type(), std::nullopt, D.device());

    {
        auto R_bw_cpu = stbl::to(R_bw, cpu_device);
        auto R_orig_cpu = stbl::to(R, cpu_device);
        auto E_cpu = stbl::to(E, cpu_device);
        auto lx_cpu = stbl::to(lengths_x, cpu_device);
        auto ly_cpu = stbl::to(lengths_y, cpu_device);

        THO_DISPATCH_V2(D.scalar_type(), "softdtw_backward_init", ([&] {
            auto R_bw_a = accessor3<scalar_t>(R_bw_cpu);
            auto R_orig_a = accessor3<const scalar_t>(R_orig_cpu);
            auto E_a = accessor3<scalar_t>(E_cpu);
            auto lx_a = accessor1<const int64_t>(lx_cpu);
            auto ly_a = accessor1<const int64_t>(ly_cpu);
            for (int64_t b = 0; b < B; b++) {
                const int64_t nx = lx_a[b];
                const int64_t ny = ly_a[b];
                R_bw_a[b][nx + 1][ny + 1] = R_orig_a[b][nx][ny];
                E_a[b][nx + 1][ny + 1] = 0;
            }
        }), AT_EXPAND(AT_FLOATING_TYPES));

        R_bw = stbl::to(R_bw_cpu, D.device());
        E = stbl::to(E_cpu, D.device());
    }

    const int max_len = std::max(N, M);

    stbl::accelerator::DeviceIndex device_idx = stbl::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(D.scalar_type(), "softdtw_cuda_backward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        if (max_len <= 1024) {
            int threads = std::min(1024, max_len);
            softdtw_backward_kernel<scalar_t><<<B, threads, 0, stream>>>(
                D.const_data_ptr<scalar_t>(),
                R_bw.const_data_ptr<scalar_t>(),
                E.mutable_data_ptr<scalar_t>(),
                lengths_x.const_data_ptr<int64_t>(),
                lengths_y.const_data_ptr<int64_t>(),
                N, M, gamma_val, bandwidth_i);
        } else {
            int n_passes = N + M - 1;
            for (int p = n_passes - 1; p >= 0; p--) {
                int max_threads = std::min({N, M, p + 1, N + M - 1 - p});
                if (max_threads <= 0) continue;
                int tpb = 256;
                int blocks_x = (max_threads + tpb - 1) / tpb;
                dim3 grid(blocks_x, B);
                softdtw_backward_tiled_kernel<scalar_t><<<grid, tpb, 0, stream>>>(
                    D.const_data_ptr<scalar_t>(),
                    R_bw.const_data_ptr<scalar_t>(),
                    E.mutable_data_ptr<scalar_t>(),
                    lengths_x.const_data_ptr<int64_t>(),
                    lengths_y.const_data_ptr<int64_t>(),
                    N, M, B, gamma_val, bandwidth_i, p);
            }

            // Exponentiate logE -> E for valid region, zero out the rest — on CPU
            auto E_cpu = stbl::to(E, cpu_device);
            auto lx_cpu = stbl::to(lengths_x, cpu_device);
            auto ly_cpu = stbl::to(lengths_y, cpu_device);

            auto E_a = accessor3<scalar_t>(E_cpu);
            auto lx_a = accessor1<const int64_t>(lx_cpu);
            auto ly_a = accessor1<const int64_t>(ly_cpu);
            for (int64_t b = 0; b < B; b++) {
                const int64_t nx = lx_a[b];
                const int64_t ny = ly_a[b];
                for (int64_t i = 1; i <= nx; i++) {
                    for (int64_t j = 1; j <= ny; j++) {
                        E_a[b][i][j] = std::exp(E_a[b][i][j]);
                    }
                }
                for (int64_t i = 0; i < N + 2; i++) {
                    for (int64_t j = 0; j < M + 2; j++) {
                        if (i < 1 || i > nx || j < 1 || j > ny) {
                            E_a[b][i][j] = 0;
                        }
                    }
                }
            }
            E = stbl::to(E_cpu, D.device());
        }
    }), AT_EXPAND(AT_FLOATING_TYPES));

    // Return E[:, 1:N+1, 1:M+1] as grad_D
    auto E_n1 = stbl::narrow(E, 1, 1, N);
    auto E_n = stbl::narrow(E_n1, 2, 1, M);
    return stbl::contiguous(E_n);
}

}  // namespace torchsoftdtw

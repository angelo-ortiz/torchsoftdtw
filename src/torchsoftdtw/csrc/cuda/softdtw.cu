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
using stbl::Tensor;
using ScalarType = torch::headeronly::ScalarType;

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

inline ScalarType acc_scalar_type(ScalarType st) {
    if (st == ScalarType::Half || st == ScalarType::BFloat16) return ScalarType::Float;
    return st;
}

inline Tensor promote_to_acc(Tensor t) {
    auto target = acc_scalar_type(t.scalar_type());
    if (t.scalar_type() == target) return t;
    return stbl::to(t, target);
}

inline Tensor lengths_to_int64(Tensor t) {
    if (t.scalar_type() == ScalarType::Long) return t;
    return stbl::to(t, ScalarType::Long);
}

inline int round_to_warp(int n) {
    return (n + 31) & ~31;
}

namespace {

template <typename scalar_t>
__global__ void set_backward_sentinels_kernel(
    scalar_t* __restrict__ R_bw,
    const scalar_t* __restrict__ R_orig,
    scalar_t* __restrict__ E,
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M, int B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int R_M = M + 2;
    const int R_size = (N + 2) * R_M;
    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    R_bw[b * R_size + (nx + 1) * R_M + (ny + 1)] = R_orig[b * R_size + nx * R_M + ny];
    E[b * R_size + (nx + 1) * R_M + (ny + 1)] = scalar_t(0);
}

template <typename scalar_t>
__global__ void extract_costs_kernel(
    const scalar_t* __restrict__ R,
    scalar_t* __restrict__ costs,
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M, int B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int R_M = M + 2;
    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    costs[b] = R[b * (N + 2) * R_M + nx * R_M + ny];
}

template <typename scalar_t>
__global__ void exponentiate_and_zero_kernel(
    scalar_t* __restrict__ E,
    const int64_t* __restrict__ lengths_x,
    const int64_t* __restrict__ lengths_y,
    int N, int M)
{
    const int b = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int R_M = M + 2;
    const int R_size = (N + 2) * R_M;
    if (idx >= R_size) return;

    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    const int row = idx / R_M;
    const int col = idx % R_M;

    scalar_t* E_b = E + b * R_size;
    if (row >= 1 && row <= nx && col >= 1 && col <= ny) {
        E_b[idx] = exp(E_b[idx]);
    } else {
        E_b[idx] = 0;
    }
}

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
    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    const int R_M = M + 2;

    scalar_t* R_b = R + b * (N + 2) * R_M;
    const scalar_t* D_b = D + b * N * M;

    const scalar_t INF = INFINITY;

    // R is already initialized to +inf with R[b][0][0] = 0 by the host.

    // Shared memory: 3 rotating buffers for R anti-diagonal values.
    // Each buffer has (blockDim.x + 1) elements: index -1 is the boundary slot.
    extern __shared__ char smem_raw[];
    const int buf_stride = blockDim.x + 1;
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    // prev2[-1..blockDim.x-1], prev1[-1..blockDim.x-1], curr[-1..blockDim.x-1]
    scalar_t* prev2 = smem + 1;
    scalar_t* prev1 = smem + buf_stride + 1;
    scalar_t* curr  = smem + 2 * buf_stride + 1;

    // Initialize all slots to +inf
    for (int idx = tid; idx < 3 * buf_stride; idx += blockDim.x) {
        smem[idx] = INF;
    }
    __syncthreads();

    // R[0][0] = 0 is the diagonal predecessor of cell (0,0)
    if (tid == 0) {
        prev2[-1] = scalar_t(0);
    }
    __syncthreads();

    const int n_passes = nx + ny - 1;
    for (int p = 0; p < n_passes; p++) {
        // Reset curr buffer
        curr[tid] = INF;
        if (tid == 0) curr[-1] = INF;

        const int i = tid;
        const int j = p - tid;

        if (i < nx && j >= 0 && j < ny) {
            bool in_band = (bandwidth < 0) || (abs(i - j) <= bandwidth);
            if (in_band) {
                const int ri = i + 1;
                const int rj = j + 1;

                // Read predecessors from shared memory
                scalar_t r0 = -prev2[i - 1] / gamma;  // diag: R[i][j]
                scalar_t r1 = -prev1[i - 1] / gamma;  // above: R[i][j+1]
                scalar_t r2 = -prev1[i]     / gamma;   // left:  R[i+1][j]

                scalar_t rmax = fmax(fmax(r0, r1), r2);
                scalar_t softmin;
                if (isinf(rmax) && rmax < 0) {
                    softmin = INF;
                } else {
                    softmin = -gamma * (log(exp(r0 - rmax) + exp(r1 - rmax) + exp(r2 - rmax)) + rmax);
                }

                scalar_t val = D_b[i * M + j] + softmin;
                R_b[ri * R_M + rj] = val;  // persist for backward
                curr[i] = val;
            }
        }
        __syncthreads();

        // Rotate buffers: prev2 <- prev1 <- curr <- prev2
        scalar_t* tmp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr = tmp;
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
    int p)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B) return;

    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    const int R_M = M + 2;

    const int i_lo = max(0, p - M + 1);
    const int i = i_lo + tid;
    const int j = p - i;

    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    if (bandwidth >= 0 && abs(i - j) > bandwidth) return;

    const scalar_t INF = INFINITY;
    scalar_t* R_b = R + b * (N + 2) * R_M;
    const scalar_t* D_b = D + b * N * M;

    const int ri = i + 1;
    const int rj = j + 1;

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
// Uses shared memory rotating buffers for E successor reads
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
    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
    const int R_M = M + 2;
    const int R_size = (N + 2) * R_M;

    const scalar_t* D_b = D + b * N * M;
    const scalar_t* R_b = R + b * R_size;
    scalar_t* E_b = E + b * R_size;

    const scalar_t NEG_INF = -INFINITY;
    const scalar_t INF = INFINITY;

    // Shared memory: 3 rotating buffers for E anti-diagonal values.
    // Each buffer has (blockDim.x + 1) elements: extra slot at index blockDim.x.
    extern __shared__ char smem_raw[];
    const int buf_stride = blockDim.x + 1;
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    scalar_t* next2 = smem;                      // from 2 diags ahead
    scalar_t* next1 = smem + buf_stride;          // from 1 diag ahead
    scalar_t* curr  = smem + 2 * buf_stride;

    // Initialize all to -inf (logE default)
    for (int idx = tid; idx < 3 * buf_stride; idx += blockDim.x) {
        smem[idx] = NEG_INF;
    }
    __syncthreads();

    // Set sentinel: logE at (nx+1, ny+1) = 0, on anti-diagonal nx+ny, position nx
    if (tid == 0 && nx < buf_stride) {
        next2[nx] = scalar_t(0);
    }
    __syncthreads();

    // Helper lambdas for reading R and D with boundary handling
    auto safe_R = [&](int row, int col) -> scalar_t {
        scalar_t v = R_b[row * R_M + col];
        return (v == INF) ? NEG_INF : v;
    };

    auto D_pad = [&](int row, int col) -> scalar_t {
        if (row >= 1 && row <= nx && col >= 1 && col <= ny)
            return D_b[(row - 1) * M + (col - 1)];
        return scalar_t(0);
    };

    const int n_passes = nx + ny - 1;
    for (int p = n_passes - 1; p >= 0; p--) {
        // Reset curr buffer
        curr[tid] = NEG_INF;
        if (tid == 0) curr[blockDim.x] = NEG_INF;

        const int i = tid;
        const int j = p - tid;

        if (i < nx && j >= 0 && j < ny) {
            bool in_band = (bandwidth < 0) || (abs(i - j) <= bandwidth);
            if (in_band) {
                const int ri = i + 1;
                const int rj = j + 1;

                scalar_t R_ij = safe_R(ri, rj);

                scalar_t la = (safe_R(ri + 1, rj) - R_ij - D_pad(ri + 1, rj)) / gamma;
                scalar_t lb = (safe_R(ri, rj + 1) - R_ij - D_pad(ri, rj + 1)) / gamma;
                scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma;

                // Read E successors from shared memory
                // E[ri+1][rj] = anti-diag p+1, position i+1 -> next1[i+1]
                // E[ri][rj+1] = anti-diag p+1, position i   -> next1[i]
                // E[ri+1][rj+1] = anti-diag p+2, position i+1 -> next2[i+1]
                scalar_t t1 = next1[i + 1] + la;
                scalar_t t2 = next1[i] + lb;
                scalar_t t3 = next2[i + 1] + lc;

                scalar_t val = logsumexp3(t1, t2, t3);
                curr[i] = val;
                E_b[ri * R_M + rj] = val;  // persist for exponentiation
            }
        }
        __syncthreads();

        // Rotate: next2 <- next1 <- curr <- next2
        scalar_t* tmp = next2;
        next2 = next1;
        next1 = curr;
        curr = tmp;
    }

    // Exponentiate logE -> E for valid region, zero out the rest
    for (int idx = tid; idx < R_size; idx += blockDim.x) {
        int row = idx / R_M;
        int col = idx % R_M;
        if (row >= 1 && row <= nx && col >= 1 && col <= ny) {
            // logE was computed via shared memory; read back from the final
            // shared-memory state for the valid region is not feasible since
            // it only holds the last 3 anti-diags. We need the full logE.
            // Fall back: recompute from the original E array.
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

    const int nx = static_cast<int>(lengths_x[b]);
    const int ny = static_cast<int>(lengths_y[b]);
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

    const int ri = i + 1;
    const int rj = j + 1;

    auto safe_R = [&](int row, int col) -> scalar_t {
        scalar_t v = R_b[row * R_M + col];
        return (v == INF) ? NEG_INF : v;
    };

    auto D_pad = [&](int row, int col) -> scalar_t {
        if (row >= 1 && row <= nx && col >= 1 && col <= ny)
            return D_b[(row - 1) * M + (col - 1)];
        return scalar_t(0);
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
std::tuple<Tensor, Tensor> softdtw_cuda_forward(
    Tensor D,
    Tensor lengths_x,
    Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    const stbl::accelerator::DeviceGuard device_guard(D.get_device());

    auto input_type = D.scalar_type();
    lengths_x = lengths_to_int64(lengths_x);
    lengths_y = lengths_to_int64(lengths_y);

    auto D_compute = promote_to_acc(D);
    auto acc_type = D_compute.scalar_type();

    const int B = static_cast<int>(D.size(0));
    const int N = static_cast<int>(D.size(1));
    const int M = static_cast<int>(D.size(2));
    const int bandwidth_i = static_cast<int>(bandwidth);

    auto R = stbl::full(
        {B, N + 2, M + 2}, static_cast<double>(INFINITY),
        acc_type, std::nullopt, D.device());
    {
        auto R_row0 = stbl::select(R, 1, 0);
        auto R_00 = stbl::select(R_row0, 1, 0);
        stbl::fill_(R_00, 0.0);
    }
    auto costs = stbl::new_empty(D, {B}, acc_type);

    const int max_len = std::max(N, M);

    stbl::accelerator::DeviceIndex device_idx = stbl::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(acc_type, "softdtw_cuda_forward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        if (max_len <= 1024) {
            int threads = round_to_warp(std::min(1024, max_len));
            size_t smem_bytes = 3 * (threads + 1) * sizeof(scalar_t);
            softdtw_forward_kernel<scalar_t><<<B, threads, smem_bytes, stream>>>(
                D_compute.const_data_ptr<scalar_t>(),
                R.mutable_data_ptr<scalar_t>(),
                costs.mutable_data_ptr<scalar_t>(),
                lengths_x.const_data_ptr<int64_t>(),
                lengths_y.const_data_ptr<int64_t>(),
                N, M, gamma_val, bandwidth_i);
        } else {
            int n_passes = N + M - 1;
            for (int p = 0; p < n_passes; p++) {
                int max_threads = std::min({N, M, p + 1, N + M - 1 - p});
                if (max_threads <= 0) continue;
                int tpb = 256;
                int blocks_x = (max_threads + tpb - 1) / tpb;
                dim3 grid(blocks_x, B);
                softdtw_forward_tiled_kernel<scalar_t><<<grid, tpb, 0, stream>>>(
                    D_compute.const_data_ptr<scalar_t>(),
                    R.mutable_data_ptr<scalar_t>(),
                    lengths_x.const_data_ptr<int64_t>(),
                    lengths_y.const_data_ptr<int64_t>(),
                    N, M, B, gamma_val, bandwidth_i, p);
            }
            extract_costs_kernel<scalar_t><<<(B + 255) / 256, std::min(256, B), 0, stream>>>(
                R.const_data_ptr<scalar_t>(),
                costs.mutable_data_ptr<scalar_t>(),
                lengths_x.const_data_ptr<int64_t>(),
                lengths_y.const_data_ptr<int64_t>(),
                N, M, B);
        }
    }), AT_EXPAND(AT_FLOATING_TYPES));

    return std::make_tuple(costs, R);
}

// Host-side launcher: backward
Tensor softdtw_cuda_backward(
    Tensor D,
    Tensor R,
    Tensor lengths_x,
    Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    const stbl::accelerator::DeviceGuard device_guard(D.get_device());

    auto input_type = D.scalar_type();
    lengths_x = lengths_to_int64(lengths_x);
    lengths_y = lengths_to_int64(lengths_y);

    auto D_compute = promote_to_acc(D);
    auto acc_type = D_compute.scalar_type();

    const int B = static_cast<int>(D.size(0));
    const int N = static_cast<int>(D.size(1));
    const int M = static_cast<int>(D.size(2));
    const int bandwidth_i = static_cast<int>(bandwidth);

    // R_bw with sentinel: R_bw[b, nx+1, ny+1] = R[b, nx, ny]
    auto R_bw = stbl::clone(R);

    // logE: initialized to -inf, sentinel logE[b, nx+1, ny+1] = 0
    auto E = stbl::full(
        {B, N + 2, M + 2}, -static_cast<double>(INFINITY),
        acc_type, std::nullopt, D.device());

    const int max_len = std::max(N, M);

    stbl::accelerator::DeviceIndex device_idx = stbl::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(acc_type, "softdtw_backward_sentinel", ([&] {
        set_backward_sentinels_kernel<scalar_t><<<(B + 255) / 256, std::min(256, B), 0, stream>>>(
            R_bw.mutable_data_ptr<scalar_t>(),
            R.const_data_ptr<scalar_t>(),
            E.mutable_data_ptr<scalar_t>(),
            lengths_x.const_data_ptr<int64_t>(),
            lengths_y.const_data_ptr<int64_t>(),
            N, M, B);
    }), AT_EXPAND(AT_FLOATING_TYPES));

    THO_DISPATCH_V2(acc_type, "softdtw_cuda_backward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        if (max_len <= 1024) {
            int threads = round_to_warp(std::min(1024, max_len));
            size_t smem_bytes = 3 * (threads + 1) * sizeof(scalar_t);
            softdtw_backward_kernel<scalar_t><<<B, threads, smem_bytes, stream>>>(
                D_compute.const_data_ptr<scalar_t>(),
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
                    D_compute.const_data_ptr<scalar_t>(),
                    R_bw.const_data_ptr<scalar_t>(),
                    E.mutable_data_ptr<scalar_t>(),
                    lengths_x.const_data_ptr<int64_t>(),
                    lengths_y.const_data_ptr<int64_t>(),
                    N, M, B, gamma_val, bandwidth_i, p);
            }

            // Exponentiate logE -> E for valid region, zero out the rest
            {
                const int R_size = (N + 2) * (M + 2);
                int tpb_exp = 256;
                int blocks_exp = (R_size + tpb_exp - 1) / tpb_exp;
                dim3 grid_exp(blocks_exp, B);
                exponentiate_and_zero_kernel<scalar_t><<<grid_exp, tpb_exp, 0, stream>>>(
                    E.mutable_data_ptr<scalar_t>(),
                    lengths_x.const_data_ptr<int64_t>(),
                    lengths_y.const_data_ptr<int64_t>(),
                    N, M);
            }
        }
    }), AT_EXPAND(AT_FLOATING_TYPES));

    // Return E[:, 1:N+1, 1:M+1] as grad_D, cast to input dtype
    auto E_n1 = stbl::narrow(E, 1, 1, N);
    auto E_out = stbl::narrow(E_n1, 2, 1, M);
    if (E_out.scalar_type() != input_type) {
        E_out = stbl::to(E_out, input_type);
    }
    return E_out;
}

}  // namespace torchsoftdtw

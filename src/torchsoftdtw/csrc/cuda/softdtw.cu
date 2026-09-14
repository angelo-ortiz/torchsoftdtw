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
#include <torch/headeronly/util/Exception.h>

// Not part of the public stable ABI surface, but the only way (pre PyTorch 2.13's
// Stream::nativeHandle()) to get the raw cudaStream_t backing the current stream,
// so that our raw kernel launches enqueue on the same stream PyTorch itself uses.
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace torchsoftdtw {

namespace stbl = torch::stable;
using stbl::Tensor;
using ScalarType = torch::headeronly::ScalarType;

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

// ---- Device-safe tensor accessors ----
// Matches PyTorch's PackedTensorAccessor32 interface, constructed from
// stable-ABI tensors. Uses int32_t indices (sufficient for dims < 2^31).

template <typename T, int N>
struct PackedAccessor {
    T* data_;
    int32_t sizes_[N];
    int32_t strides_[N];

    PackedAccessor() = default;

    PackedAccessor(T* data, const int64_t* sizes, const int64_t* strides)
        : data_(data) {
        for (int i = 0; i < N; i++) {
            sizes_[i] = static_cast<int32_t>(sizes[i]);
            strides_[i] = static_cast<int32_t>(strides[i]);
        }
    }

    __device__ __forceinline__ int32_t size(int dim) const { return sizes_[dim]; }
    __device__ __forceinline__ int32_t stride(int dim) const { return strides_[dim]; }
    __device__ __forceinline__ T* data() const { return data_; }
};

template <typename T>
struct Accessor3D : PackedAccessor<T, 3> {
    using PackedAccessor<T, 3>::PackedAccessor;

    __device__ __forceinline__ T& operator()(int i, int j, int k) const {
        return this->data_[i * this->strides_[0] + j * this->strides_[1] + k * this->strides_[2]];
    }
};

template <typename T>
struct Accessor1D : PackedAccessor<T, 1> {
    using PackedAccessor<T, 1>::PackedAccessor;

    __device__ __forceinline__ T& operator()(int i) const {
        return this->data_[i * this->strides_[0]];
    }
};

template <typename T>
Accessor3D<T> make_acc3d(Tensor t) {
    return Accessor3D<T>(
        static_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename T>
Accessor3D<const T> make_acc3d_const(Tensor t) {
    return Accessor3D<const T>(
        static_cast<const T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename T>
Accessor1D<T> make_acc1d(Tensor t) {
    return Accessor1D<T>(
        static_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename T>
Accessor1D<const T> make_acc1d_const(Tensor t) {
    return Accessor1D<const T>(
        static_cast<const T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

inline int round_to_warp(int n) {
    return (n + 31) & ~31;
}

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t logsumexp3(scalar_t a, scalar_t b, scalar_t c) {
    scalar_t m = fmax(fmax(a, b), c);
    if (isinf(m) && m < 0) return m;
    return m + log(exp(a - m) + exp(b - m) + exp(c - m));
}

// =====================================================================
// Forward kernel: one block per batch element, anti-diagonal wavefront
// Uses shared memory rotating buffers for fast predecessor reads
// =====================================================================
template <typename scalar_t>
__global__ void softdtw_forward_kernel(
    Accessor3D<const scalar_t> D,
    Accessor3D<scalar_t> R,
    Accessor1D<scalar_t> costs,
    Accessor1D<const int64_t> lengths_x,
    Accessor1D<const int64_t> lengths_y,
    int N, int M,
    scalar_t gamma, int bandwidth)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nx = static_cast<int>(lengths_x(b));
    const int ny = static_cast<int>(lengths_y(b));

    const scalar_t INF = INFINITY;

    // Initialize R to +inf, R[b,0,0] = 0
    const int R_size = (N + 2) * (M + 2);
    scalar_t* R_b = &R(b, 0, 0);
    for (int idx = tid; idx < R_size; idx += blockDim.x) {
        R_b[idx] = INF;
    }
    __syncthreads();
    if (tid == 0) {
        R(b, 0, 0) = scalar_t(0);
    }
    __syncthreads();

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
        __syncthreads();

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

                scalar_t val = D(b, i, j) + softmin;
                R(b, ri, rj) = val;  // persist for backward
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
        costs(b) = R(b, nx, ny);
    }
}

// =====================================================================
// Tiled forward kernel: one launch per anti-diagonal, for sequences > 1024
// =====================================================================
template <typename scalar_t>
__global__ void softdtw_forward_tiled_kernel(
    Accessor3D<const scalar_t> D,
    Accessor3D<scalar_t> R,
    Accessor1D<const int64_t> lengths_x,
    Accessor1D<const int64_t> lengths_y,
    int N, int M, int B,
    scalar_t gamma, int bandwidth,
    int p)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B) return;

    const int nx = static_cast<int>(lengths_x(b));
    const int ny = static_cast<int>(lengths_y(b));

    const int i_lo = max(0, p - M + 1);
    const int i = i_lo + tid;
    const int j = p - i;

    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    if (bandwidth >= 0 && abs(i - j) > bandwidth) return;

    const scalar_t INF = INFINITY;
    const int ri = i + 1;
    const int rj = j + 1;

    scalar_t r0 = -R(b, ri - 1, rj - 1) / gamma;
    scalar_t r1 = -R(b, ri - 1, rj) / gamma;
    scalar_t r2 = -R(b, ri, rj - 1) / gamma;

    scalar_t rmax = fmax(fmax(r0, r1), r2);
    scalar_t softmin;
    if (isinf(rmax) && rmax < 0) {
        softmin = INF;
    } else {
        softmin = -gamma * (log(exp(r0 - rmax) + exp(r1 - rmax) + exp(r2 - rmax)) + rmax);
    }

    R(b, ri, rj) = D(b, i, j) + softmin;
}

// =====================================================================
// Backward kernel: log-space stable, one block per batch element
// Uses shared memory rotating buffers for E successor reads
// =====================================================================
template <typename scalar_t>
__global__ void softdtw_backward_kernel(
    Accessor3D<const scalar_t> D,
    Accessor3D<const scalar_t> R,
    Accessor3D<scalar_t> E,
    Accessor1D<const int64_t> lengths_x,
    Accessor1D<const int64_t> lengths_y,
    int N, int M,
    scalar_t gamma, int bandwidth)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nx = static_cast<int>(lengths_x(b));
    const int ny = static_cast<int>(lengths_y(b));
    const int R_M = M + 2;
    const int R_size = (N + 2) * R_M;

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
        scalar_t v = R(b, row, col);
        return (v == INF) ? NEG_INF : v;
    };

    auto D_pad = [&](int row, int col) -> scalar_t {
        if (row >= 1 && row <= nx && col >= 1 && col <= ny)
            return D(b, row - 1, col - 1);
        return scalar_t(0);
    };

    const int n_passes = nx + ny - 1;
    for (int p = n_passes - 1; p >= 0; p--) {
        // Reset curr buffer
        curr[tid] = NEG_INF;
        if (tid == 0) curr[blockDim.x] = NEG_INF;
        __syncthreads();

        const int i = tid;
        const int j = p - tid;

        if (i < nx && j >= 0 && j < ny) {
            bool in_band = (bandwidth < 0) || (abs(i - j) <= bandwidth);
            if (in_band) {
                const int ri = i + 1;
                const int rj = j + 1;

                scalar_t R_ij = safe_R(ri, rj);

                scalar_t la = (safe_R(ri + 1, rj)     - R_ij - D_pad(ri + 1, rj))     / gamma;
                scalar_t lb = (safe_R(ri, rj + 1)     - R_ij - D_pad(ri, rj + 1))     / gamma;
                scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma;

                // Read E successors from shared memory
                // E[ri+1][rj] = anti-diag p+1, position i+1 -> next1[i+1]
                // E[ri][rj+1] = anti-diag p+1, position i   -> next1[i]
                // E[ri+1][rj+1] = anti-diag p+2, position i+1 -> next2[i+1]
                scalar_t t1 = next1[i + 1] + la;
                scalar_t t2 = next1[i]     + lb;
                scalar_t t3 = next2[i + 1] + lc;

                scalar_t val = logsumexp3(t1, t2, t3);
                curr[i] = val;
                E(b, ri, rj) = val;  // persist for exponentiation
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
    __syncthreads();
    scalar_t* E_b = &E(b, 0, 0);
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

// =====================================================================
// Tiled backward kernel: one launch per anti-diagonal
// =====================================================================
template <typename scalar_t>
__global__ void softdtw_backward_tiled_kernel(
    Accessor3D<const scalar_t> D,
    Accessor3D<const scalar_t> R,
    Accessor3D<scalar_t> E,
    Accessor1D<const int64_t> lengths_x,
    Accessor1D<const int64_t> lengths_y,
    int N, int M, int B,
    scalar_t gamma, int bandwidth,
    int p)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B) return;

    const int nx = static_cast<int>(lengths_x(b));
    const int ny = static_cast<int>(lengths_y(b));

    const int i_lo = max(0, p - M + 1);
    const int i = i_lo + tid;
    const int j = p - i;

    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    if (bandwidth >= 0 && abs(i - j) > bandwidth) return;

    const scalar_t NEG_INF = -INFINITY;
    const scalar_t INF = INFINITY;

    const int ri = i + 1;
    const int rj = j + 1;

    auto safe_R = [&](int row, int col) -> scalar_t {
        scalar_t v = R(b, row, col);
        return (v == INF) ? NEG_INF : v;
    };

    auto D_pad = [&](int row, int col) -> scalar_t {
        if (row >= 1 && row <= nx && col >= 1 && col <= ny)
            return D(b, row - 1, col - 1);
        return scalar_t(0);
    };

    scalar_t R_ij = safe_R(ri, rj);

    scalar_t la = (safe_R(ri + 1, rj)     - R_ij - D_pad(ri + 1, rj))     / gamma;
    scalar_t lb = (safe_R(ri, rj + 1)     - R_ij - D_pad(ri, rj + 1))     / gamma;
    scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma;

    scalar_t t1 = E(b, ri + 1, rj)     + la;
    scalar_t t2 = E(b, ri, rj + 1)     + lb;
    scalar_t t3 = E(b, ri + 1, rj + 1) + lc;

    E(b, ri, rj) = logsumexp3(t1, t2, t3);
}

}  // anonymous namespace


// =====================================================================
// Host-side launcher: forward
// =====================================================================
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
    const stbl::Device cpu_device(stbl::DeviceType::CPU);

    stbl::accelerator::DeviceIndex device_idx = stbl::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(acc_type, "softdtw_cuda_forward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        auto D_acc = make_acc3d_const<scalar_t>(D_compute);
        auto R_acc = make_acc3d<scalar_t>(R);
        auto costs_acc = make_acc1d<scalar_t>(costs);
        auto lx_acc = make_acc1d_const<int64_t>(lengths_x);
        auto ly_acc = make_acc1d_const<int64_t>(lengths_y);

        if (max_len <= 1024) {
            int threads = round_to_warp(std::min(1024, max_len));
            size_t smem_bytes = 3 * (threads + 1) * sizeof(scalar_t);
            softdtw_forward_kernel<scalar_t><<<B, threads, smem_bytes, stream>>>(
                D_acc, R_acc, costs_acc, lx_acc, ly_acc,
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
                    D_acc, R_acc, lx_acc, ly_acc,
                    N, M, B, gamma_val, bandwidth_i, p);
            }
            // Extract costs on CPU
            auto R_cpu = stbl::to(R, cpu_device);
            auto lx_cpu = stbl::to(lengths_x, cpu_device);
            auto ly_cpu = stbl::to(lengths_y, cpu_device);
            auto costs_cpu = stbl::empty({B}, acc_type, std::nullopt, cpu_device);

            using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<scalar_t, 3>;
            using TensorAccessor1 = torch::headeronly::HeaderOnlyTensorAccessor<const int64_t, 1>;
            using ScalarAccessor = torch::headeronly::HeaderOnlyTensorAccessor<scalar_t, 1>;
            auto R_a = TensorAccessor(
                static_cast<scalar_t*>(R_cpu.data_ptr()), R_cpu.sizes().data(), R_cpu.strides().data());
            auto lx_a = TensorAccessor1(
                static_cast<const int64_t*>(lx_cpu.data_ptr()), lx_cpu.sizes().data(), lx_cpu.strides().data());
            auto ly_a = TensorAccessor1(
                static_cast<const int64_t*>(ly_cpu.data_ptr()), ly_cpu.sizes().data(), ly_cpu.strides().data());
            auto c_a = ScalarAccessor(
                static_cast<scalar_t*>(costs_cpu.data_ptr()), costs_cpu.sizes().data(), costs_cpu.strides().data());
            for (int64_t b = 0; b < B; b++) {
                c_a[b] = R_a[b][lx_a[b]][ly_a[b]];
            }
            costs = stbl::to(costs_cpu, D.device());
        }
    }), AT_EXPAND(AT_FLOATING_TYPES));

    return std::make_tuple(costs, R);
}

// =====================================================================
// Host-side launcher: backward
// =====================================================================
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
    const stbl::Device cpu_device(stbl::DeviceType::CPU);

    // R_bw with sentinel: R_bw[b, nx+1, ny+1] = R[b, nx, ny]
    auto R_bw = stbl::clone(R);

    // logE: initialized to -inf, sentinel logE[b, nx+1, ny+1] = 0
    auto E = stbl::full(
        {B, N + 2, M + 2}, -static_cast<double>(INFINITY),
        acc_type, std::nullopt, D.device());

    {
        auto R_bw_cpu = stbl::to(R_bw, cpu_device);
        auto R_orig_cpu = stbl::to(R, cpu_device);
        auto E_cpu = stbl::to(E, cpu_device);
        auto lx_cpu = stbl::to(lengths_x, cpu_device);
        auto ly_cpu = stbl::to(lengths_y, cpu_device);

        THO_DISPATCH_V2(acc_type, "softdtw_backward_init", ([&] {
            using TA3 = torch::headeronly::HeaderOnlyTensorAccessor<scalar_t, 3>;
            using TA3c = torch::headeronly::HeaderOnlyTensorAccessor<const scalar_t, 3>;
            using TA1c = torch::headeronly::HeaderOnlyTensorAccessor<const int64_t, 1>;

            auto R_bw_a = TA3(static_cast<scalar_t*>(R_bw_cpu.data_ptr()),
                R_bw_cpu.sizes().data(), R_bw_cpu.strides().data());
            auto R_orig_a = TA3c(static_cast<const scalar_t*>(R_orig_cpu.data_ptr()),
                R_orig_cpu.sizes().data(), R_orig_cpu.strides().data());
            auto E_a = TA3(static_cast<scalar_t*>(E_cpu.data_ptr()),
                E_cpu.sizes().data(), E_cpu.strides().data());
            auto lx_a = TA1c(static_cast<const int64_t*>(lx_cpu.data_ptr()),
                lx_cpu.sizes().data(), lx_cpu.strides().data());
            auto ly_a = TA1c(static_cast<const int64_t*>(ly_cpu.data_ptr()),
                ly_cpu.sizes().data(), ly_cpu.strides().data());

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

    THO_DISPATCH_V2(acc_type, "softdtw_cuda_backward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);

        auto D_acc = make_acc3d_const<scalar_t>(D_compute);
        auto R_acc = make_acc3d_const<scalar_t>(R_bw);
        auto E_acc = make_acc3d<scalar_t>(E);
        auto lx_acc = make_acc1d_const<int64_t>(lengths_x);
        auto ly_acc = make_acc1d_const<int64_t>(lengths_y);

        if (max_len <= 1024) {
            int threads = round_to_warp(std::min(1024, max_len));
            size_t smem_bytes = 3 * (threads + 1) * sizeof(scalar_t);
            softdtw_backward_kernel<scalar_t><<<B, threads, smem_bytes, stream>>>(
                D_acc, R_acc, E_acc, lx_acc, ly_acc,
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
                    D_acc, R_acc, E_acc, lx_acc, ly_acc,
                    N, M, B, gamma_val, bandwidth_i, p);
            }

            // Exponentiate logE -> E for valid region, zero out the rest
            auto E_cpu = stbl::to(E, cpu_device);
            auto lx_cpu = stbl::to(lengths_x, cpu_device);
            auto ly_cpu = stbl::to(lengths_y, cpu_device);

            using TA3 = torch::headeronly::HeaderOnlyTensorAccessor<scalar_t, 3>;
            using TA1c = torch::headeronly::HeaderOnlyTensorAccessor<const int64_t, 1>;
            auto E_a = TA3(static_cast<scalar_t*>(E_cpu.data_ptr()),
                E_cpu.sizes().data(), E_cpu.strides().data());
            auto lx_a = TA1c(static_cast<const int64_t*>(lx_cpu.data_ptr()),
                lx_cpu.sizes().data(), lx_cpu.strides().data());
            auto ly_a = TA1c(static_cast<const int64_t*>(ly_cpu.data_ptr()),
                ly_cpu.sizes().data(), ly_cpu.strides().data());
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

    // Return E[:, 1:N+1, 1:M+1] as grad_D, cast to input dtype
    auto E_n1 = stbl::narrow(E, 1, 1, N);
    auto E_out = stbl::narrow(E_n1, 2, 1, M);
    if (E_out.scalar_type() != input_type) {
        E_out = stbl::to(E_out, input_type);
    }
    return E_out;
}

}  // namespace torchsoftdtw

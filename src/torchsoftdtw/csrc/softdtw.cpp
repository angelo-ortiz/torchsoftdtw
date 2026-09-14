#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>

/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the STABLE_TORCH_LIBRARY static initializers
   below are run.
   PyMODINIT_FUNC (rather than a bare extern "C") is required: it carries the
   __declspec(dllexport) that makes the symbol visible on Windows, where
   PyTorch's BuildExtension suppresses setuptools' automatic /EXPORT:PyInit__C. */
PyMODINIT_FUNC PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
    };
    return PyModule_Create(&module_def);
}

namespace torchsoftdtw {

namespace stbl = torch::stable;

using stbl::Tensor;
template <typename T, size_t N> using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;

// accessor<T, N>(...) would need a comma in its explicit template-argument
// list. That comma sits inside "<...>" rather than "(...)", which the
// preprocessor's paren-nesting tracker does not see as protected, so writing
// it directly inside a THO_DISPATCH_V2 body (a macro argument) confuses that
// macro's internal argument counting. Fixed-N, single-type-argument helpers
// sidestep the issue entirely.
template <typename T>
inline TensorAccessor<T, 1> accessor1(Tensor t) {
    return TensorAccessor<T, 1>(
        reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename T>
inline TensorAccessor<T, 3> accessor3(Tensor t) {
    return TensorAccessor<T, 3>(
        reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

// lengths_x/lengths_y may arrive as int32 or int64 (checked in
// softdtw_forward_checks). Every kernel below only ever deals with int64_t
// lengths, so normalize once, up front, instead of templating the CPU
// dispatch and the CUDA kernels over a second integer type.
inline Tensor lengths_to_int64(Tensor t) {
    if (t.scalar_type() == torch::headeronly::ScalarType::Long) return t;
    return stbl::to(t, torch::headeronly::ScalarType::Long);
}

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

// Forward declarations of CUDA launchers (defined in cuda/softdtw.cu)
#ifdef WITH_CUDA
std::tuple<Tensor, Tensor> softdtw_cuda_forward(
    Tensor D, Tensor lengths_x, Tensor lengths_y,
    double gamma, int64_t bandwidth);
Tensor softdtw_cuda_backward(
    Tensor D, Tensor R, Tensor lengths_x,
    Tensor lengths_y, double gamma, int64_t bandwidth);
#endif

namespace {

template <typename scalar_t>
scalar_t logsumexp3(scalar_t a, scalar_t b, scalar_t c) {
    scalar_t m = std::max({a, b, c});
    if (std::isinf(m) && m < 0) return m;
    return m + std::log(std::exp(a - m) + std::exp(b - m) + std::exp(c - m));
}

}  // anonymous namespace


std::tuple<Tensor, Tensor> softdtw_cpu_forward(
    Tensor D,
    Tensor lengths_x,
    Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    lengths_x = lengths_to_int64(lengths_x);
    lengths_y = lengths_to_int64(lengths_y);

    const int64_t B = D.size(0);
    const int64_t N = D.size(1);
    const int64_t M = D.size(2);

    auto D_compute = promote_to_acc(D);
    auto acc_type = D_compute.scalar_type();

    auto R = stbl::full(
        {B, N + 2, M + 2}, std::numeric_limits<double>::infinity(),
        acc_type, std::nullopt, D.device());
    {
        auto R_row0 = stbl::select(R, 1, 0);
        auto R_00 = stbl::select(R_row0, 1, 0);
        stbl::fill_(R_00, 0.0);
    }
    auto costs = stbl::new_empty(D, {B}, acc_type);

    THO_DISPATCH_V2(acc_type, "softdtw_cpu_forward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);
        const scalar_t INF = std::numeric_limits<scalar_t>::infinity();

        auto D_a = accessor3<const scalar_t>(D_compute);
        auto R_a = accessor3<scalar_t>(R);
        auto lx_a = accessor1<const int64_t>(lengths_x);
        auto ly_a = accessor1<const int64_t>(lengths_y);
        auto c_a = accessor1<scalar_t>(costs);

        stbl::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
            for (int64_t b = start; b < end; b++) {
                const int64_t nx = lx_a[b];
                const int64_t ny = ly_a[b];

                auto D_b = D_a[b];
                auto R_b = R_a[b];

                for (int64_t i = 0; i < nx; i++) {
                    for (int64_t j = 0; j < ny; j++) {
                        if (bandwidth >= 0 && std::abs(i - j) > bandwidth) continue;

                        int64_t ri = i + 1;
                        int64_t rj = j + 1;

                        scalar_t r0 = -R_b[ri - 1][rj - 1] / gamma_val;
                        scalar_t r1 = -R_b[ri - 1][rj] / gamma_val;
                        scalar_t r2 = -R_b[ri][rj - 1] / gamma_val;

                        scalar_t rmax = std::max({r0, r1, r2});
                        scalar_t softmin;
                        if (std::isinf(rmax) && rmax < 0) {
                            softmin = INF;
                        } else {
                            softmin = -gamma_val * (std::log(
                                std::exp(r0 - rmax) + std::exp(r1 - rmax) + std::exp(r2 - rmax)
                            ) + rmax);
                        }

                        R_b[ri][rj] = D_b[i][j] + softmin;
                    }
                }

                c_a[b] = R_b[nx][ny];
            }
        });
    }), AT_EXPAND(AT_FLOATING_TYPES));

    return std::make_tuple(costs, R);
}


Tensor softdtw_cpu_backward(
    Tensor D,
    Tensor R,
    Tensor lengths_x,
    Tensor lengths_y,
    double gamma,
    int64_t bandwidth)
{
    auto input_type = D.scalar_type();
    lengths_x = lengths_to_int64(lengths_x);
    lengths_y = lengths_to_int64(lengths_y);

    const int64_t B = D.size(0);
    const int64_t N = D.size(1);
    const int64_t M = D.size(2);

    auto D_compute = promote_to_acc(D);
    auto acc_type = D_compute.scalar_type();

    // Set up R boundary for backward (R is already in acc_type from forward)
    auto R_bw = stbl::clone(R);

    // logE initialized to -inf
    auto E = stbl::full(
        {B, N + 2, M + 2}, -std::numeric_limits<double>::infinity(),
        acc_type, std::nullopt, D.device());

    THO_DISPATCH_V2(acc_type, "softdtw_cpu_backward", ([&] {
        scalar_t gamma_val = static_cast<scalar_t>(gamma);
        const scalar_t INF = std::numeric_limits<scalar_t>::infinity();
        const scalar_t NINF = -INF;

        auto D_a = accessor3<const scalar_t>(D_compute);
        auto R_bw_a = accessor3<scalar_t>(R_bw);
        auto R_a = accessor3<const scalar_t>(R);
        auto E_a = accessor3<scalar_t>(E);
        auto lx_a = accessor1<const int64_t>(lengths_x);
        auto ly_a = accessor1<const int64_t>(lengths_y);

        stbl::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
            for (int64_t b = start; b < end; b++) {
                const int64_t nx = lx_a[b];
                const int64_t ny = ly_a[b];

                auto D_b = D_a[b];
                auto R_bw_b = R_bw_a[b];
                auto R_b = R_a[b];
                auto E_b = E_a[b];

                // Sentinel: one step beyond the terminal cell
                R_bw_b[nx + 1][ny + 1] = R_b[nx][ny];
                E_b[nx + 1][ny + 1] = 0;

                auto safe_R = [&](int64_t row, int64_t col) -> scalar_t {
                    scalar_t v = R_bw_b[row][col];
                    return (v == INF) ? NINF : v;
                };

                auto D_pad = [&](int64_t row, int64_t col) -> scalar_t {
                    if (row >= 1 && row <= nx && col >= 1 && col <= ny)
                        return D_b[row - 1][col - 1];
                    return 0;
                };

                for (int64_t i = nx - 1; i >= 0; i--) {
                    for (int64_t j = ny - 1; j >= 0; j--) {
                        if (bandwidth >= 0 && std::abs(i - j) > bandwidth) continue;

                        int64_t ri = i + 1;
                        int64_t rj = j + 1;

                        scalar_t R_ij = safe_R(ri, rj);

                        scalar_t la = (safe_R(ri + 1, rj) - R_ij - D_pad(ri + 1, rj)) / gamma_val;
                        scalar_t lb = (safe_R(ri, rj + 1) - R_ij - D_pad(ri, rj + 1)) / gamma_val;
                        scalar_t lc = (safe_R(ri + 1, rj + 1) - R_ij - D_pad(ri + 1, rj + 1)) / gamma_val;

                        scalar_t t1 = E_b[ri + 1][rj] + la;
                        scalar_t t2 = E_b[ri][rj + 1] + lb;
                        scalar_t t3 = E_b[ri + 1][rj + 1] + lc;

                        E_b[ri][rj] = logsumexp3(t1, t2, t3);
                    }
                }

                // Exponentiate valid region
                for (int64_t i = 1; i <= nx; i++) {
                    for (int64_t j = 1; j <= ny; j++) {
                        E_b[i][j] = std::exp(E_b[i][j]);
                    }
                }
                // Zero out padded region
                for (int64_t i = 0; i < N + 2; i++) {
                    for (int64_t j = 0; j < M + 2; j++) {
                        if (i < 1 || i > nx || j < 1 || j > ny) {
                            E_b[i][j] = 0;
                        }
                    }
                }
            }
        });
    }), AT_EXPAND(AT_FLOATING_TYPES));

    auto E_n1 = stbl::narrow(E, 1, 1, N);
    auto E_out = stbl::narrow(E_n1, 2, 1, M);
    if (E_out.scalar_type() != input_type) {
        E_out = stbl::to(E_out, input_type);
    }
    return E_out;
}


// ----------------------------------------------------------------------------------------------------------------------
// torch.ops.torchsoftdtw.* bindings
//
// Schemas are registered once via STABLE_TORCH_LIBRARY, with separate CPU/CUDA
// kernels registered via STABLE_TORCH_LIBRARY_IMPL so the dispatcher picks the
// right one based on the input tensor's device. Kernels are built exclusively
// against torch::stable / the libtorch stable ABI (torch/csrc/stable, plus
// header-only torch/headeronly utilities), so this extension does not need to
// be rebuilt for every libtorch/Python version it runs against.
// ----------------------------------------------------------------------------------------------------------------------

namespace {

void softdtw_forward_checks(
    const Tensor& D, const Tensor& lengths_x,
    const Tensor& lengths_y, double gamma)
{
    STD_TORCH_CHECK(D.dim() == 3, "D must be 3D (B, N, M)");
    STD_TORCH_CHECK(lengths_x.dim() == 1 && lengths_y.dim() == 1, "lengths must be 1D");
    STD_TORCH_CHECK(D.size(0) == lengths_x.size(0) && D.size(0) == lengths_y.size(0),
                "Batch size mismatch");
    STD_TORCH_CHECK(gamma > 0, "gamma must be positive");

    auto is_float_type = [](ScalarType t) {
        return t == ScalarType::Float || t == ScalarType::Double
            || t == ScalarType::Half || t == ScalarType::BFloat16;
    };
    STD_TORCH_CHECK(is_float_type(D.scalar_type()),
                "D must be float16, bfloat16, float32 or float64");

    auto is_int32_or_int64 = [](ScalarType t) {
        return t == ScalarType::Int || t == ScalarType::Long;
    };
    STD_TORCH_CHECK(is_int32_or_int64(lengths_x.scalar_type()), "lengths_x must be int32 or int64");
    STD_TORCH_CHECK(is_int32_or_int64(lengths_y.scalar_type()), "lengths_y must be int32 or int64");
}

std::tuple<Tensor, Tensor> softdtw_cpu_forward_op(
    Tensor D, Tensor lengths_x, Tensor lengths_y,
    double gamma, int64_t bandwidth)
{
    softdtw_forward_checks(D, lengths_x, lengths_y, gamma);
    return softdtw_cpu_forward(D, lengths_x, lengths_y, gamma, bandwidth);
}

Tensor softdtw_cpu_backward_op(
    Tensor D, Tensor R, Tensor lengths_x, Tensor lengths_y,
    double gamma, int64_t bandwidth)
{
    return softdtw_cpu_backward(D, R, lengths_x, lengths_y, gamma, bandwidth);
}

#ifdef WITH_CUDA
std::tuple<Tensor, Tensor> softdtw_cuda_forward_op(
    Tensor D, Tensor lengths_x, Tensor lengths_y,
    double gamma, int64_t bandwidth)
{
    softdtw_forward_checks(D, lengths_x, lengths_y, gamma);
    return softdtw_cuda_forward(D, lengths_x, lengths_y, gamma, bandwidth);
}

Tensor softdtw_cuda_backward_op(
    Tensor D, Tensor R, Tensor lengths_x, Tensor lengths_y,
    double gamma, int64_t bandwidth)
{
    return softdtw_cuda_backward(D, R, lengths_x, lengths_y, gamma, bandwidth);
}
#endif

}  // anonymous namespace


STABLE_TORCH_LIBRARY(torchsoftdtw, m) {
    m.def("forward(Tensor D, Tensor lengths_x, Tensor lengths_y, float gamma, int bandwidth) -> (Tensor, Tensor)");
    m.def("backward(Tensor D, Tensor R, Tensor lengths_x, Tensor lengths_y, float gamma, int bandwidth) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchsoftdtw, CPU, m) {
    m.impl("forward", TORCH_BOX(&softdtw_cpu_forward_op));
    m.impl("backward", TORCH_BOX(&softdtw_cpu_backward_op));
}

#ifdef WITH_CUDA
STABLE_TORCH_LIBRARY_IMPL(torchsoftdtw, CUDA, m) {
    m.impl("forward", TORCH_BOX(&softdtw_cuda_forward_op));
    m.impl("backward", TORCH_BOX(&softdtw_cuda_backward_op));
}
#endif

}  // namespace torchsoftdtw

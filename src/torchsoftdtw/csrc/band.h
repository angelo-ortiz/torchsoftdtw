#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define TORCHSOFTDTW_HOST_DEVICE __host__ __device__
#else
#define TORCHSOFTDTW_HOST_DEVICE
#endif

namespace torchsoftdtw {

/* Sakoe-Chiba band around the diagonal joining (0, 0) to (nx - 1, ny - 1).

   With K = min(nx, ny), L = max(nx, ny) and slope s = (L - 1) / (K - 1), a cell
   at short-axis index k and long-axis index l is in band iff
       |l - k * s| <= max(bandwidth, s / 2),
   i.e. the bandwidth is measured in long-axis steps. The s / 2 floor guarantees
   a connected warping path to the terminal cell even for bandwidth = 0. For
   square inputs s = 1 and this reduces to |i - j| <= bandwidth.
   Evaluated in integer arithmetic (both sides multiplied by 2 * (K - 1)) so the
   CPU and CUDA kernels agree exactly. A negative bandwidth disables the band. */
TORCHSOFTDTW_HOST_DEVICE inline bool in_band(
    int64_t i, int64_t j, int64_t nx, int64_t ny, int64_t bandwidth)
{
    if (bandwidth < 0) return true;
    const bool x_short = nx <= ny;
    const int64_t K = x_short ? nx : ny;
    const int64_t L = x_short ? ny : nx;
    if (K <= 1) return true;
    const int64_t k = x_short ? i : j;
    const int64_t l = x_short ? j : i;
    int64_t diff = l * (K - 1) - k * (L - 1);
    if (diff < 0) diff = -diff;
    const int64_t band = 2 * bandwidth * (K - 1);
    const int64_t width = band > L - 1 ? band : L - 1;
    return 2 * diff <= width;
}

}  // namespace torchsoftdtw

#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cmath>
#include "types.cuh"
#include "../constants.cuh"

// =======================================================
// Phase field || psi = +1 in the self phase || psi = -1 in the other phase
// =======================================================

__device__ __forceinline__ real_t psi(const real_t rho_self, const real_t rho_other) noexcept
{
    const real_t rho_sum = rho_self + rho_other;

    const real_t rho_diff = rho_self - rho_other;

    return rho_diff / rho_sum;
}

// =======================================================
// Jet mask
// =======================================================

__device__ __forceinline__ bool isJet(const label_t x, const label_t z) noexcept
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz) <= (jet_radius * jet_radius) ? 1 : 0;
}

// =======================================================
// Convective BC
// =======================================================

__host__ __device__ __forceinline__ real_t clamp01(const real_t x) noexcept
{
    return x < static_cast<real_t>(0)
               ? static_cast<real_t>(0)
               : (x > static_cast<real_t>(1)
                      ? static_cast<real_t>(1)
                      : x);
}

__device__ __forceinline__ real_t convective_outlet_value(const real_t phiB_old, const real_t phiF, const real_t uc) noexcept
{
    return phiB_old - uc * (phiB_old - phiF);
}

#endif

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
// Sponge layer
// =======================================================

inline constexpr real_t sponge_K = static_cast<real_t>(100.0);

inline constexpr label_t sponge_y_end = NY - static_cast<label_t>(2);

inline constexpr label_t sponge_y_start = sponge_y_end - sponge_cells + static_cast<label_t>(1);

__host__ __device__ __forceinline__ real_t clamp01(const real_t x) noexcept
{
    return x < static_cast<real_t>(0)
               ? static_cast<real_t>(0)
               : (x > static_cast<real_t>(1)
                      ? static_cast<real_t>(1)
                      : x);
}

// =======================================================
// Sponge coordinate
// =======================================================

__host__ __device__ __forceinline__ real_t sponge_s(const label_t y) noexcept
{
    constexpr real_t denom = static_cast<real_t>(
        sponge_y_end > sponge_y_start
            ? sponge_y_end - sponge_y_start
            : static_cast<label_t>(1));

    const real_t s = (static_cast<real_t>(y) - static_cast<real_t>(sponge_y_start)) / denom;

    return clamp01(s);
}

// =======================================================
// Smooth sponge profile
// =======================================================

__host__ __device__ __forceinline__ real_t sponge_profile(const real_t s) noexcept
{
    return s * s * s * (static_cast<real_t>(10) - static_cast<real_t>(15) * s + static_cast<real_t>(6) * s * s);
}

// =======================================================
// Local viscosity, relaxation time, and relaxation frequency
// =======================================================

__host__ __device__ __forceinline__ real_t nu_sponge(const label_t y) noexcept
{
    const real_t s = sponge_s(y);
    const real_t phi = sponge_profile(s);

    return nu * (static_cast<real_t>(1) + sponge_K * phi);
}

__host__ __device__ __forceinline__ real_t tau_sponge(const label_t y) noexcept
{
    return static_cast<real_t>(0.5) + nu_sponge(y) / cs2;
}

__host__ __device__ __forceinline__ real_t omega_sponge(const label_t y) noexcept
{
    return static_cast<real_t>(1) / tau_sponge(y);
}

__host__ __device__ __forceinline__ real_t oms_sponge(const label_t y) noexcept
{
    return static_cast<real_t>(1) - omega_sponge(y);
}

#endif

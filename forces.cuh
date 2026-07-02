#ifndef FORCES_CUH
#define FORCES_CUH

#include <cmath>

#include "constants.cuh"
#include "memory.cuh"
#include "stencil_ct.cuh"

#include "utilities/indexing.cuh"
#include "utilities/bounds.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"
#include "utilities/constexprFor.cuh"

// =======================================================
// Color-gradient / force calculation   F = grad(psi)
// =======================================================

__device__ __forceinline__ void force_outlet_limited(const real_t *__restrict__ rho_self, const real_t *__restrict__ rho_other,
                                                     const label_t x, const label_t y, const label_t z,
                                                     real_t &Fx_component, real_t &Fy_component, real_t &Fz_component) noexcept
{
    real_t sx = static_cast<real_t>(0);
    real_t sy = static_cast<real_t>(0);
    real_t sz = static_cast<real_t>(0);

    constexpr_for<1, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            constexpr int cx_i = LbmStencil::cx<i>();
            constexpr int cy_i = LbmStencil::cy<i>();
            constexpr int cz_i = LbmStencil::cz<i>();

            constexpr real_t cx = static_cast<real_t>(cx_i);
            constexpr real_t cy = static_cast<real_t>(cy_i);
            constexpr real_t cz = static_cast<real_t>(cz_i);

            constexpr real_t wi = LbmStencil::w<i>();

            const label_t xn = wrapx(static_cast<label_t>(static_cast<int>(x) + cx_i));
            const label_t zn = wrapz(static_cast<label_t>(static_cast<int>(z) + cz_i));

            const label_t yn = force_y_outlet_limited(static_cast<int>(y) + cy_i);

            const label_t idn = idx(xn, yn, zn);

            const real_t psi_n = psi(rho_self[idn], rho_other[idn]);

            sx += wi * psi_n * cx;
            sy += wi * psi_n * cy;
            sz += wi * psi_n * cz;
        });

    Fx_component = inv_cs2 * sx;
    Fy_component = inv_cs2 * sy;
    Fz_component = inv_cs2 * sz;
}

// =======================================================
// Absolute value of color gradient
// =======================================================

__device__ __forceinline__ real_t absforce_calcul(const real_t Fx, const real_t Fy, const real_t Fz) noexcept
{
    return sqrtf(Fx * Fx + Fy * Fy + Fz * Fz);
}

// =======================================================
// cos^2 rule    || cos^2(theta_i) = [(F . c_i)^2] / |F|^2
// =======================================================

template <label_t I>
__device__ __forceinline__ real_t cos2rule(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absF) noexcept
{
    if constexpr (I == 0)
    {
        return static_cast<real_t>(0);
    }

    if (absF <= static_cast<real_t>(1.0e-4))
    {
        return static_cast<real_t>(0);
    }

    constexpr real_t cx = static_cast<real_t>(LbmStencil::cx<I>());
    constexpr real_t cy = static_cast<real_t>(LbmStencil::cy<I>());
    constexpr real_t cz = static_cast<real_t>(LbmStencil::cz<I>());

    const real_t Fici = Fx * cx + Fy * cy + Fz * cz;

    return (Fici * Fici) / (absF * absF);
}

// =======================================================
// Effective relaxation time at the interface
// =======================================================

__device__ __forceinline__ real_t tau_interface(const real_t rho_self, const real_t tau_self,
                                                const real_t rho_other, const real_t tau_other) noexcept
{
    const real_t psix = psi(rho_self, rho_other);

    return static_cast<real_t>(0.5) * (static_cast<real_t>(1) + psix) * tau_self +
           static_cast<real_t>(0.5) * (static_cast<real_t>(1) - psix) * tau_other;
}

// =======================================================
// Perturbation amplitude
// =======================================================

__device__ __forceinline__ real_t A_calculation(const real_t tau_eff) noexcept
{
    return sigma_over_4cs4 / tau_eff;
}

// =======================================================
// Pre-computation for second collision operator
// =======================================================
// const real_t tau_self0, const real_t tau_other0,

__device__ __forceinline__ void preOmega2_outlet_limited(const real_t *__restrict__ rho_self,
                                                         const real_t *__restrict__ rho_other,
                                                         const label_t x, const label_t y, const label_t z,
                                                         real_t &Fx, real_t &Fy, real_t &Fz,
                                                         real_t &absF, real_t &A) noexcept
{
    force_outlet_limited(rho_self, rho_other, x, y, z, Fx, Fy, Fz);

    const real_t tau_eff = taub;

    A = A_calculation(tau_eff);

    absF = absforce_calcul(Fx, Fy, Fz);
}

#endif

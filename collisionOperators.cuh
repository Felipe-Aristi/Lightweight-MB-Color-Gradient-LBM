#ifndef COLLISIONOPERATORS_CUH
#define COLLISIONOPERATORS_CUH

#include "constants.cuh"
#include "forces.cuh"
#include "stencil_ct.cuh"

#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

// =======================================================
// Perturbation collision operator Omega^(2)
// =======================================================
//  Omega_i^(2) = (A/2) |F| [ w_i cos^2(theta_i) - B_i ]

template <label_t I>
__device__ __forceinline__ real_t Omega2(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absF, const real_t A) noexcept
{
    const real_t cos2rule_i = cos2rule<I>(Fx, Fy, Fz, absF);

    constexpr real_t wi = LbmStencil::w<I>();
    constexpr real_t Bi = LbmStencil::B<I>();

    return static_cast<real_t>(0.5) * A * absF * (wi * cos2rule_i - Bi);
}

// =======================================================
// cos(phi_i) = (F . c_i) / (|F| |c_i|)
// =======================================================

template <label_t I>
__device__ __forceinline__ real_t cosphi(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absF) noexcept
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

    constexpr real_t inv_cnorm = LbmStencil::invcnorm<I>();

    const real_t dot = Fx * cx + Fy * cy + Fz * cz;

    return dot * inv_cnorm / absF;
}

// =======================================================
// Recoloring antidiffusion contribution
// =======================================================
// Delta_i = beta * (rho_R rho_B / rho^2) * cos(phi_i) * f_i^eq(rho, u = 0)

template <label_t I>
__device__ __forceinline__ real_t recolorDelta(const real_t rhor, const real_t rhob,
                                               const real_t Fx, const real_t Fy, const real_t Fz,
                                               const real_t absF) noexcept
{
    if constexpr (I == 0)
    {
        return static_cast<real_t>(0);
    }

    const real_t rho = rhor + rhob;
    constexpr real_t wi = LbmStencil::w<I>();

    const real_t cos_i = cosphi<I>(Fx, Fy, Fz, absF);

    const real_t coeff = beta_recolor * (rhor * rhob) / (rho * rho);

    const real_t fieq = wi * rho;

    return coeff * cos_i * fieq;
}

#endif
#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../utilities/mathUtilities.cuh"

// =======================================================
// Inlet extrapolation boundary condition
// =======================================================

__device__ __forceinline__ void inlet_bc_calculation(MomentsDevice A, const label_t x, const label_t z) noexcept
{
    constexpr label_t yB = static_cast<label_t>(0);
    constexpr label_t yF = static_cast<label_t>(1);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t jetMask = static_cast<real_t>(isJet(x, z));

    A.rhor[idB] = (static_cast<real_t>(1) - jetMask) * rhor0;
    A.rhob[idB] = jetMask * rhob0;

    A.ux[idB] = static_cast<real_t>(0);
    A.uy[idB] = jetMask * jet_velocity;
    A.uz[idB] = static_cast<real_t>(0);

    // Non-equilibrium extrapolation from the nearest fluid node.
    A.Pixx[idB] = A.Pixx[idF];
    A.Pixy[idB] = A.Pixy[idF];
    A.Piyy[idB] = A.Piyy[idF];
    A.Piyz[idB] = A.Piyz[idF];
    A.Pizz[idB] = A.Pizz[idF];
    A.Pixz[idB] = A.Pixz[idF];
}

// =======================================================
// Convective outlet boundary condition
// =======================================================

__device__ __forceinline__ real_t outlet_blue_fraction(const real_t rhor,
                                                       const real_t rhob,
                                                       const real_t fallback) noexcept
{
    const real_t rho = rhor + rhob;

    if (!(rho > static_cast<real_t>(1.0e-12)))
    {
        return fallback;
    }

    return clamp01(rhob / rho);
}

__device__ __forceinline__ void outlet_neumann_bc_calculation(MomentsDevice A,
                                                              const MomentsDevice Aold,
                                                              const label_t x,
                                                              const label_t z) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t rrF = A.rhor[idF];
    const real_t rbF = A.rhob[idF];
    const real_t uxF = A.ux[idF];
    const real_t uyF = A.uy[idF];
    const real_t uzF = A.uz[idF];

    A.rhor[idB] = rrF;
    A.rhob[idB] = rbF;
    A.ux[idB] = uxF;
    A.uy[idB] = uyF;
    A.uz[idB] = uzF;

    // Non-equilibrium extrapolation from the nearest fluid node.
    A.Pixx[idB] = A.Pixx[idF];
    A.Pixy[idB] = A.Pixy[idF];
    A.Piyy[idB] = A.Piyy[idF];
    A.Piyz[idB] = A.Piyz[idF];
    A.Pizz[idB] = A.Pizz[idF];
    A.Pixz[idB] = A.Pixz[idF];
}

#endif

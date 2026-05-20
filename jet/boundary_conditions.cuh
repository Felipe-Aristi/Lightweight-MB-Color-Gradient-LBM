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

    const real_t uyB = isJet(x, z) * jet_velocity;

    A.rho[idB] = rho0;

    A.ux[idB] = static_cast<real_t>(0);
    A.uy[idB] = uyB;
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
// Neumann outlet boundary condition
// =======================================================

__device__ __forceinline__ void outlet_neumann_bc_calculation(MomentsDevice A, const label_t x, const label_t z) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    A.rho[idB] = A.rho[idF];

    A.ux[idB] = A.ux[idF];
    A.uy[idB] = A.uy[idF];
    A.uz[idB] = A.uz[idF];

    A.Pixx[idB] = A.Pixx[idF];
    A.Pixy[idB] = A.Pixy[idF];
    A.Piyy[idB] = A.Piyy[idF];
    A.Piyz[idB] = A.Piyz[idF];
    A.Pizz[idB] = A.Pizz[idF];
    A.Pixz[idB] = A.Pixz[idF];
}

#endif
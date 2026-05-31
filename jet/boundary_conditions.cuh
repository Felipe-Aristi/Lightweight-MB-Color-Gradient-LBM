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

__device__ __forceinline__ void outlet_neumann_bc_calculation(MomentsDevice B, const MomentsDevice Aold, const label_t x, const label_t z) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t rrF = B.rhor[idF];
    const real_t rbF = B.rhob[idF];

    const real_t uxF = B.ux[idF];
    const real_t uyF = B.uy[idF];
    const real_t uzF = B.uz[idF];

    const real_t rrB_old = Aold.rhor[idB];
    const real_t rbB_old = Aold.rhob[idB];

    const real_t uxB_old = Aold.ux[idB];
    const real_t uyB_old = Aold.uy[idB];
    const real_t uzB_old = Aold.uz[idB];

    const real_t uc = clamp01(uyF);

    const real_t rhoB_old = rrB_old + rbB_old;

    real_t rrB;
    real_t rbB;
    real_t uxB;
    real_t uyB;
    real_t uzB;

    if (rhoB_old <= static_cast<real_t>(1.0e-20))
    {
        rrB = rrF;
        rbB = rbF;

        uxB = uxF;
        uyB = uyF;
        uzB = uzF;
    }
    else
    {
        rrB = convective_outlet_value(rrB_old, rrF, uc);
        rbB = convective_outlet_value(rbB_old, rbF, uc);

        uxB = convective_outlet_value(uxB_old, uxF, uc);
        uyB = convective_outlet_value(uyB_old, uyF, uc);
        uzB = convective_outlet_value(uzB_old, uzF, uc);
    }

    B.rhor[idB] = rrB;
    B.rhob[idB] = rbB;

    B.ux[idB] = uxB;
    B.uy[idB] = uyB;
    B.uz[idB] = uzB;

    B.Pixx[idB] = B.Pixx[idF];
    B.Pixy[idB] = B.Pixy[idF];
    B.Piyy[idB] = B.Piyy[idF];
    B.Piyz[idB] = B.Piyz[idF];
    B.Pizz[idB] = B.Pizz[idF];
    B.Pixz[idB] = B.Pixz[idF];
}

#endif

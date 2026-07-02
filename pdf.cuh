#ifndef PDF_CUH
#define PDF_CUH

#include "constants.cuh"
#include "memory.cuh"
#include "stencil_ct.cuh"

#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"
#include "utilities/constexprFor.cuh"

// ===================================================
// Distribution functions
// ===================================================

template <label_t I>
__device__ __forceinline__ real_t feq(const real_t rho,
                                      const real_t ux,
                                      const real_t uy,
                                      const real_t uz) noexcept
{
    constexpr real_t cx = static_cast<real_t>(LbmStencil::cx<I>());
    constexpr real_t cy = static_cast<real_t>(LbmStencil::cy<I>());
    constexpr real_t cz = static_cast<real_t>(LbmStencil::cz<I>());
    constexpr real_t wi = LbmStencil::w<I>();

    const real_t cu = ux * cx + uy * cy + uz * cz;
    const real_t usq = ux * ux + uy * uy + uz * uz;

    const real_t cu2 = cu * cu;

    const real_t A2eq = (cu * inv_cs2) - (usq * inv_2cs2) + (cu * cu) * inv_2cs4;
    const real_t A3eq = cu * (cu2 * inv_6cs6 - usq * inv_2cs4);

    return wi * rho * (static_cast<real_t>(1.0) + A2eq + A3eq);
}

template <label_t I>
__device__ __forceinline__ real_t fneqr(const real_t Pixx,
                                        const real_t Pixy,
                                        const real_t Piyy,
                                        const real_t Piyz,
                                        const real_t Pizz,
                                        const real_t Pixz,
                                        const real_t ux,
                                        const real_t uy,
                                        const real_t uz) noexcept
{
    constexpr real_t Hxx = LbmStencil::Hxx<I>();
    constexpr real_t Hxy = LbmStencil::Hxy<I>();
    constexpr real_t Hyy = LbmStencil::Hyy<I>();
    constexpr real_t Hyz = LbmStencil::Hyz<I>();
    constexpr real_t Hzz = LbmStencil::Hzz<I>();
    constexpr real_t Hxz = LbmStencil::Hxz<I>();

    constexpr real_t Hxxy = LbmStencil::Hxxy<I>();
    constexpr real_t Hxxz = LbmStencil::Hxxz<I>();
    constexpr real_t Hxyy = LbmStencil::Hxyy<I>();
    constexpr real_t Hxzz = LbmStencil::Hxzz<I>();
    constexpr real_t Hyyz = LbmStencil::Hyyz<I>();
    constexpr real_t Hyzz = LbmStencil::Hyzz<I>();
    constexpr real_t Hxyz = LbmStencil::Hxyz<I>();

    constexpr real_t wi = LbmStencil::w<I>();

    const real_t A2neq = (Pixx * Hxx + static_cast<real_t>(2.0) * Pixy * Hxy + Piyy * Hyy + static_cast<real_t>(2.0) * Piyz * Hyz + Pizz * Hzz + static_cast<real_t>(2.0) * Pixz * Hxz) * inv_2cs4;

    const real_t a3xxy = Pixx * uy + static_cast<real_t>(2.0) * Pixy * ux;
    const real_t a3xxz = Pixx * uz + static_cast<real_t>(2.0) * Pixz * ux;
    const real_t a3xyy = Piyy * ux + static_cast<real_t>(2.0) * Pixy * uy;
    const real_t a3xzz = Pizz * ux + static_cast<real_t>(2.0) * Pixz * uz;
    const real_t a3yyz = Piyy * uz + static_cast<real_t>(2.0) * Piyz * uy;
    const real_t a3yzz = Pizz * uy + static_cast<real_t>(2.0) * Piyz * uz;
    const real_t a3xyz = Pixy * uz + Pixz * uy + Piyz * ux;

    const real_t A3neq = (a3xxy * Hxxy + a3xxz * Hxxz + a3xyy * Hxyy + a3xzz * Hxzz + a3yyz * Hyyz + a3yzz * Hyzz + static_cast<real_t>(2.0) * a3xyz * Hxyz) * inv_2cs6;

    return wi * (A2neq + A3neq);
}

template <label_t I>
__device__ __forceinline__ real_t fpost(const real_t rho,
                                        const real_t ux, const real_t uy, const real_t uz,
                                        const real_t Pixx, const real_t Pixy, const real_t Piyy,
                                        const real_t Piyz, const real_t Pizz, const real_t Pixz) noexcept
{
    return feq<I>(rho, ux, uy, uz) + oms * fneqr<I>(Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, ux, uy, uz);
}

// ===================================================
// Guo forcing term calculation
// ===================================================

template <label_t I>
__device__ __forceinline__ real_t guo_force(const real_t Fx, const real_t Fy, const real_t Fz,
                                            const real_t ux, const real_t uy, const real_t uz) noexcept
{
    constexpr real_t wi = LbmStencil::w<I>();

    constexpr real_t cx = static_cast<real_t>(LbmStencil::cx<I>());
    constexpr real_t cy = static_cast<real_t>(LbmStencil::cy<I>());
    constexpr real_t cz = static_cast<real_t>(LbmStencil::cz<I>());

    constexpr real_t Hxx = LbmStencil::Hxx<I>();
    constexpr real_t Hxy = LbmStencil::Hxy<I>();
    constexpr real_t Hyy = LbmStencil::Hyy<I>();
    constexpr real_t Hyz = LbmStencil::Hyz<I>();
    constexpr real_t Hzz = LbmStencil::Hzz<I>();
    constexpr real_t Hxz = LbmStencil::Hxz<I>();

    constexpr real_t Hxxy = LbmStencil::Hxxy<I>();
    constexpr real_t Hxxz = LbmStencil::Hxxz<I>();
    constexpr real_t Hxyy = LbmStencil::Hxyy<I>();
    constexpr real_t Hxzz = LbmStencil::Hxzz<I>();
    constexpr real_t Hyyz = LbmStencil::Hyyz<I>();
    constexpr real_t Hyzz = LbmStencil::Hyzz<I>();
    constexpr real_t Hxyz = LbmStencil::Hxyz<I>();

    const real_t A1 =
        (Fx * cx + Fy * cy + Fz * cz) * inv_cs2;

    const real_t A2 =
        (Fx * ux * Hxx + Fy * uy * Hyy + Fz * uz * Hzz + (Fx * uy + Fy * ux) * Hxy + (Fy * uz + Fz * uy) * Hyz + (Fx * uz + Fz * ux) * Hxz) * inv_cs4;

    const real_t A3 =
        ((Fx * ux * uy + half * Fy * ux * ux) * Hxxy +
         (Fx * ux * uz + half * Fz * ux * ux) * Hxxz +
         (Fy * ux * uy + half * Fx * uy * uy) * Hxyy +
         (Fz * ux * uz + half * Fx * uz * uz) * Hxzz +
         (Fy * uy * uz + half * Fz * uy * uy) * Hyyz +
         (Fz * uy * uz + half * Fy * uz * uz) * Hyzz +
         (Fx * uy * uz + Fy * ux * uz + Fz * ux * uy) * Hxyz) *
        inv_cs6;

    return wi * (A1 + A2 + A3);
}

// ===================================================
// Regularized non-equilibrium distribution with forcing
//
// fneq_forced = fneq_regularized - (dt / 2) * F_i
// ===================================================

template <label_t I>
__device__ __forceinline__ real_t fneqr_forced(const real_t Pixx,
                                               const real_t Pixy,
                                               const real_t Piyy,
                                               const real_t Piyz,
                                               const real_t Pizz,
                                               const real_t Pixz,
                                               const real_t ux,
                                               const real_t uy,
                                               const real_t uz,
                                               const real_t Fx,
                                               const real_t Fy,
                                               const real_t Fz) noexcept
{
    const real_t fneq_regularized = fneqr<I>(Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, ux, uy, uz);

    const real_t Fi = guo_force<I>(Fx, Fy, Fz, ux, uy, uz);

    return fneq_regularized - half * Fi;
}

#endif

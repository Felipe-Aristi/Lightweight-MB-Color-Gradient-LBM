#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "memory.cuh"
#include "stencil_ct.cuh"

#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
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
    constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
    constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
    constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());
    constexpr real_t wi = D3Q27::w<I>();

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
    constexpr real_t Hxx = D3Q27::Hxx<I>();
    constexpr real_t Hxy = D3Q27::Hxy<I>();
    constexpr real_t Hyy = D3Q27::Hyy<I>();
    constexpr real_t Hyz = D3Q27::Hyz<I>();
    constexpr real_t Hzz = D3Q27::Hzz<I>();
    constexpr real_t Hxz = D3Q27::Hxz<I>();

    constexpr real_t Hxxy = D3Q27::Hxxy<I>();
    constexpr real_t Hxxz = D3Q27::Hxxz<I>();
    constexpr real_t Hxyy = D3Q27::Hxyy<I>();
    constexpr real_t Hxzz = D3Q27::Hxzz<I>();
    constexpr real_t Hyyz = D3Q27::Hyyz<I>();
    constexpr real_t Hyzz = D3Q27::Hyzz<I>();
    constexpr real_t Hxyz = D3Q27::Hxyz<I>();

    constexpr real_t wi = D3Q27::w<I>();

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
// Main LBM kernel RCS
// ===================================================

__device__ __forceinline__ void RCS(const MomentsDevice A,
                                    MomentsDevice B,
                                    const label_t x,
                                    const label_t y,
                                    const label_t z) noexcept
{
    const label_t id = idx(x, y, z);

    const real_t *__restrict__ rho = A.rho;
    const real_t *__restrict__ ux = A.ux;
    const real_t *__restrict__ uy = A.uy;
    const real_t *__restrict__ uz = A.uz;
    const real_t *__restrict__ Pixx = A.Pixx;
    const real_t *__restrict__ Pixy = A.Pixy;
    const real_t *__restrict__ Piyy = A.Piyy;
    const real_t *__restrict__ Piyz = A.Piyz;
    const real_t *__restrict__ Pizz = A.Pizz;
    const real_t *__restrict__ Pixz = A.Pixz;

    real_t *__restrict__ rho_next = B.rho;
    real_t *__restrict__ ux_next = B.ux;
    real_t *__restrict__ uy_next = B.uy;
    real_t *__restrict__ uz_next = B.uz;
    real_t *__restrict__ Pixx_next = B.Pixx;
    real_t *__restrict__ Pixy_next = B.Pixy;
    real_t *__restrict__ Piyy_next = B.Piyy;
    real_t *__restrict__ Piyz_next = B.Piyz;
    real_t *__restrict__ Pizz_next = B.Pizz;
    real_t *__restrict__ Pixz_next = B.Pixz;

    real_t sum = static_cast<real_t>(0);

    real_t jx = static_cast<real_t>(0);
    real_t jy = static_cast<real_t>(0);
    real_t jz = static_cast<real_t>(0);

    real_t Axx = static_cast<real_t>(0);
    real_t Axy = static_cast<real_t>(0);
    real_t Ayy = static_cast<real_t>(0);
    real_t Ayz = static_cast<real_t>(0);
    real_t Azz = static_cast<real_t>(0);
    real_t Axz = static_cast<real_t>(0);

    const real_t oms_local = oms_sponge(y);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            // Pull treaming -->  f_i(x,t+dt) = f_i^post(x - c_i,t)

            const label_t src = pullidPeri<i>(x, y, z);

            // Read moments from A at the source cell
            const real_t rho_s = rho[src];

            const real_t ux_s = ux[src];
            const real_t uy_s = uy[src];
            const real_t uz_s = uz[src];

            const real_t Pixx_s = Pixx[src];
            const real_t Pixy_s = Pixy[src];
            const real_t Piyy_s = Piyy[src];
            const real_t Piyz_s = Piyz[src];
            const real_t Pizz_s = Pizz[src];
            const real_t Pixz_s = Pixz[src];

            // RCS
            
            const real_t fieq = feq<i>(rho_s, ux_s, uy_s, uz_s);
            const real_t fineqr = fneqr<i>(Pixx_s, Pixy_s, Piyy_s, Piyz_s, Pizz_s, Pixz_s, ux_s, uy_s, uz_s);

            const real_t fi = fieq + oms_local * fineqr;

            // New moments calculation
            sum += fi;

            constexpr real_t cx = static_cast<real_t>(D3Q27::cx<i>());
            constexpr real_t cy = static_cast<real_t>(D3Q27::cy<i>());
            constexpr real_t cz = static_cast<real_t>(D3Q27::cz<i>());

            jx += fi * cx;
            jy += fi * cy;
            jz += fi * cz;

            constexpr real_t Hxx = D3Q27::Hxx<i>();
            constexpr real_t Hxy = D3Q27::Hxy<i>();
            constexpr real_t Hyy = D3Q27::Hyy<i>();
            constexpr real_t Hyz = D3Q27::Hyz<i>();
            constexpr real_t Hzz = D3Q27::Hzz<i>();
            constexpr real_t Hxz = D3Q27::Hxz<i>();

            Axx += fi * Hxx;
            Axy += fi * Hxy;
            Ayy += fi * Hyy;
            Ayz += fi * Hyz;
            Azz += fi * Hzz;
            Axz += fi * Hxz; });

    const real_t rho_t = sum;

    rho_next[id] = rho_t;

    const real_t inv_rho = static_cast<real_t>(1) / rho_t;

    const real_t vx = jx * inv_rho;
    const real_t vy = jy * inv_rho;
    const real_t vz = jz * inv_rho;

    ux_next[id] = vx;
    uy_next[id] = vy;
    uz_next[id] = vz;

    Pixx_next[id] = Axx - rho_t * vx * vx;
    Pixy_next[id] = Axy - rho_t * vx * vy;
    Piyy_next[id] = Ayy - rho_t * vy * vy;
    Piyz_next[id] = Ayz - rho_t * vy * vz;
    Pizz_next[id] = Azz - rho_t * vz * vz;
    Pixz_next[id] = Axz - rho_t * vx * vz;
}

#endif

#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "memory.cuh"
#include "stencil_ct.cuh"
#include "pdf.cuh"
#include "collisionOperators.cuh"

#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"
#include "utilities/constexprFor.cuh"

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

    const real_t *__restrict__ rhor = A.rhor;
    const real_t *__restrict__ rhob = A.rhob;
    const real_t *__restrict__ ux = A.ux;
    const real_t *__restrict__ uy = A.uy;
    const real_t *__restrict__ uz = A.uz;
    const real_t *__restrict__ Pixx = A.Pixx;
    const real_t *__restrict__ Pixy = A.Pixy;
    const real_t *__restrict__ Piyy = A.Piyy;
    const real_t *__restrict__ Piyz = A.Piyz;
    const real_t *__restrict__ Pizz = A.Pizz;
    const real_t *__restrict__ Pixz = A.Pixz;

    real_t *__restrict__ rhor_next = B.rhor;
    real_t *__restrict__ rhob_next = B.rhob;
    real_t *__restrict__ ux_next = B.ux;
    real_t *__restrict__ uy_next = B.uy;
    real_t *__restrict__ uz_next = B.uz;
    real_t *__restrict__ Pixx_next = B.Pixx;
    real_t *__restrict__ Pixy_next = B.Pixy;
    real_t *__restrict__ Piyy_next = B.Piyy;
    real_t *__restrict__ Piyz_next = B.Piyz;
    real_t *__restrict__ Pizz_next = B.Pizz;
    real_t *__restrict__ Pixz_next = B.Pixz;

    real_t sumr = static_cast<real_t>(0);
    real_t sumb = static_cast<real_t>(0);

    real_t jx = static_cast<real_t>(0);
    real_t jy = static_cast<real_t>(0);
    real_t jz = static_cast<real_t>(0);

    real_t Axx = static_cast<real_t>(0);
    real_t Axy = static_cast<real_t>(0);
    real_t Ayy = static_cast<real_t>(0);
    real_t Ayz = static_cast<real_t>(0);
    real_t Azz = static_cast<real_t>(0);
    real_t Axz = static_cast<real_t>(0);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            // Pull treaming -->  f_i(x,t+dt) = f_i^post(x - c_i,t)

            // Bubble case
            //const label_t src = pullidPeri<i>(x, y, z);

            // Jet case
            const label_t xs = wrapx(pullx<i>(x));
            const label_t ys = pully<i>(y);
            const label_t zs = wrapz(pullz<i>(z));

            const label_t src = idx(xs, ys, zs);

            const bool src_is_inlet_ghost = (ys == static_cast<label_t>(0));
            const bool src_is_outlet_ghost = (ys == NY - static_cast<label_t>(1));
            const bool src_is_y_ghost = src_is_inlet_ghost || src_is_outlet_ghost;

            if (src_is_y_ghost)
            {
                const label_t yF =src_is_inlet_ghost
                        ? static_cast<label_t>(1)
                        : NY - static_cast<label_t>(2);

                const label_t idF = idx(xs, yF, zs);

                const real_t rrB = rhor[src];
                const real_t rbB = rhob[src];

                const real_t rhoB = rrB + rbB;
                const real_t inv_rhoB = static_cast<real_t>(1) / rhoB;

                const real_t aR = rrB * inv_rhoB;
                const real_t aB = rbB * inv_rhoB;

                // Boundary velocity for the equilibrium part.
                const real_t uxB = ux[src];
                const real_t uyB = uy[src];
                const real_t uzB = uz[src];

                // Fluid non-equilibrium tensor and fluid velocity.
                const real_t PixxF = Pixx[idF];
                const real_t PixyF = Pixy[idF];
                const real_t PiyyF = Piyy[idF];
                const real_t PiyzF = Piyz[idF];
                const real_t PizzF = Pizz[idF];
                const real_t PixzF = Pixz[idF];

                const real_t uxF = ux[idF];
                const real_t uyF = uy[idF];
                const real_t uzF = uz[idF];

                const real_t gieq = feq<i>(rhoB, uxB, uyB, uzB);

                const real_t gineqr = fneqr<i>(PixxF, PixyF, PiyyF, PiyzF, PizzF, PixzF,uxF, uyF, uzF);

                const real_t gi = gieq + oms * gineqr;

                const real_t fr_i = aR * gi;
                const real_t fb_i = aB * gi;

                sumr += fr_i;
                sumb += fb_i;

                const real_t g_i = fr_i + fb_i;

                constexpr real_t cx = static_cast<real_t>(D3Q27::cx<i>());
                constexpr real_t cy = static_cast<real_t>(D3Q27::cy<i>());
                constexpr real_t cz = static_cast<real_t>(D3Q27::cz<i>());

                jx += g_i * cx;
                jy += g_i * cy;
                jz += g_i * cz;

                constexpr real_t Hxx = D3Q27::Hxx<i>();
                constexpr real_t Hxy = D3Q27::Hxy<i>();
                constexpr real_t Hyy = D3Q27::Hyy<i>();
                constexpr real_t Hyz = D3Q27::Hyz<i>();
                constexpr real_t Hzz = D3Q27::Hzz<i>();
                constexpr real_t Hxz = D3Q27::Hxz<i>();

                Axx += g_i * Hxx;
                Axy += g_i * Hxy;
                Ayy += g_i * Hyy;
                Ayz += g_i * Hyz;
                Azz += g_i * Hzz;
                Axz += g_i * Hxz;

                return;
            }

            // Read moments from A at the source cell
            const real_t rhor_s = rhor[src];
            const real_t rhob_s = rhob[src];

            const real_t rho_s = rhor_s + rhob_s;
            const real_t inv_rhos = static_cast<real_t>(1) / (rhor_s + rhob_s);
            const real_t aR = rhor_s * inv_rhos;
            const real_t aB = rhob_s * inv_rhos;

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
            
            const real_t gieq = feq<i>(rho_s, ux_s, uy_s, uz_s);
            const real_t gineqr = fneqr<i>(Pixx_s, Pixy_s, Piyy_s, Piyz_s, Pizz_s, Pixz_s, ux_s, uy_s, uz_s);

            const real_t omega1 = gieq + oms * gineqr;

            // Interface indicator
            const real_t interface_indicator = static_cast<real_t>(4) * rhor_s * rhob_s * inv_rhos * inv_rhos;

            real_t gi = omega1;

            real_t Deltai = static_cast<real_t>(0);

            if (interface_indicator > static_cast<real_t>(1.0e-4) )
            {
                real_t Fx;
                real_t Fy;
                real_t Fz;

                real_t absF;
                real_t Acoef;

                preOmega2_outlet_limited(rhor, rhob, xs, ys, zs, Fx, Fy, Fz, absF, Acoef);

                // Second collision operator
                gi += Omega2<i>(Fx,Fy,Fz,absF, Acoef);

                // Recoloring operator
                Deltai = recolorDelta<i>(rhor_s, rhob_s,Fx,Fy,Fz,absF);
            }
            
            // Recolored red and blue incoming populations
            const real_t fr_i = aR * gi + Deltai;
            const real_t fb_i = aB * gi - Deltai;


            // New moments calculation
            sumr += fr_i;
            sumb += fb_i;

            const real_t g_i = fr_i + fb_i;

            constexpr real_t cx = static_cast<real_t>(D3Q27::cx<i>());
            constexpr real_t cy = static_cast<real_t>(D3Q27::cy<i>());
            constexpr real_t cz = static_cast<real_t>(D3Q27::cz<i>());

            jx += g_i * cx;
            jy += g_i * cy;
            jz += g_i * cz;

            constexpr real_t Hxx = D3Q27::Hxx<i>();
            constexpr real_t Hxy = D3Q27::Hxy<i>();
            constexpr real_t Hyy = D3Q27::Hyy<i>();
            constexpr real_t Hyz = D3Q27::Hyz<i>();
            constexpr real_t Hzz = D3Q27::Hzz<i>();
            constexpr real_t Hxz = D3Q27::Hxz<i>();

            Axx += g_i * Hxx;
            Axy += g_i * Hxy;
            Ayy += g_i * Hyy;
            Ayz += g_i * Hyz;
            Azz += g_i * Hzz;
            Axz += g_i * Hxz; });

    const real_t rhor_t = sumr;
    const real_t rhob_t = sumb;

    const real_t rho_t = rhor_t + rhob_t;
    const real_t inv_rho = static_cast<real_t>(1) / rho_t;

    const real_t vx = jx * inv_rho;
    const real_t vy = jy * inv_rho;
    const real_t vz = jz * inv_rho;

    rhor_next[id] = rhor_t;
    rhob_next[id] = rhob_t;

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

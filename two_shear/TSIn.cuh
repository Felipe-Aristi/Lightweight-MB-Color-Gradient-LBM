#ifndef TSIN_CUH
#define TSIN_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../memory.cuh"

__device__ __forceinline__ void initializeTS(MomentsDevice A, const label_t x, const label_t y, const label_t z)
{
    const label_t id = idx(x, y, z);

    const real_t xp = static_cast<real_t>(x - static_cast<label_t>(1)) +
                      static_cast<real_t>(0.5);

    const real_t yp = static_cast<real_t>(y - static_cast<label_t>(1)) +
                      static_cast<real_t>(0.5);

    const real_t Lx = static_cast<real_t>(NX - static_cast<label_t>(2));
    const real_t Ly = static_cast<real_t>(NY - static_cast<label_t>(2));

    const real_t y1 = static_cast<real_t>(0.25) * Ly;
    const real_t y2 = static_cast<real_t>(0.75) * Ly;

    real_t ux0;

    if (yp < static_cast<real_t>(0.5) * Ly)
    {
        ux0 = shear_U0 * tanhf((yp - y1) / shear_delta);
    }
    else
    {
        ux0 = shear_U0 * tanhf((y2 - yp) / shear_delta);
    }

    const real_t kx = static_cast<real_t>(2.0) * pi / Lx;

    const real_t dy1 = yp - y1;
    const real_t dy2 = yp - y2;

    const real_t delta2 = shear_delta * shear_delta;

    const real_t g1 = expf(-(dy1 * dy1) / (static_cast<real_t>(2.0) * delta2));

    const real_t g2 = expf(-(dy2 * dy2) / (static_cast<real_t>(2.0) * delta2));

    const real_t uy0 = shear_eps * shear_U0 * sinf(kx * xp) * (g1 + g2);

    A.rho[id] = rho0;

    A.ux[id] = ux0;
    A.uy[id] = uy0;
    A.uz[id] = static_cast<real_t>(0);

    A.Pixx[id] = static_cast<real_t>(0);
    A.Pixy[id] = static_cast<real_t>(0);
    A.Piyy[id] = static_cast<real_t>(0);
    A.Piyz[id] = static_cast<real_t>(0);
    A.Pizz[id] = static_cast<real_t>(0);
    A.Pixz[id] = static_cast<real_t>(0);
}

#endif

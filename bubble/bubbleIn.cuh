#ifndef BUBBLEIN_CUH
#define BUBBLEIN_CUH

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/bounds.cuh"
#include "../utilities/types.cuh"
#include "../utilities/indexing.cuh"

// =======================================================
// Bubble mask
// =======================================================

__device__ __forceinline__ void bubble_mask(MomentsDevice A, const label_t x, const label_t y, const label_t z) noexcept
{
    const label_t id = idx(x, y, z);

    const real_t dx = static_cast<real_t>(x) - bubble_x0;
    const real_t dy = static_cast<real_t>(y) - bubble_y0;
    const real_t dz = static_cast<real_t>(z) - bubble_z0;

    const real_t r2 = dx * dx + dy * dy + dz * dz;
    const real_t R2 = bubble_radius * bubble_radius;

    const real_t is_bubble = static_cast<real_t>(r2 <= R2);

    A.rhor[id] = (static_cast<real_t>(1) - is_bubble) * rhor0;
    A.rhob[id] = is_bubble * rhob0;
}

// =======================================================
// Static bubble initialization
// =======================================================

__device__ __forceinline__ void init_static_bubble(MomentsDevice A, const label_t x, const label_t y, const label_t z) noexcept
{
    const label_t id = idx(x, y, z);

    bubble_mask(A, x, y, z);

    A.ux[id] = static_cast<real_t>(0);
    A.uy[id] = static_cast<real_t>(0);
    A.uz[id] = static_cast<real_t>(0);

    A.Pixx[id] = static_cast<real_t>(0);
    A.Pixy[id] = static_cast<real_t>(0);
    A.Piyy[id] = static_cast<real_t>(0);
    A.Piyz[id] = static_cast<real_t>(0);
    A.Pizz[id] = static_cast<real_t>(0);
    A.Pixz[id] = static_cast<real_t>(0);
}

#endif
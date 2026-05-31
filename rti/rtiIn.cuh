#ifndef RTIIN_CUH
#define RTIIN_CUH

#include <cmath>

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"

// =======================================================
// Perturbed RTI interface
// y_interface(x,z) = y0 + a cos(kx x) cos(kz z)
// =======================================================

__device__ __forceinline__ real_t rti_interface_position(const label_t x, const label_t z) noexcept
{
    const real_t x_real = static_cast<real_t>(x);
    const real_t z_real = static_cast<real_t>(z);

    const real_t Lx = static_cast<real_t>(NX - static_cast<label_t>(2));
    const real_t Lz = static_cast<real_t>(NZ - static_cast<label_t>(2));

    const real_t kx = static_cast<real_t>(2.0) * pi * static_cast<real_t>(rti_mode_x) / Lx;

    const real_t kz = static_cast<real_t>(2.0) * pi * static_cast<real_t>(rti_mode_z) / Lz;

    return rti_y0 + rti_amplitude * cosf(kx * x_real) * cosf(kz * z_real);
}

__device__ __forceinline__ real_t rti_phase_field(const label_t x, const label_t y, const label_t z) noexcept
{
    const real_t y_real = static_cast<real_t>(y);
    const real_t y_interface = rti_interface_position(x, z);

    const real_t eta = (y_real - y_interface) / rti_interface_width;

    return tanhf(eta);
}

// =======================================================
// RTI domain initialization
// =======================================================

__device__ __forceinline__ void init_rti_domain(MomentsDevice A,
                                                const label_t x,
                                                const label_t y,
                                                const label_t z) noexcept
{
    const label_t id = idx(x, y, z);

    const real_t phi = rti_phase_field(x, y, z);

    A.rhor[id] = static_cast<real_t>(0.5) * (static_cast<real_t>(1) - phi) * rhor0;

    A.rhob[id] = static_cast<real_t>(0.5) * (static_cast<real_t>(1) + phi) * rhob0;

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

// =======================================================
// Wall initialization
// =======================================================

__device__ __forceinline__ void init_rti_y_walls(MomentsDevice A,
                                                 const label_t x,
                                                 const label_t z) noexcept
{
    constexpr label_t yBottomB = static_cast<label_t>(0);
    constexpr label_t yBottomF = static_cast<label_t>(1);

    constexpr label_t yTopB = NY - static_cast<label_t>(1);
    constexpr label_t yTopF = NY - static_cast<label_t>(2);

    const label_t idBottomB = idx(x, yBottomB, z);
    const label_t idBottomF = idx(x, yBottomF, z);

    const label_t idTopB = idx(x, yTopB, z);
    const label_t idTopF = idx(x, yTopF, z);

    A.rhor[idBottomB] = A.rhor[idBottomF];
    A.rhob[idBottomB] = A.rhob[idBottomF];

    A.ux[idBottomB] = static_cast<real_t>(0);
    A.uy[idBottomB] = static_cast<real_t>(0);
    A.uz[idBottomB] = static_cast<real_t>(0);

    A.Pixx[idBottomB] = A.Pixx[idBottomF];
    A.Pixy[idBottomB] = A.Pixy[idBottomF];
    A.Piyy[idBottomB] = A.Piyy[idBottomF];
    A.Piyz[idBottomB] = A.Piyz[idBottomF];
    A.Pizz[idBottomB] = A.Pizz[idBottomF];
    A.Pixz[idBottomB] = A.Pixz[idBottomF];

    A.rhor[idTopB] = A.rhor[idTopF];
    A.rhob[idTopB] = A.rhob[idTopF];

    A.ux[idTopB] = static_cast<real_t>(0);
    A.uy[idTopB] = static_cast<real_t>(0);
    A.uz[idTopB] = static_cast<real_t>(0);

    A.Pixx[idTopB] = A.Pixx[idTopF];
    A.Pixy[idTopB] = A.Pixy[idTopF];
    A.Piyy[idTopB] = A.Piyy[idTopF];
    A.Piyz[idTopB] = A.Piyz[idTopF];
    A.Pizz[idTopB] = A.Pizz[idTopF];
    A.Pixz[idTopB] = A.Pixz[idTopF];
}

#endif
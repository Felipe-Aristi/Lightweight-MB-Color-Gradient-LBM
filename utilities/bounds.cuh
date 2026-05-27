#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include "../constants.cuh"
#include "indexing.cuh"

// =======================================================
// Physical domain
// =======================================================

__host__ __device__ [[nodiscard]] constexpr inline bool interior(const label_t x,
                                                                 const label_t y,
                                                                 const label_t z) noexcept
{
    return (x >= NX || y >= NY || z >= NZ ||
            x == static_cast<label_t>(0) || x == NX - static_cast<label_t>(1) ||
            y == static_cast<label_t>(0) || y == NY - static_cast<label_t>(1) ||
            z == static_cast<label_t>(0) || z == NZ - static_cast<label_t>(1));
}

__device__ [[nodiscard]] constexpr inline bool inlet_outlet_interior(const label_t x,
                                                                     const label_t z) noexcept
{
    return (x >= NX || z >= NZ ||
            x == static_cast<label_t>(0) || x == NX - static_cast<label_t>(1) ||
            z == static_cast<label_t>(0) || z == NZ - static_cast<label_t>(1));
}

// =======================================================
//  Periodic boundary conditions
// =======================================================

__host__ __device__ [[nodiscard]] constexpr inline label_t wrapx(const label_t x) noexcept
{
    if (x == static_cast<label_t>(0))
    {
        return NX - static_cast<label_t>(2);
    }

    if (x == NX - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return x;
}

__host__ __device__ [[nodiscard]] constexpr inline label_t wrapy(const label_t y) noexcept
{
    if (y == static_cast<label_t>(0))
    {
        return NY - static_cast<label_t>(2);
    }

    if (y == NY - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return y;
}

__host__ __device__ [[nodiscard]] constexpr inline label_t wrapz(const label_t z) noexcept
{
    if (z == static_cast<label_t>(0))
    {
        return NZ - static_cast<label_t>(2);
    }

    if (z == NZ - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return z;
}


// =======================================================
// Pull index
// =======================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr inline label_t pullx(const label_t x) noexcept
{
    return static_cast<label_t>(
        static_cast<int>(x) - D3Q27::cx<I>());
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr inline label_t pully(const label_t y) noexcept
{
    return static_cast<label_t>(
        static_cast<int>(y) - D3Q27::cy<I>());
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr inline label_t pullz(const label_t z) noexcept
{
    return static_cast<label_t>(
        static_cast<int>(z) - D3Q27::cz<I>());
}

template <label_t I>
__device__ [[nodiscard]] __forceinline__ label_t pullid(const label_t x,
                                                        const label_t y,
                                                        const label_t z) noexcept
{
    return idx(
        pullx<I>(x),
        pully<I>(y),
        pullz<I>(z));
}

// =======================================================
// Pull cases
// =======================================================


template <label_t I>
__device__ [[nodiscard]] __forceinline__ label_t pullidPeri(const label_t x,
                                                            const label_t y,
                                                            const label_t z) noexcept
{
    return idx(
        wrapx(pullx<I>(x)),
        wrapy(pully<I>(y)),
        wrapz(pullz<I>(z)));

    // wrapy(pully<I>(y)) ,  pully<I>(y)
}

template <label_t I>
__device__ [[nodiscard]] __forceinline__ label_t pullidJet(const label_t x,
                                                            const label_t y,
                                                            const label_t z) noexcept
{
    return idx(
        wrapx(pullx<I>(x)),
        pully<I>(y),
        wrapz(pullz<I>(z)));

}



#endif

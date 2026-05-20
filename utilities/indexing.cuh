#ifndef INDEXING_CUH
#define INDEXING_CUH

#include <cstddef>
#include <cmath>

#include "../constants.cuh"
#include "types.cuh"

__host__ __device__ [[nodiscard]] constexpr inline label_t idx(const label_t x, const label_t y, const label_t z) noexcept
{
    return x + NX * (y + NY * z);
}

#endif
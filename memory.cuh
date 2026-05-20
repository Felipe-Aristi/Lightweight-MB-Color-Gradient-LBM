#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <cuda_runtime.h>
#include <utility>

#include "constants.cuh"
#include "utilities/cudaUtilities.cuh"

// =======================================================
// Device moments
// =======================================================

struct MomentsDevice
{
    real_t *rho = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;

    real_t *Pixx = nullptr;
    real_t *Pixy = nullptr;
    real_t *Piyy = nullptr;
    real_t *Piyz = nullptr;
    real_t *Pizz = nullptr;
    real_t *Pixz = nullptr;
};

// =======================================================
// Main device structure
// =======================================================

struct LbmDevice
{
    MomentsDevice A; // current state
    MomentsDevice B; // next state
};

// =======================================================
// Host buffers for output
// =======================================================

struct LbmHost
{
    real_t *rho = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;
};

// =======================================================
// Internal allocation helpers
// =======================================================

inline void allocate_device(MomentsDevice &d)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.rho), bytesCell));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.ux), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.uy), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.uz), bytesCell));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Pixx), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Pixy), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Piyy), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Piyz), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Pizz), bytesCell));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.Pixz), bytesCell));
}

inline void zero_device(const MomentsDevice &d)
{
    CUDA_CHECK(cudaMemset(d.rho, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.ux, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uz, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.Pixx, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyz, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pizz, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixz, 0, bytesCell));
}

inline void free_device(MomentsDevice &d)
{
    if (d.rho)
        CUDA_CHECK(cudaFree(d.rho));

    if (d.ux)
        CUDA_CHECK(cudaFree(d.ux));
    if (d.uy)
        CUDA_CHECK(cudaFree(d.uy));
    if (d.uz)
        CUDA_CHECK(cudaFree(d.uz));

    if (d.Pixx)
        CUDA_CHECK(cudaFree(d.Pixx));
    if (d.Pixy)
        CUDA_CHECK(cudaFree(d.Pixy));
    if (d.Piyy)
        CUDA_CHECK(cudaFree(d.Piyy));
    if (d.Piyz)
        CUDA_CHECK(cudaFree(d.Piyz));
    if (d.Pizz)
        CUDA_CHECK(cudaFree(d.Pizz));
    if (d.Pixz)
        CUDA_CHECK(cudaFree(d.Pixz));

    d = MomentsDevice{};
}

// =======================================================
// Host allocation
// =======================================================

inline LbmHost allocate_host_memory()
{
    LbmHost h{};

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h.rho), bytesCell));

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h.ux), bytesCell));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h.uy), bytesCell));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h.uz), bytesCell));

    return h;
}

// =======================================================
// Device allocation
// =======================================================

inline LbmDevice allocate_device_memory()
{
    LbmDevice d{};

    allocate_device(d.A);
    allocate_device(d.B);

    zero_device(d.A);
    zero_device(d.B);

    return d;
}

// =======================================================
// Free memory
// =======================================================

inline void free_host_memory(LbmHost &h)
{
    if (h.rho)
        CUDA_CHECK(cudaFreeHost(h.rho));

    if (h.ux)
        CUDA_CHECK(cudaFreeHost(h.ux));
    if (h.uy)
        CUDA_CHECK(cudaFreeHost(h.uy));
    if (h.uz)
        CUDA_CHECK(cudaFreeHost(h.uz));

    h = LbmHost{};
}

inline void free_device_memory(LbmDevice &d)
{
    free_device(d.A);
    free_device(d.B);

    d = LbmDevice{};
}

// =======================================================
// Swap moments
// =======================================================

inline void swap_moments(MomentsDevice &A, MomentsDevice &B) noexcept
{
    std::swap(A, B);
}

// =======================================================
// Copy device to host for output
// =======================================================

inline void copy_out_D2H(LbmHost &h, const MomentsDevice &d)
{
    CUDA_CHECK(cudaMemcpy(h.rho, d.rho, bytesCell, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h.ux, d.ux, bytesCell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uy, d.uy, bytesCell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uz, d.uz, bytesCell, cudaMemcpyDeviceToHost));
}

#endif
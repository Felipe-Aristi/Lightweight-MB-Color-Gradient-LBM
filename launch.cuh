#ifndef LAUNCH_CUH
#define LAUNCH_CUH

#include "memory.cuh"
#include "kernels.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"

// =======================================================
// Bubble initialization
// =======================================================

inline void launch_InitStaticBubble(MomentsDevice A, const CudaConfig &cfg)
{
    InitStaticBubble<<<cfg.grid, cfg.block>>>(A);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//==================================================
// Jet initialization
// =======================================================

inline void launch_InitJetBulk(MomentsDevice A, const CudaConfig &cfg)
{
    InitJetBulk<<<cfg.grid, cfg.block>>>(A);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

inline void launch_InitJetInlet(MomentsDevice A, const CudaConfig &cfg)
{
    InitJetInlet<<<grid2D_xz(cfg), block2D_xz(cfg)>>>(A);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

inline void launch_InitJet(MomentsDevice A, const CudaConfig &cfg)
{
    launch_InitJetBulk(A, cfg);
    launch_InitJetInlet(A, cfg);
}

// =======================================================
// Jet boundary conditions
// =======================================================

inline void launch_JetBoundaryConditions(MomentsDevice B, const MomentsDevice Aold, const CudaConfig &cfg)
{
    JetBoundaryConditions<<<grid2D_xz(cfg), block2D_xz(cfg)>>>(B, Aold);

    CUDA_CHECK(cudaGetLastError());
}

// =======================================================
// Main RCS step
// =======================================================

inline void launch_RCS(const MomentsDevice A, MomentsDevice B, const CudaConfig &cfg)
{
    RCSKernel<<<cfg.grid, cfg.block>>>(A, B);

    CUDA_CHECK(cudaGetLastError());
}

#endif

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "memory.cuh"

// =======================================================
// Initialization kernels
// =======================================================

__global__ void InitStaticBubble(MomentsDevice A);

__global__ void InitJetBulk(MomentsDevice A);

__global__ void InitJetInlet(MomentsDevice A);

// =======================================================
// Boundary-condition kernels
// =======================================================

__global__ void JetBoundaryConditions(MomentsDevice B, const MomentsDevice Aold);

// =======================================================
// Main RCS kernel
// =======================================================

__global__ void RCSKernel(const MomentsDevice A, MomentsDevice B);

#endif

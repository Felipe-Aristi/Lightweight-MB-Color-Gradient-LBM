#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "memory.cuh"

// =======================================================
// Initialization kernels
// =======================================================

__global__ void InitStaticBubble(MomentsDevice A);

// __global__ void InitJetBulk(MomentsDevice A);

// __global__ void InitJetInlet(MomentsDevice A);

// =======================================================
// Main kernels RCS and BCs
// =======================================================

// __global__ void InletBC(MomentsDevice A);

// __global__ void OutletNeumannBC(MomentsDevice A);

// __global__ void JetBoundaryConditions(MomentsDevice A);

__global__ void RCSKernel(const MomentsDevice A, MomentsDevice B);

#endif

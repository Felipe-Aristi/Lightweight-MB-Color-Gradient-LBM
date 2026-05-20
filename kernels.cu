#include <cmath>

#include "constants.cuh"
#include "memory.cuh"
#include "kernels.cuh"
#include "lbm.cuh"

#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"

#include "two_shear/TSIn.cuh"
#include "jet/jetIn.cuh"
#include "jet/boundary_conditions.cuh"

// =======================================================
// Two-shear-layer initialization
// =======================================================

__global__ void InitTwoShearLayers(MomentsDevice A)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    initializeTS(A, x, y, z);
}

// =======================================================
// JET initialization
// =======================================================

__global__ void InitJetBulk(MomentsDevice A)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    init_jet_bulk(A, x, y, z);
}

__global__ void InitJetInlet(MomentsDevice A)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    init_jet_inlet(A, x, z);
}

// =======================================================
// Jet boundary conditions
// =======================================================

__global__ void InletBC(MomentsDevice A)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    inlet_bc_calculation(A, x, z);
}

__global__ void OutletNeumannBC(MomentsDevice A)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    outlet_neumann_bc_calculation(A, x, z);
}

// =======================================================
// Main RCS kernel
// =======================================================

__global__ void RCSKernel(const MomentsDevice A, MomentsDevice B)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    RCS(A, B, x, y, z);
}

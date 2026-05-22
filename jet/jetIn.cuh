// #ifndef JETIN_CUH
// #define JETIN_CUH

// #include "../utilities/bounds.cuh"
// #include "../utilities/indexing.cuh"
// #include "../utilities/types.cuh"
// #include "../utilities/mathUtilities.cuh"

// #include "../constants.cuh"
// #include "../memory.cuh"

// // =======================================================
// // Initialize bulk domain
// // =======================================================

// __device__ __forceinline__ void init_jet_bulk(MomentsDevice A, const label_t x, const label_t y, const label_t z) noexcept
// {
//     const label_t id = idx(x, y, z);

//     A.rho[id] = rho0;

//     A.ux[id] = static_cast<real_t>(0);
//     A.uy[id] = static_cast<real_t>(0);
//     A.uz[id] = static_cast<real_t>(0);

//     A.Pixx[id] = static_cast<real_t>(0);
//     A.Pixy[id] = static_cast<real_t>(0);
//     A.Piyy[id] = static_cast<real_t>(0);
//     A.Piyz[id] = static_cast<real_t>(0);
//     A.Pizz[id] = static_cast<real_t>(0);
//     A.Pixz[id] = static_cast<real_t>(0);
// }

// // =======================================================
// // Initialize inlet
// // =======================================================

// __device__ __forceinline__ void init_jet_inlet(MomentsDevice A, const label_t x, const label_t z) noexcept
// {
//     constexpr label_t yB = static_cast<label_t>(0);

//     const label_t id = idx(x, yB, z);

//     const real_t uy0 = isJet(x, z) * jet_velocity;

//     A.rho[id] = rho0;

//     A.ux[id] = static_cast<real_t>(0);
//     A.uy[id] = uy0;
//     A.uz[id] = static_cast<real_t>(0);

//     A.Pixx[id] = static_cast<real_t>(0);
//     A.Pixy[id] = static_cast<real_t>(0);
//     A.Piyy[id] = static_cast<real_t>(0);
//     A.Piyz[id] = static_cast<real_t>(0);
//     A.Pizz[id] = static_cast<real_t>(0);
//     A.Pixz[id] = static_cast<real_t>(0);
// }

// #endif
#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <array>
#include <cstddef>
#include "utilities/types.cuh"

// Steps
inline constexpr int NSTEP = 5000000;
inline constexpr int NOUTPUT = 1000;
inline constexpr int NSTATS_SAMPLE = 20;

// Grid
inline constexpr label_t NX = static_cast<label_t>(128);
inline constexpr label_t NZ = static_cast<label_t>(128);
inline constexpr label_t NY = static_cast<label_t>(400);
inline constexpr label_t NXNY = NX * NY;
inline constexpr label_t sponge_cells = static_cast<label_t>(34);

inline constexpr label_t Ncells = NX * NY * NZ;

// VELOCITY SET D2Q27 definition
inline constexpr label_t Q = 27;

// Memory sizes
inline constexpr std::size_t bytesCell = std::size_t(Ncells) * sizeof(real_t);

// Jet parameters
inline constexpr real_t jet_radius = static_cast<real_t>(7.0);
inline constexpr real_t jet_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_velocity = static_cast<real_t>(0.05);

// Two-shear-layer initial condition
inline constexpr real_t pi = static_cast<real_t>(3.14159265358979323846);

inline constexpr real_t shear_U0 = static_cast<real_t>(0.03);
inline constexpr real_t shear_delta = static_cast<real_t>(6.0);
inline constexpr real_t shear_eps = static_cast<real_t>(0.05);

// Statistics
inline constexpr real_t stats_start_tstar = static_cast<real_t>(7000.0);

// Some useful constans
inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t cs6 = cs4 * cs2;

inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);
inline constexpr real_t inv_6cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(6.0) * cs6);
inline constexpr real_t inv_2cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(2.0) * cs6);

// Fluidparameters
inline constexpr real_t rho0 = static_cast<real_t>(1);

inline constexpr real_t Re = static_cast<real_t>(5000);
inline constexpr real_t nu = (static_cast<real_t>(2) * jet_radius * jet_velocity) / Re;

inline constexpr real_t tau = static_cast<real_t>(0.5) + nu / (cs2); // static_cast<real_t>(0.6)

inline constexpr real_t omega = static_cast<real_t>(1) / tau;
inline constexpr real_t oms = static_cast<real_t>(1) - omega;

#endif

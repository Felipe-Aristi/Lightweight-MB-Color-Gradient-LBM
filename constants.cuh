#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <array>
#include <cstddef>
#include "utilities/types.cuh"

// Steps
inline constexpr int NSTEP = 300000;
inline constexpr int NOUTPUT = 10000;
inline constexpr int NSTATS_SAMPLE = 20;

// Grid
inline constexpr label_t NX = static_cast<label_t>(128);
inline constexpr label_t NZ = static_cast<label_t>(128);
inline constexpr label_t NY = static_cast<label_t>(128);
inline constexpr label_t NXNY = NX * NY;

inline constexpr label_t Ncells = NX * NY * NZ;

// Velocity set selection
#if defined(LBM_D3Q19)
inline constexpr label_t Q = 19;
#else
inline constexpr label_t Q = 27;
#endif

// Memory sizes
inline constexpr std::size_t bytesCell = std::size_t(Ncells) * sizeof(real_t);

// Jet parameters
inline constexpr real_t jet_radius = static_cast<real_t>(12.0);
inline constexpr real_t jet_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_velocity = static_cast<real_t>(0.05);

// Bubble parameters
inline constexpr real_t bubble_radius = static_cast<real_t>(30.0);
inline constexpr real_t bubble_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t bubble_y0 = static_cast<real_t>(NY - 1) / static_cast<real_t>(2);
inline constexpr real_t bubble_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t pi = static_cast<real_t>(3.141592653589793);

// Rayleigh Taylor Instability parameters
inline constexpr real_t rti_y0 = static_cast<real_t>(NY - 1) / static_cast<real_t>(2);
inline constexpr real_t rti_amplitude = static_cast<real_t>(4.0);
inline constexpr real_t gravity = static_cast<real_t>(1.0e-6);
inline constexpr real_t rti_interface_width = static_cast<real_t>(4.0);
inline constexpr int rti_mode_x = 1;
inline constexpr int rti_mode_z = 1;

// Statistics
inline constexpr real_t stats_start_tstar = static_cast<real_t>(7000.0);

// Some useful constans
inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t cs6 = cs4 * cs2;

inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t inv_cs4 = static_cast<real_t>(1) / cs4;
inline constexpr real_t inv_cs6 = static_cast<real_t>(1) / cs6;

inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);
inline constexpr real_t inv_6cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(6.0) * cs6);
inline constexpr real_t inv_2cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(2.0) * cs6);

inline constexpr real_t half = static_cast<real_t>(0.5);

// FLuid parameters
inline constexpr real_t rhor0 = static_cast<real_t>(1);
inline constexpr real_t rhob0 = static_cast<real_t>(1);

inline constexpr real_t Re = static_cast<real_t>(100);
inline constexpr real_t nu = (static_cast<real_t>(2) * jet_radius * jet_velocity) / Re;

inline constexpr real_t taur = static_cast<real_t>(1); // static_cast<real_t>(0.5) + nu / (cs2); // static_cast<real_t>(0.6)

inline constexpr real_t taub = static_cast<real_t>(1); // static_cast<real_t>(0.5) + nu / (cs2);

inline constexpr real_t omegar = static_cast<real_t>(1) / taur;
inline constexpr real_t omegab = static_cast<real_t>(1) / taub;

inline constexpr real_t oms = static_cast<real_t>(1) - omegab;

// Weber number
inline constexpr real_t We = static_cast<real_t>(2500);

inline constexpr real_t sigma = static_cast<real_t>(0.1); // static_cast<real_t>((rhob0 * jet_velocity * jet_velocity * static_cast<real_t>(2.0) * jet_radius) / We);
inline constexpr real_t beta_recolor = static_cast<real_t>(0.90);

inline constexpr real_t sigma_over_4cs4 = sigma / (static_cast<real_t>(4) * cs4);

#endif

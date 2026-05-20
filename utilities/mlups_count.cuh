#ifndef MLUPS_COUNT_CUH
#define MLUPS_COUNT_CUH

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include "../constants.cuh"
#include "cudaUtilities.cuh"

struct MlupsCounter
{
    std::uint64_t cells_per_step = 0;
    std::uint64_t steps = 0;
    float elapsed_ms = 0.0f;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    bool running = false;
};

[[nodiscard]] constexpr inline std::uint64_t mlups_active_cells() noexcept
{
    return static_cast<std::uint64_t>(NX - static_cast<label_t>(2)) *
           static_cast<std::uint64_t>(NY - static_cast<label_t>(2)) *
           static_cast<std::uint64_t>(NZ - static_cast<label_t>(2));
}

[[nodiscard]] inline MlupsCounter make_mlups_counter(
    const std::uint64_t cells_per_step = mlups_active_cells())
{
    MlupsCounter counter{};
    counter.cells_per_step = cells_per_step;
    CUDA_CHECK(cudaEventCreate(&counter.start_event));
    CUDA_CHECK(cudaEventCreate(&counter.stop_event));
    return counter;
}

inline void mlups_start(MlupsCounter &counter)
{
    counter.steps = 0;
    counter.elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(counter.start_event));
    counter.running = true;
}

inline void mlups_pause(MlupsCounter &counter)
{
    if (!counter.running)
    {
        return;
    }

    CUDA_CHECK(cudaEventRecord(counter.stop_event));
    CUDA_CHECK(cudaEventSynchronize(counter.stop_event));

    float window_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &window_ms,
        counter.start_event,
        counter.stop_event));

    counter.elapsed_ms += window_ms;
    counter.running = false;
}

inline void mlups_resume(MlupsCounter &counter)
{
    if (counter.running)
    {
        return;
    }

    CUDA_CHECK(cudaEventRecord(counter.start_event));
    counter.running = true;
}

inline void mlups_stop(MlupsCounter &counter)
{
    mlups_pause(counter);
}

inline void mlups_destroy(MlupsCounter &counter)
{
    if (counter.start_event != nullptr)
    {
        CUDA_CHECK(cudaEventDestroy(counter.start_event));
        counter.start_event = nullptr;
    }

    if (counter.stop_event != nullptr)
    {
        CUDA_CHECK(cudaEventDestroy(counter.stop_event));
        counter.stop_event = nullptr;
    }
}

inline void mlups_count_step(MlupsCounter &counter,
                             const std::uint64_t step_count = 1) noexcept
{
    counter.steps += step_count;
}

[[nodiscard]] inline double mlups_elapsed_seconds(MlupsCounter &counter)
{
    float elapsed_ms = counter.elapsed_ms;

    if (counter.running)
    {
        CUDA_CHECK(cudaEventRecord(counter.stop_event));
        CUDA_CHECK(cudaEventSynchronize(counter.stop_event));

        float window_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &window_ms,
            counter.start_event,
            counter.stop_event));

        elapsed_ms += window_ms;
    }

    return static_cast<double>(elapsed_ms) * 1.0e-3;
}

[[nodiscard]] inline std::uint64_t mlups_lattice_updates(
    const MlupsCounter &counter) noexcept
{
    return counter.cells_per_step * counter.steps;
}

[[nodiscard]] inline double mlups_value(MlupsCounter &counter)
{
    const double seconds = mlups_elapsed_seconds(counter);

    if (seconds <= 0.0)
    {
        return 0.0;
    }

    return static_cast<double>(mlups_lattice_updates(counter)) /
           (seconds * 1.0e6);
}

inline void mlups_print(MlupsCounter &counter,
                        std::ostream &out = std::cout)
{
    const std::ios::fmtflags old_flags = out.flags();
    const std::streamsize old_precision = out.precision();

    out << std::fixed << std::setprecision(3)
        << "MLUPS: " << mlups_value(counter)
        << " | elapsed: " << mlups_elapsed_seconds(counter) << " s"
        << " | steps: " << counter.steps
        << " | cells/step: " << counter.cells_per_step
        << '\n';

    out.flags(old_flags);
    out.precision(old_precision);
}

inline void mlups_print_progress(MlupsCounter &counter,
                                 const int step,
                                 const int final_step = NSTEP,
                                 std::ostream &out = std::cout)
{
    const std::ios::fmtflags old_flags = out.flags();
    const std::streamsize old_precision = out.precision();

    const double percent =
        final_step > 0 ? 100.0 * static_cast<double>(step) /
                             static_cast<double>(final_step)
                       : 0.0;

    out << '\r'
        << "Step: " << step << " / " << final_step
        << " (" << std::fixed << std::setprecision(2) << percent << "%)"
        << " | MLUPS: " << std::setprecision(3) << mlups_value(counter)
        << " | elapsed: " << std::setprecision(1)
        << mlups_elapsed_seconds(counter) << " s"
        << "        " << std::flush;

    out.flags(old_flags);
    out.precision(old_precision);
}

inline void mlups_finish_progress(std::ostream &out = std::cout)
{
    out << '\n';
}

struct MlupsPause
{
    explicit MlupsPause(MlupsCounter &counter_in)
        : counter(counter_in)
    {
        was_running = counter.running;
        mlups_pause(counter);
    }

    ~MlupsPause()
    {
        if (was_running)
        {
            mlups_resume(counter);
        }
    }

    MlupsPause(const MlupsPause &) = delete;
    MlupsPause &operator=(const MlupsPause &) = delete;

private:
    MlupsCounter &counter;
    bool was_running = false;
};

#endif

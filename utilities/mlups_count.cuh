#ifndef MLUPS_COUNT_CUH
#define MLUPS_COUNT_CUH

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "../constants.cuh"

struct MlupsCounter
{
    using clock = std::chrono::steady_clock;

    std::uint64_t cells_per_step = 0;
    std::uint64_t steps = 0;
    std::chrono::duration<double> elapsed{};
    clock::time_point last_start{};
    bool running = false;
};

[[nodiscard]] constexpr inline std::uint64_t mlups_active_cells() noexcept
{
    return static_cast<std::uint64_t>(NX - static_cast<label_t>(2)) *
           static_cast<std::uint64_t>(NY - static_cast<label_t>(2)) *
           static_cast<std::uint64_t>(NZ - static_cast<label_t>(2));
}

[[nodiscard]] inline MlupsCounter make_mlups_counter(
    const std::uint64_t cells_per_step = mlups_active_cells()) noexcept
{
    MlupsCounter counter{};
    counter.cells_per_step = cells_per_step;
    return counter;
}

inline void mlups_start(MlupsCounter &counter)
{
    counter.steps = 0;
    counter.elapsed = std::chrono::duration<double>{0.0};
    counter.last_start = MlupsCounter::clock::now();
    counter.running = true;
}

inline void mlups_pause(MlupsCounter &counter)
{
    if (!counter.running)
    {
        return;
    }

    counter.elapsed += MlupsCounter::clock::now() - counter.last_start;
    counter.running = false;
}

inline void mlups_resume(MlupsCounter &counter)
{
    if (counter.running)
    {
        return;
    }

    counter.last_start = MlupsCounter::clock::now();
    counter.running = true;
}

inline void mlups_stop(MlupsCounter &counter)
{
    mlups_pause(counter);
}

inline void mlups_count_step(MlupsCounter &counter,
                             const std::uint64_t step_count = 1) noexcept
{
    counter.steps += step_count;
}

[[nodiscard]] inline double mlups_elapsed_seconds(const MlupsCounter &counter)
{
    std::chrono::duration<double> elapsed = counter.elapsed;

    if (counter.running)
    {
        elapsed += MlupsCounter::clock::now() - counter.last_start;
    }

    return elapsed.count();
}

[[nodiscard]] inline std::uint64_t mlups_lattice_updates(
    const MlupsCounter &counter) noexcept
{
    return counter.cells_per_step * counter.steps;
}

[[nodiscard]] inline double mlups_value(const MlupsCounter &counter)
{
    const double seconds = mlups_elapsed_seconds(counter);

    if (seconds <= 0.0)
    {
        return 0.0;
    }

    return static_cast<double>(mlups_lattice_updates(counter)) /
           (seconds * 1.0e6);
}

inline void mlups_print(const MlupsCounter &counter,
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

inline void mlups_print_progress(const MlupsCounter &counter,
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

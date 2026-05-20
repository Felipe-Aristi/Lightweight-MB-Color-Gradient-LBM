#include <cuda_runtime.h>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"
#include "utilities/mlups_count.cuh"

#include "io/vtiWriter.cuh"

int main()
{
    constexpr int deviceID = 0;
    constexpr int final_step = NSTEP - 1;

    CUDA_CHECK(cudaSetDevice(deviceID));

    CudaConfig cfg = print_device_and_make_config(deviceID);

    LbmDevice d = allocate_device_memory();
    LbmHost h = allocate_host_memory();

    // launch_InitTwoShearLayers(d.A, cfg);
    launch_InitJet(d.A, cfg);

    MlupsCounter mlups = make_mlups_counter();
    mlups_start(mlups);

    {
        const MlupsPause output_pause(mlups);
        mlups_print_progress(mlups, 0, final_step);
        write_midplane_jet_vti_step_device(0, d.A);
    }

    for (int step = 1; step < NSTEP; ++step)
    {
        launch_JetBoundaryConditions(d.A, cfg);

        launch_RCS(d.A, d.B, cfg);

        swap_moments(d.A, d.B);
        mlups_count_step(mlups);

        if (NOUTPUT > 0 && step % NOUTPUT == 0)
        {
            const MlupsPause output_pause(mlups);
            mlups_print_progress(mlups, step, final_step);
            // write_vti_step_device(step, d.A, h);
            write_midplane_jet_vti_step_device(step, d.A);
        }
    }

    mlups_stop(mlups);
    mlups_print_progress(mlups, final_step, final_step);
    mlups_finish_progress();
    mlups_print(mlups);
    mlups_destroy(mlups);

    free_host_memory(h);
    free_device_memory(d);

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

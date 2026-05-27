#ifndef VTIWRITER_CUH
#define VTIWRITER_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/types.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/cudaUtilities.cuh"

// =======================================================
// Output directory for jet case
// =======================================================

inline std::filesystem::path default_out_dir()
{
    std::ostringstream folder_name;

    folder_name << "Re"
                << static_cast<int>(std::round(Re))
                << "_We"
                << static_cast<int>(std::round(We))
                << "_jet_vtifiles";

    return std::filesystem::current_path() / "JET_VTK" / folder_name.str();
}

// =======================================================
// Step filename
// =======================================================

inline std::filesystem::path vti_filename(
    const int step,
    std::filesystem::path folder = default_out_dir())
{
    std::ostringstream name;

    name << "jet_"
         << std::setw(8) << std::setfill('0') << step
         << ".vti";

    return folder / name.str();
}

// =======================================================
// Write one scalar field in VTI ASCII format
// =======================================================

inline void write_vti_scalar(
    std::ofstream &out,
    const std::string &name,
    const real_t *field)
{
    out << "        <DataArray type=\"Float32\" Name=\"" << name
        << "\" format=\"ascii\">\n";

    out << std::scientific << std::setprecision(8);

    for (label_t z = 0; z < NZ; ++z)
    {
        for (label_t y = 0; y < NY; ++y)
        {
            for (label_t x = 0; x < NX; ++x)
            {
                const label_t id = idx(x, y, z);
                out << static_cast<float>(field[id]) << " ";
            }

            out << "\n";
        }
    }

    out << "        </DataArray>\n";
}

// =======================================================
// Minimal VTI writer
// Saves only: rhor, rhob, ux, uy, uz
// =======================================================

inline void write_vti(
    const std::filesystem::path &filename,
    const LbmHost &h)
{
    if (!filename.parent_path().empty())
    {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename);

    if (!out)
    {
        throw std::runtime_error(
            "Cannot open VTI file for writing: " + filename.string());
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

    out << "  <ImageData WholeExtent=\"0 " << NX - 1
        << " 0 " << NY - 1
        << " 0 " << NZ - 1
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    out << "    <Piece Extent=\"0 " << NX - 1
        << " 0 " << NY - 1
        << " 0 " << NZ - 1
        << "\">\n";

    out << "      <PointData Scalars=\"rhor\">\n";

    write_vti_scalar(out, "rhor", h.rhor);
    write_vti_scalar(out, "rhob", h.rhob);

    write_vti_scalar(out, "ux", h.ux);
    write_vti_scalar(out, "uy", h.uy);
    write_vti_scalar(out, "uz", h.uz);

    out << "      </PointData>\n";
    out << "      <CellData>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error(
            "Error while finalizing VTI file: " + filename.string());
    }

    std::cout << "Wrote VTI file: " << filename << "\n";
}

// =======================================================
// Device-to-host copy and VTI output
// =======================================================

inline void write_vti_step_device(
    const int step,
    const MomentsDevice &d,
    LbmHost &h,
    std::filesystem::path folder = default_out_dir())
{
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_out_D2H(h, d);

    write_vti(
        vti_filename(step, std::move(folder)),
        h);
}

#endif
#ifndef VTIWRITER_CUH
#define VTIWRITER_CUH

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/types.cuh"
#include "../utilities/cudaUtilities.cuh"
#include "../utilities/indexing.cuh"

// =======================================================
// Output directories
// =======================================================

inline std::filesystem::path default_out_dir()
{
    std::ostringstream folder_name;

    folder_name << "Re"
                << static_cast<int>(std::round(Re))
                << "_vtifiles";

    return std::filesystem::current_path() / "JET_VTK" / folder_name.str();
}

inline std::filesystem::path default_slice_out_dir()
{
    std::ostringstream folder_name;

    folder_name << "Re"
                << static_cast<int>(std::round(Re))
                << "_midplane_vtifiles";

    return std::filesystem::current_path() / "JET_VTK" / folder_name.str();
}

// =======================================================
// VTK type
// =======================================================

inline const char *vtk_real_type()
{
    if constexpr (std::is_same_v<real_t, float>)
    {
        return "Float32";
    }
    else
    {
        return "Float64";
    }
}

// =======================================================
// Base64 encoder
// =======================================================

inline std::string base64_encode(
    const unsigned char *data,
    const std::size_t len)
{
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string out;

    out.reserve(((len + 2) / 3) * 4);

    for (std::size_t i = 0; i < len; i += 3)
    {
        const std::uint32_t b0 = data[i];
        const std::uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        const std::uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;

        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(table[(triple >> 18) & 0x3F]);
        out.push_back(table[(triple >> 12) & 0x3F]);

        if (i + 1 < len)
        {
            out.push_back(table[(triple >> 6) & 0x3F]);
        }
        else
        {
            out.push_back('=');
        }

        if (i + 2 < len)
        {
            out.push_back(table[triple & 0x3F]);
        }
        else
        {
            out.push_back('=');
        }
    }

    return out;
}

// =======================================================
// Binary array encoding
// =======================================================
//
// VTK XML binary format expects:
//
// [UInt64 byte_count][raw bytes]
//
// and the whole block is Base64 encoded.

inline std::string encode_scalar_array_binary(
    const real_t *data,
    const std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) *
        static_cast<std::uint64_t>(sizeof(real_t));

    std::vector<unsigned char> buffer(
        sizeof(std::uint64_t) + static_cast<std::size_t>(nbytes));

    std::memcpy(
        buffer.data(),
        &nbytes,
        sizeof(std::uint64_t));

    std::memcpy(
        buffer.data() + sizeof(std::uint64_t),
        data,
        static_cast<std::size_t>(nbytes));

    return base64_encode(buffer.data(), buffer.size());
}

inline std::string encode_vec3_array_binary(
    const real_t *ux,
    const real_t *uy,
    const real_t *uz,
    const std::size_t npts)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(3) *
        static_cast<std::uint64_t>(npts) *
        static_cast<std::uint64_t>(sizeof(real_t));

    std::vector<unsigned char> buffer(
        sizeof(std::uint64_t) + static_cast<std::size_t>(nbytes));

    std::memcpy(
        buffer.data(),
        &nbytes,
        sizeof(std::uint64_t));

    real_t *payload =
        reinterpret_cast<real_t *>(buffer.data() + sizeof(std::uint64_t));

    for (std::size_t i = 0; i < npts; ++i)
    {
        payload[3 * i + 0] = ux[i];
        payload[3 * i + 1] = uy[i];
        payload[3 * i + 2] = uz[i];
    }

    return base64_encode(buffer.data(), buffer.size());
}

// =======================================================
// Mid-plane VTI writer
// =======================================================
//
// Writes the x-y plane at z = z_index.
//
// This version writes the full x-y slice:
//
// x = 0, ..., NX - 1
// y = 0, ..., NY - 1
//
// Therefore the output dimensions are NX x NY x 1.

inline void write_midplane_jet_vti(
    const std::filesystem::path &filename,
    const real_t *rho,
    const real_t *ux,
    const real_t *uy,
    const real_t *uz,
    const label_t z_index)
{
    if (!filename.parent_path().empty())
    {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename, std::ios::binary);

    if (!out)
    {
        throw std::runtime_error(
            "Cannot open mid-plane VTI file for writing: " + filename.string());
    }

    constexpr std::size_t npts =
        static_cast<std::size_t>(NX) *
        static_cast<std::size_t>(NY);

    const std::string enc_rho = encode_scalar_array_binary(rho, npts);

    const std::string enc_ux = encode_scalar_array_binary(ux, npts);
    const std::string enc_uy = encode_scalar_array_binary(uy, npts);
    const std::string enc_uz = encode_scalar_array_binary(uz, npts);

    const std::string enc_u = encode_vec3_array_binary(ux, uy, uz, npts);

    out << "<?xml version=\"1.0\"?>\n";

    out << "<VTKFile type=\"ImageData\" version=\"1.0\" "
        << "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    out << "  <ImageData WholeExtent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 0"
        << "\" Origin=\"0 0 " << z_index
        << "\" Spacing=\"1 1 1\">\n";

    out << "    <Piece Extent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 0\">\n";

    out << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rho\" format=\"binary\">\n";
    out << enc_rho << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"ux\" format=\"binary\">\n";
    out << enc_ux << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"uy\" format=\"binary\">\n";
    out << enc_uy << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"uz\" format=\"binary\">\n";
    out << enc_uz << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"binary\">\n";
    out << enc_u << "\n";
    out << "        </DataArray>\n";

    out << "      </PointData>\n";
    out << "      <CellData>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error(
            "Error while finalizing mid-plane VTI file: " + filename.string());
    }
}

// =======================================================
// Device-to-host mid-plane extraction
// =======================================================
//
// This function copies only the middle z-plane from GPU to CPU
// and writes one VTI file for the given step.
//
// Since idx = x + NX * (y + NY * z), the x-y plane at fixed z
// is contiguous in memory.

inline void write_midplane_jet_vti_step_device(
    const int step,
    const MomentsDevice &d,
    const std::filesystem::path &out_dir)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr label_t z_mid = NZ / static_cast<label_t>(2);

    constexpr std::size_t slice_cells =
        static_cast<std::size_t>(NX) *
        static_cast<std::size_t>(NY);

    std::vector<real_t> rho_slice(slice_cells);

    std::vector<real_t> ux_slice(slice_cells);
    std::vector<real_t> uy_slice(slice_cells);
    std::vector<real_t> uz_slice(slice_cells);

    const label_t base_id = idx(
        static_cast<label_t>(0),
        static_cast<label_t>(0),
        z_mid);

    CUDA_CHECK(cudaMemcpy(
        rho_slice.data(),
        d.rho + base_id,
        slice_cells * sizeof(real_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        ux_slice.data(),
        d.ux + base_id,
        slice_cells * sizeof(real_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        uy_slice.data(),
        d.uy + base_id,
        slice_cells * sizeof(real_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        uz_slice.data(),
        d.uz + base_id,
        slice_cells * sizeof(real_t),
        cudaMemcpyDeviceToHost));

    std::filesystem::create_directories(out_dir);

    std::ostringstream name;

    name << "jet_midplane_"
         << std::setw(8) << std::setfill('0') << step
         << ".vti";

    write_midplane_jet_vti(
        out_dir / name.str(),
        rho_slice.data(),
        ux_slice.data(),
        uy_slice.data(),
        uz_slice.data(),
        z_mid);
}

inline void write_midplane_jet_vti_step_device(
    const int step,
    const MomentsDevice &d)
{
    write_midplane_jet_vti_step_device(
        step,
        d,
        default_slice_out_dir());
}

// =======================================================
// Optional full-domain writer
// =======================================================

inline void write_full_vti(
    const std::filesystem::path &filename,
    const LbmHost &h)
{
    if (!filename.parent_path().empty())
    {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename, std::ios::binary);

    if (!out)
    {
        throw std::runtime_error(
            "Cannot open full-domain VTI file for writing: " + filename.string());
    }

    constexpr std::size_t npts = static_cast<std::size_t>(Ncells);

    const std::string enc_rho = encode_scalar_array_binary(h.rho, npts);
    const std::string enc_u = encode_vec3_array_binary(h.ux, h.uy, h.uz, npts);

    out << "<?xml version=\"1.0\"?>\n";

    out << "<VTKFile type=\"ImageData\" version=\"1.0\" "
        << "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    out << "  <ImageData WholeExtent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    out << "    <Piece Extent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1)
        << "\">\n";

    out << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rho\" format=\"binary\">\n";
    out << enc_rho << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"binary\">\n";
    out << enc_u << "\n";
    out << "        </DataArray>\n";

    out << "      </PointData>\n";
    out << "      <CellData>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error(
            "Error while finalizing full-domain VTI file: " + filename.string());
    }
}

inline void write_full_vti_step_device(
    const int step,
    const MomentsDevice &d,
    LbmHost &h,
    const std::filesystem::path &out_dir = default_out_dir())
{
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_out_D2H(h, d);

    std::filesystem::create_directories(out_dir);

    std::ostringstream name;

    name << "lbm_"
         << std::setw(8) << std::setfill('0') << step
         << ".vti";

    write_full_vti(out_dir / name.str(), h);
}

#endif
#ifndef VTIWRITER_CUH
#define VTIWRITER_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
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

// =======================================================
// Output directory
// =======================================================

inline std::filesystem::path default_out_dir()
{
    std::ostringstream folder_name;

    folder_name << "Re"
                << static_cast<int>(std::round(Re))
                << "_We"
                << static_cast<int>(std::round(We))
                << "_bubble_vtifiles";

    return std::filesystem::current_path() / "BUBBLE_VTK" / folder_name.str();
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
// Base64 stream encoder
// =======================================================

class Base64Stream
{
public:
    explicit Base64Stream(std::ostream &stream)
        : out(stream)
    {
        encoded.reserve(buffer_limit);
    }

    void write(const void *data, const std::size_t len)
    {
        const auto *bytes = static_cast<const unsigned char *>(data);

        std::size_t i = 0;

        if (tail_size > 0)
        {
            while (tail_size < 3 && i < len)
            {
                tail[tail_size++] = bytes[i++];
            }

            if (tail_size == 3)
            {
                emit(tail[0], tail[1], tail[2], 3);
                tail_size = 0;
            }
        }

        while (i + 3 <= len)
        {
            emit(bytes[i], bytes[i + 1], bytes[i + 2], 3);
            i += 3;
        }

        while (i < len)
        {
            tail[tail_size++] = bytes[i++];
        }
    }

    void finish()
    {
        if (tail_size == 1)
        {
            emit(tail[0], 0, 0, 1);
        }
        else if (tail_size == 2)
        {
            emit(tail[0], tail[1], 0, 2);
        }

        tail_size = 0;
        flush();
    }

private:
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    static constexpr std::size_t buffer_limit = 1U << 20;

    std::ostream &out;
    std::string encoded;

    unsigned char tail[3]{};
    std::size_t tail_size = 0;

    void emit(
        const unsigned char b0,
        const unsigned char b1,
        const unsigned char b2,
        const std::size_t valid)
    {
        const std::uint32_t triple =
            (static_cast<std::uint32_t>(b0) << 16) |
            (static_cast<std::uint32_t>(b1) << 8) |
            static_cast<std::uint32_t>(b2);

        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(valid > 1 ? table[(triple >> 6) & 0x3F] : '=');
        encoded.push_back(valid > 2 ? table[triple & 0x3F] : '=');

        if (encoded.size() >= buffer_limit)
        {
            flush();
        }
    }

    void flush()
    {
        if (!encoded.empty())
        {
            out.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
            encoded.clear();
        }
    }
};

// =======================================================
// Binary array writers
// =======================================================
//
// VTK XML binary format expects:
//
// [UInt64 byte_count][raw bytes]
//
// and then the whole block is Base64 encoded.

inline void write_binary_scalar_array(
    std::ostream &out,
    const real_t *data,
    const std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) *
        static_cast<std::uint64_t>(sizeof(real_t));

    Base64Stream encoder(out);

    encoder.write(&nbytes, sizeof(nbytes));
    encoder.write(data, static_cast<std::size_t>(nbytes));

    encoder.finish();

    out << '\n';
}

inline void write_binary_sum_array(
    std::ostream &out,
    const real_t *a,
    const real_t *b,
    const std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) *
        static_cast<std::uint64_t>(sizeof(real_t));

    Base64Stream encoder(out);

    encoder.write(&nbytes, sizeof(nbytes));

    constexpr std::size_t chunk_vals = 1U << 16;

    std::vector<real_t> buffer(chunk_vals);

    for (std::size_t offset = 0; offset < nvals;)
    {
        std::size_t count = nvals - offset;

        if (count > chunk_vals)
        {
            count = chunk_vals;
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t id = offset + i;
            buffer[i] = a[id] + b[id];
        }

        encoder.write(buffer.data(), count * sizeof(real_t));

        offset += count;
    }

    encoder.finish();

    out << '\n';
}

inline void write_binary_vec3_array(
    std::ostream &out,
    const real_t *ux,
    const real_t *uy,
    const real_t *uz,
    const std::size_t npts)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(3) *
        static_cast<std::uint64_t>(npts) *
        static_cast<std::uint64_t>(sizeof(real_t));

    Base64Stream encoder(out);

    encoder.write(&nbytes, sizeof(nbytes));

    constexpr std::size_t chunk_points = 1U << 16;

    std::vector<real_t> buffer(3 * chunk_points);

    for (std::size_t offset = 0; offset < npts;)
    {
        std::size_t count = npts - offset;

        if (count > chunk_points)
        {
            count = chunk_points;
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t src = offset + i;

            buffer[3 * i + 0] = ux[src];
            buffer[3 * i + 1] = uy[src];
            buffer[3 * i + 2] = uz[src];
        }

        encoder.write(buffer.data(), 3 * count * sizeof(real_t));

        offset += count;
    }

    encoder.finish();

    out << '\n';
}

// =======================================================
// Full-domain multicomponent VTI writer
// =======================================================
//
// Written fields:
//
// rhor
// rhob
// rho = rhor + rhob
// ux
// uy
// uz
// velocity = (ux, uy, uz)

inline void write_vti(
    const std::filesystem::path &filename,
    const LbmHost &h,
    const bool write_velocity_components = true)
{
    if (!filename.parent_path().empty())
    {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename, std::ios::binary);

    if (!out)
    {
        throw std::runtime_error(
            "Cannot open VTI file for writing: " + filename.string());
    }

    constexpr std::size_t npts = static_cast<std::size_t>(Ncells);

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
        << "\" Name=\"rhor\" format=\"binary\">\n";
    write_binary_scalar_array(out, h.rhor, npts);
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rhob\" format=\"binary\">\n";
    write_binary_scalar_array(out, h.rhob, npts);
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rho\" format=\"binary\">\n";
    write_binary_sum_array(out, h.rhor, h.rhob, npts);
    out << "        </DataArray>\n";

    if (write_velocity_components)
    {
        out << "        <DataArray type=\"" << vtk_real_type()
            << "\" Name=\"ux\" format=\"binary\">\n";
        write_binary_scalar_array(out, h.ux, npts);
        out << "        </DataArray>\n";

        out << "        <DataArray type=\"" << vtk_real_type()
            << "\" Name=\"uy\" format=\"binary\">\n";
        write_binary_scalar_array(out, h.uy, npts);
        out << "        </DataArray>\n";

        out << "        <DataArray type=\"" << vtk_real_type()
            << "\" Name=\"uz\" format=\"binary\">\n";
        write_binary_scalar_array(out, h.uz, npts);
        out << "        </DataArray>\n";
    }

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"binary\">\n";
    write_binary_vec3_array(out, h.ux, h.uy, h.uz, npts);
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
            "Error while finalizing VTI file: " + filename.string());
    }
}

// =======================================================
// Step filename
// =======================================================

inline std::filesystem::path vti_filename(
    const int step,
    std::filesystem::path folder = default_out_dir())
{
    std::ostringstream name;

    name << "bubble_"
         << std::setw(8) << std::setfill('0') << step
         << ".vti";

    return folder / name.str();
}

// =======================================================
// Device-to-host copy and VTI output
// =======================================================

inline void write_vti_step_device(
    const int step,
    const MomentsDevice &d,
    LbmHost &h,
    std::filesystem::path folder = default_out_dir(),
    const bool write_velocity_components = true)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_out_D2H(h, d);

    write_vti(
        vti_filename(step, std::move(folder)),
        h,
        write_velocity_components);
}

#endif
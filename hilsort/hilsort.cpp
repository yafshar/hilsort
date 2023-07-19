#include "hilbert.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <utility>

template <typename T>
constexpr unsigned int kAssumedBits{8 * sizeof(T)};

namespace py = pybind11;

static bitmask_t bitTranspose_3_8(bitmask_t inCoords)
{
    bitmask_t inFieldEnds{1};
    bitmask_t inMask{255};

    unsigned int inB{8};
    unsigned int utB;

    while ((utB = inB / 2))
    {
        bitmask_t const utFieldEnds = inFieldEnds | (inFieldEnds << (inB + utB));
        bitmask_t const utMask = (utFieldEnds << utB) - utFieldEnds;
        inFieldEnds = utFieldEnds;

        bitmask_t in = inCoords & inMask;
        inCoords >>= inB;
        in = (in | (in << inB)) & utMask;

        bitmask_t utCoords = in;

        in = inCoords & inMask;
        inCoords >>= inB;
        in = (in | (in << inB)) & utMask;
        utCoords |= in << utB;

        in = inCoords & inMask;
        inCoords >>= inB;
        in = (in | (in << inB)) & utMask;
        utCoords |= in << inB;

        inCoords = utCoords;
        inB = utB;
        inMask = utMask;
    }

    return inCoords;
}

bitmask_t hilbert_c2i_3_8(bitmask_t *coord)
{
    bitmask_t coords = coord[2] << 8;
    coords |= coord[1];
    coords <<= 8;
    coords |= coord[0];
    coords = bitTranspose_3_8(coords);
    coords ^= coords >> 3;

    halfmask_t const ndOnes{7};
    halfmask_t const nd1Ones{3};
    halfmask_t flipBit{0};
    bitmask_t index{0};
    unsigned int b{24};
    unsigned int rotation{0};

    do
    {
        halfmask_t bits = (coords >> (b -= 3)) & ndOnes;
        halfmask_t const fbits = flipBit ^ bits;
        bits = ((fbits >> rotation) | (fbits << (3 - rotation))) & ndOnes;
        index <<= 3;
        index |= bits;
        flipBit = static_cast<halfmask_t>(1) << rotation;
        bits &= -bits & nd1Ones;
        while (bits)
        {
            bits >>= 1;
            ++rotation;
        }
        if (++rotation >= 3)
        {
            rotation -= 3;
        }

    } while (b);

    index ^= static_cast<bitmask_t>(1198372);
    index ^= index >> 1;
    index ^= index >> 2;
    index ^= index >> 4;
    index ^= index >> 8;
    index ^= index >> 16;
    return index;
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_sort_bind(unsigned int ndims, unsigned int nbits, py::array_t<T, py::array::c_style> data_np, py::array_t<bitmask_t, py::array::c_style> coords_np)
{
    auto buf1 = data_np.request();
    T *data = static_cast<T *>(buf1.ptr);

    auto buf2 = coords_np.request();
    bitmask_t *coords = static_cast<bitmask_t *>(buf2.ptr);

    std::size_t npoints = buf1.size / ndims;
    std::vector<std::pair<bitmask_t, std::size_t>> bins(npoints);

    for (std::size_t i = 0; i < npoints; ++i)
    {
        bitmask_t bin = hilbert_c2i(ndims, nbits, &coords[i * ndims]);
        bins[i] = std::pair<bitmask_t, std::size_t>(bin, i);
    }

    std::sort(bins.begin(), bins.end());

    auto result_array = py::array(
        py::buffer_info(
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            2,
            {buf1.shape[0], buf1.shape[1]},
            {sizeof(T) * buf1.shape[1], sizeof(T)}));

    auto buf3 = result_array.request();
    T *result = static_cast<T *>(buf3.ptr);

    for (std::size_t i = 0; i < npoints; ++i)
    {
        std::size_t old_ind = bins[i].second * ndims;
        std::size_t new_ind = i * ndims;
        for (std::size_t j = 0; j < ndims; ++j)
        {
            result[new_ind + j] = data[old_ind + j];
        }
    }

    return result_array;
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_sort_bind_3_8(py::array_t<T, py::array::c_style> data_np, py::array_t<bitmask_t, py::array::c_style> coords_np)
{
    auto buf1 = data_np.request();
    T *data = static_cast<T *>(buf1.ptr);

    auto buf2 = coords_np.request();
    bitmask_t *coords = static_cast<bitmask_t *>(buf2.ptr);

    std::size_t npoints = buf1.size / 3;
    std::vector<std::pair<bitmask_t, std::size_t>> bins(npoints);

    for (std::size_t i = 0; i < npoints; ++i)
    {
        bitmask_t bin = hilbert_c2i_3_8(&coords[i * 3]);
        bins[i] = std::pair<bitmask_t, std::size_t>(bin, i);
    }

    std::sort(bins.begin(), bins.end());

    auto result_array = py::array(
        py::buffer_info(
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            2,
            {buf1.shape[0], buf1.shape[1]},
            {sizeof(T) * buf1.shape[1], sizeof(T)}));

    auto buf3 = result_array.request();
    T *result = static_cast<T *>(buf3.ptr);

    for (std::size_t i = 0; i < npoints; ++i)
    {
        std::size_t const old_ind = bins[i].second * 3;
        std::size_t const new_ind = i * 3;
        result[new_ind] = data[old_ind];
        result[new_ind + 1] = data[old_ind + 1];
        result[new_ind + 2] = data[old_ind + 2];
    }

    return result_array;
}

template <typename T>
int hilbert_cmp_bind(unsigned int nbits, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    void *coord1 = static_cast<void *>(buf1.ptr);
    void *coord2 = static_cast<void *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    return hilbert_cmp(ndims, nbytes, nbits, coord1, coord2);
}

template <typename T>
unsigned int hilbert_box_vtx_bind(unsigned int nbits, int find_min, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    void *coord1 = static_cast<void *>(buf1.ptr);
    void *coord2 = static_cast<void *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    return hilbert_box_vtx(ndims, nbytes, nbits, find_min, coord1, coord2);
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_min_box_vtx_bind(unsigned int nbits, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    T *coord1 = static_cast<T *>(buf1.ptr);
    T *coord2 = static_cast<T *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    auto cornerlo_array = py::array(py::buffer_info(nullptr, sizeof(T), py::format_descriptor<T>::format(), 1, {ndims}, {sizeof(T)}));
    auto cornerlo_buf = cornerlo_array.request();
    T *cornerlo = static_cast<T *>(cornerlo_buf.ptr);
    for (unsigned int i = 0; i < ndims; ++i)
        cornerlo[i] = coord1[i];

    std::vector<T> work(coord2, coord2 + ndims);

    hilbert_box_vtx(ndims, nbytes, nbits, 1, (void *)cornerlo, (void *)work.data());

    return cornerlo_array;
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_max_box_vtx_bind(unsigned int nbits, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    T *coord1 = static_cast<T *>(buf1.ptr);
    T *coord2 = static_cast<T *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    auto cornerhi_array = py::array(py::buffer_info(nullptr, sizeof(T), py::format_descriptor<T>::format(), 1, {ndims}, {sizeof(T)}));
    auto cornerhi_buf = cornerhi_array.request();
    T *cornerhi = static_cast<T *>(cornerhi_buf.ptr);
    for (unsigned int i = 0; i < ndims; ++i)
        cornerhi[i] = coord2[i];

    std::vector<T> work(coord1, coord1 + ndims);

    hilbert_box_vtx(ndims, nbytes, nbits, 0, (void *)work.data(), (void *)cornerhi);
    return cornerhi_array;
}

template <typename T>
unsigned int hilbert_box_pt_bind(unsigned int nbits, int find_min, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    void *coord1 = static_cast<void *>(buf1.ptr);
    void *coord2 = static_cast<void *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    return hilbert_box_pt(ndims, nbytes, nbits, find_min, coord1, coord2);
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_min_box_pt_bind(unsigned int nbits, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    T *coord1 = static_cast<T *>(buf1.ptr);
    T *coord2 = static_cast<T *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    auto cornerlo_array = py::array(py::buffer_info(nullptr, sizeof(T), py::format_descriptor<T>::format(), 1, {ndims}, {sizeof(T)}));
    auto cornerlo_buf = cornerlo_array.request();
    T *cornerlo = static_cast<T *>(cornerlo_buf.ptr);
    for (unsigned int i = 0; i < ndims; ++i)
        cornerlo[i] = coord1[i];

    std::vector<T> work(coord2, coord2 + ndims);

    hilbert_box_pt(ndims, nbytes, nbits, 1, (void *)cornerlo, (void *)work.data());

    return cornerlo_array;
}

template <typename T>
py::array_t<T, py::array::c_style> hilbert_max_box_pt_bind(unsigned int nbits, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    if (buf1.size != buf2.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    T *coord1 = static_cast<T *>(buf1.ptr);
    T *coord2 = static_cast<T *>(buf2.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    auto cornerhi_array = py::array(py::buffer_info(nullptr, sizeof(T), py::format_descriptor<T>::format(), 1, {ndims}, {sizeof(T)}));
    auto cornerhi_buf = cornerhi_array.request();
    T *cornerhi = static_cast<T *>(cornerhi_buf.ptr);
    for (unsigned int i = 0; i < ndims; ++i)
        cornerhi[i] = coord2[i];

    std::vector<T> work(coord1, coord1 + ndims);

    hilbert_box_pt(ndims, nbytes, nbits, 0, (void *)work.data(), (void *)cornerhi);
    return cornerhi_array;
}

template <typename T>
int hilbert_nextinbox_bind(unsigned int nbits, int find_prev, py::array_t<T, py::array::c_style> coord1_np, py::array_t<T, py::array::c_style> coord2_np, py::array_t<T, py::array::c_style> point_np)
{
    if (nbits > kAssumedBits<bitmask_t>)
    {
        throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
    }

    auto buf1 = coord1_np.request();
    auto buf2 = coord2_np.request();
    auto buf3 = point_np.request();
    if (buf1.size != buf2.size || buf1.size != buf3.size)
    {
        throw std::runtime_error("Input array sizes do not match!");
    }

    void *coord1 = static_cast<void *>(buf1.ptr);
    void *coord2 = static_cast<void *>(buf2.ptr);
    void *point = static_cast<void *>(buf3.ptr);

    auto ndims = static_cast<unsigned int>(buf1.size);
    auto nbytes = static_cast<unsigned int>(sizeof(T));

    return hilbert_nextinbox(ndims, nbytes, nbits, find_prev, coord1, coord2, point);
}

PYBIND11_MODULE(_hilsort, m)
{
    m.doc() = R"pbdoc(
        Python binding to Hilbert space-filling curve coordinates, without recursion, 
        from integer index, and vice versa, and other Hilbert-related calculations.  
        Plus extra functionality for sorting points in Euclidean space using space-filling curves.
       )pbdoc";

    m.def(
        "hilbert_i2c", [](unsigned int ndims, unsigned int nbits, bitmask_t index) -> py::array_t<bitmask_t>
        {
            if (ndims * nbits > kAssumedBits<bitmask_t>)
            {
                throw std::runtime_error("Assumptions: ndims * nbits <= (sizeof index) * 8.");
            }
            auto result_array = py::array(py::buffer_info(nullptr, sizeof(bitmask_t), py::format_descriptor<bitmask_t>::format(), 1, {ndims}, {sizeof(bitmask_t)}));
            auto buf = result_array.request();
            bitmask_t *result = static_cast<bitmask_t *>(buf.ptr);
            hilbert_i2c(ndims, nbits, index, result); 
            return result_array; },
        R"pbdoc(
        Convert an index into a Hilbert curve to a set of coordinates.
        
        Inputs:
            ndims: Number of coordinate axes.
            nbits: Number of bits/coordinate.
            index: The index, contains ndims*nbits bits (so ndims*nbits must be <= 8*sizeof(bitmask_t)).
        
        Returns:
            coord: The list of ndims coordinates, each with nbits bits.
        
        Assumptions:
            ndims*nbits <= (sizeof index) * (bits_per_byte)
       )pbdoc",
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("index"));

    m.def(
        "hilbert_c2i", [](unsigned int nbits, py::array_t<bitmask_t, py::array::c_style> array) -> bitmask_t
        {
            auto buf = array.request();
            unsigned int ndims = static_cast<unsigned int>(buf.size);
            if (ndims * nbits > kAssumedBits<bitmask_t>)
            {
                throw std::runtime_error("Assumptions: ndims * nbits <= (sizeof index) * 8.");
            }
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);
            bitmask_t index = hilbert_c2i(ndims, nbits, coord);
            return index; },
        R"pbdoc(
        Convert coordinates of a point on a Hilbert curve to its index.
        
        Inputs:
            nbits: Number of bits/coordinate.
            coord: Array of n nbits-bit coordinates.
            
        Returns:
            index: Output index value.  ndims*nbits bits.
        
        Assumptions:
            ndims*nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord"));

    m.def(
        "hilbert_c2i", [](unsigned int nbits, py::array_t<std::int64_t, py::array::c_style> array) -> bitmask_t
        {
            auto buf = array.request();
            unsigned int ndims = static_cast<unsigned int>(buf.size);
            if (ndims * nbits > kAssumedBits<bitmask_t>)
            {
                throw std::runtime_error("Assumptions: ndims * nbits <= (sizeof index) * 8.");
            }
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);
            bitmask_t index = hilbert_c2i(ndims, nbits, coord);
            return index; },
        R"pbdoc(
        Convert coordinates of a point on a Hilbert curve to its index.

        Inputs:
            nbits: Number of bits/coordinate.
            coord: Array of n nbits-bit coordinates.

        Returns:
            index: Output index value.  ndims*nbits bits.

        Assumptions:
            ndims*nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord"));

    m.def(
        "hilbert_c2i_3_8", [](py::array_t<bitmask_t, py::array::c_style> array) -> bitmask_t
        {
            auto buf = array.request();
            if (buf.size != 3)
            {
                throw std::runtime_error("Must have 3 coordinates!");
            }
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);
            bitmask_t index = hilbert_c2i_3_8(coord);
            return index; },
        R"pbdoc(
        Convert coordinates of a 3D point on a Hilbert curve to its index.

        Inputs:
            coord: Array of n 8-bit coordinates.

        Returns:
            index: Output index value.  3*nbits bits.
       )pbdoc",
        py::arg("coord"));

    m.def(
        "hilbert_c2i_3_8", [](py::array_t<std::int64_t, py::array::c_style> array) -> bitmask_t
        {
            auto buf = array.request();
            if (buf.size != 3)
            {
                throw std::runtime_error("Must have 3 coordinates!");
            }
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);
            bitmask_t index = hilbert_c2i_3_8(coord);
            return index; },
        R"pbdoc(
        Convert coordinates of a 3D point on a Hilbert curve to its index.

        Inputs:
            coord: Array of n 8-bit coordinates.

        Returns:
            index: Output index value.  3*nbits bits.
       )pbdoc",
        py::arg("coord"));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<std::uint32_t>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<bitmask_t>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<std::int32_t>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<std::int64_t>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<float>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind", &hilbert_sort_bind<double>,
        py::arg("ndims"),
        py::arg("nbits"),
        py::arg("data"),
        py::arg("coords"));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<std::uint32_t>,
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<bitmask_t>,
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<std::int32_t>,
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<std::int64_t>,
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<float>,
        py::arg("data"),
        py::arg("coords").noconvert(true));

    m.def(
        "hilbert_sort_bind_3_8", &hilbert_sort_bind_3_8<double>,
        py::arg("data"),
        py::arg("coords"));

    m.def(
        "hilbert_cmp", &hilbert_cmp_bind<std::uint32_t>,
        R"pbdoc(
        Determine which of two points lies further along the Hilbert curve.

        Inputs:
            nbits:  Number of bits/coordinate.
            coord1: Array of ndims nbytes-byte coordinates
            coord2: Array of ndims nbytes-byte coordinates

        Returns:
            -1, 0, or 1 according to whether coord1<coord2, coord1==coord2, coord1>coord2

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_cmp", &hilbert_cmp_bind<bitmask_t>,
        R"pbdoc(
        Determine which of two points lies further along the Hilbert curve.

        Inputs:
            nbits:  Number of bits/coordinate.
            coord1: Array of ndims nbytes-byte coordinates
            coord2: Array of ndims nbytes-byte coordinates

        Returns:
            -1, 0, or 1 according to whether coord1<coord2, coord1==coord2, coord1>coord2

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_cmp", &hilbert_cmp_bind<std::int32_t>,
        R"pbdoc(
        Determine which of two points lies further along the Hilbert curve.

        Inputs:
            nbits:  Number of bits/coordinate.
            coord1: Array of ndims nbytes-byte coordinates
            coord2: Array of ndims nbytes-byte coordinates

        Returns:
            -1, 0, or 1 according to whether coord1<coord2, coord1==coord2, coord1>coord2

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_cmp", &hilbert_cmp_bind<std::int64_t>,
        R"pbdoc(
        Determine which of two points lies further along the Hilbert curve.

        Inputs:
            nbits:  Number of bits/coordinate.
            coord1: Array of ndims nbytes-byte coordinates
            coord2: Array of ndims nbytes-byte coordinates

        Returns:
            -1, 0, or 1 according to whether coord1<coord2, coord1==coord2, coord1>coord2

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_cmp", [](py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> int
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("Input array sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);

            return hilbert_ieee_cmp(ndims, coord1, coord2); },
        R"pbdoc(
        Determine which of two points lies further along the Hilbert curve.

        Inputs:
            coord1: Array of ndims nbytes-byte coordinates
            coord2: Array of ndims nbytes-byte coordinates

        Returns:
            -1, 0, or 1 according to whether coord1<coord2, coord1==coord2, coord1>coord2
       )pbdoc",
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_box_vtx", &hilbert_box_vtx_bind<std::uint32_t>,
        R"pbdoc(
        Determine the first or last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least vertex sought?
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_vtx", &hilbert_box_vtx_bind<bitmask_t>,
        R"pbdoc(
        Determine the first or last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least vertex sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_vtx", &hilbert_box_vtx_bind<std::int32_t>,
        R"pbdoc(
        Determine the first or last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least vertex sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_vtx", &hilbert_box_vtx_bind<std::int64_t>,
        R"pbdoc(
        Determine the first or last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least vertex sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_box_vtx", [](int findmin, py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> unsigned int
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);

            return hilbert_ieee_box_vtx(ndims, findmin, coord1, coord2); },
        R"pbdoc(
        Determine the first or last vertex of a box to lie on a Hilbert curve.

        Inputs:
            findmin: Is it the least vertex sought?
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners
       )pbdoc",
        py::arg("findmin"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_min_box_vtx", &hilbert_min_box_vtx_bind<std::uint32_t>,
        R"pbdoc(
        Determine the first vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_vtx", &hilbert_min_box_vtx_bind<bitmask_t>,
        R"pbdoc(
        Determine the first vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_vtx", &hilbert_min_box_vtx_bind<std::int32_t>,
        R"pbdoc(
        Determine the first vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_vtx", &hilbert_min_box_vtx_bind<std::int64_t>,
        R"pbdoc(
        Determine the first vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_max_box_vtx", &hilbert_max_box_vtx_bind<std::uint32_t>,
        R"pbdoc(
        Determine the last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_vtx", &hilbert_max_box_vtx_bind<bitmask_t>,
        R"pbdoc(
        Determine the last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_vtx", &hilbert_max_box_vtx_bind<std::int32_t>,
        R"pbdoc(
        Determine the last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_vtx", &hilbert_max_box_vtx_bind<std::int64_t>,
        R"pbdoc(
        Determine the last vertex of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last vertex of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_min_box_vtx", [](py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> py::array_t<double, py::array::c_style>
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("Input sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);
            auto nbytes = static_cast<unsigned int>(sizeof(double));

            auto cornerlo_array = py::array(py::buffer_info(nullptr, sizeof(double), py::format_descriptor<double>::format(), 1, {ndims}, {sizeof(double)}));
            auto cornerlo_buf = cornerlo_array.request();
            double *cornerlo = static_cast<double *>(cornerlo_buf.ptr);
            for (unsigned int i = 0; i < ndims; ++i)
                cornerlo[i] = coord1[i];

            std::vector<double> work(coord2, coord2 + ndims);

            hilbert_ieee_box_vtx(ndims, 1, cornerlo, work.data()); 
            
            return cornerlo_array; },
        R"pbdoc(
        Determine the first vertex of a box to lie on a Hilbert curve.

        Inputs:
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            The first vertex of a box to lie on a Hilbert curve.

       )pbdoc",
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_max_box_vtx", [](py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> py::array_t<double, py::array::c_style>
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("Input sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);
            auto nbytes = static_cast<unsigned int>(sizeof(double));

            auto cornerhi_array = py::array(py::buffer_info(nullptr, sizeof(double), py::format_descriptor<double>::format(), 1, {ndims}, {sizeof(double)}));
            auto cornerhi_buf = cornerhi_array.request();
            double *cornerhi = static_cast<double *>(cornerhi_buf.ptr);
            for (unsigned int i = 0; i < ndims; ++i)
                cornerhi[i] = coord2[i];

            std::vector<double> work(coord1, coord1 + ndims);

            hilbert_ieee_box_vtx(ndims, 0, work.data(), cornerhi); 
            
            return cornerhi_array; },
        R"pbdoc(
        Determine the last vertex of a box to lie on a Hilbert curve.

        Inputs:
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            The last vertex of a box to lie on a Hilbert curve.

       )pbdoc",
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_box_pt", &hilbert_box_pt_bind<std::uint32_t>,
        R"pbdoc(
        Determine the first or last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least point sought?
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_pt", &hilbert_box_pt_bind<bitmask_t>,
        R"pbdoc(
        Determine the first or last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least point sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_pt", &hilbert_box_pt_bind<std::int32_t>,
        R"pbdoc(
        Determine the first or last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least point sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_box_pt", &hilbert_box_pt_bind<std::int64_t>,
        R"pbdoc(
        Determine the first or last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            findmin: Is it the least point sought?
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("findmin"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_box_pt", [](int findmin, py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> unsigned int
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);

            return hilbert_ieee_box_pt(ndims, findmin, coord1, coord2); },
        R"pbdoc(
        Determine the first or last point of a box to lie on a Hilbert curve.

        Inputs:
            findmin: Is it the least point sought?
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            coord1 and coord2 modified to refer to selected corner
            value returned is log2 of size of largest power-of-two-aligned box that
            contains the selected corner and no other corners
       )pbdoc",
        py::arg("findmin"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_min_box_pt", &hilbert_min_box_pt_bind<std::uint32_t>,
        R"pbdoc(
        Determine the first point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_pt", &hilbert_min_box_pt_bind<bitmask_t>,
        R"pbdoc(
        Determine the first point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_pt", &hilbert_min_box_pt_bind<std::int32_t>,
        R"pbdoc(
        Determine the first point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_min_box_pt", &hilbert_min_box_pt_bind<std::int64_t>,
        R"pbdoc(
        Determine the first point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            first point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_max_box_pt", &hilbert_max_box_pt_bind<std::uint32_t>,
        R"pbdoc(
        Determine the last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_pt", &hilbert_max_box_pt_bind<bitmask_t>,
        R"pbdoc(
        Determine the last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_pt", &hilbert_max_box_pt_bind<std::int32_t>,
        R"pbdoc(
        Determine the last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true));

    m.def(
        "hilbert_max_box_pt", &hilbert_max_box_pt_bind<std::int64_t>,
        R"pbdoc(
        Determine the last point of a box to lie on a Hilbert curve.

        Inputs:
            nbits:   Number of bits/coordinate.
            coord1:  Array of ndims nbytes-byte coordinates - one corner of box
            coord2:  Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            last point of a box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_min_box_pt", [](py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> py::array_t<double, py::array::c_style>
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("Input sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);
            auto nbytes = static_cast<unsigned int>(sizeof(double));

            auto cornerlo_array = py::array(py::buffer_info(nullptr, sizeof(double), py::format_descriptor<double>::format(), 1, {ndims}, {sizeof(double)}));
            auto cornerlo_buf = cornerlo_array.request();
            double *cornerlo = static_cast<double *>(cornerlo_buf.ptr);
            for (unsigned int i = 0; i < ndims; ++i)
                cornerlo[i] = coord1[i];

            std::vector<double> work(coord2, coord2 + ndims);

            hilbert_ieee_box_pt(ndims, 1, cornerlo, work.data()); 
            
            return cornerlo_array; },
        R"pbdoc(
        Determine the first point of a box to lie on a Hilbert curve.

        Inputs:
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            The first point of a box to lie on a Hilbert curve.

       )pbdoc",
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_ieee_max_box_pt", [](py::array_t<double, py::array::c_style> coord1_np, py::array_t<double, py::array::c_style> coord2_np) -> py::array_t<double, py::array::c_style>
        {
            auto buf1 = coord1_np.request();
            auto buf2 = coord2_np.request();
            if (buf1.size != buf2.size)
            {
                throw std::runtime_error("Input sizes do not match!");
            }
            
            double *coord1 = static_cast<double *>(buf1.ptr);
            double *coord2 = static_cast<double *>(buf2.ptr);

            auto ndims = static_cast<unsigned int>(buf1.size);
            auto nbytes = static_cast<unsigned int>(sizeof(double));

            auto cornerhi_array = py::array(py::buffer_info(nullptr, sizeof(double), py::format_descriptor<double>::format(), 1, {ndims}, {sizeof(double)}));
            auto cornerhi_buf = cornerhi_array.request();
            double *cornerhi = static_cast<double *>(cornerhi_buf.ptr);
            for (unsigned int i = 0; i < ndims; ++i)
                cornerhi[i] = coord2[i];

            std::vector<double> work(coord1, coord1 + ndims);

            hilbert_ieee_box_pt(ndims, 0, work.data(), cornerhi); 
            
            return cornerhi_array; },
        R"pbdoc(
        Determine the last point of a box to lie on a Hilbert curve.

        Inputs:
            coord1: Array of ndims nbytes-byte coordinates - one corner of box
            coord2: Array of ndims nbytes-byte coordinates - opposite corner

        Returns:
            The last point of a box to lie on a Hilbert curve.

       )pbdoc",
        py::arg("coord1"),
        py::arg("coord2"));

    m.def(
        "hilbert_nextinbox", &hilbert_nextinbox_bind<std::uint32_t>,
        R"pbdoc(
        Determine the first point of a box after a given point to lie on a Hilbert curve.

        Inputs:
            nbits:     Number of bits/coordinate.
            find_prev: Is the previous point sought?
            coord1:    Array of ndims nbytes-byte coordinates - one corner of box
            coord2:    Array of ndims nbytes-byte coordinates - opposite corner
            point:     Array of ndims nbytes-byte coordinates - lower bound on point returned

        Output:
            if returns 1:
                coord1 and coord2 modified to refer to least point after "point" in box
            else returns 0:
                arguments unchanged; "point" is beyond the last point of the box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("find_prev"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true),
        py::arg("point").noconvert(true));

    m.def(
        "hilbert_nextinbox", &hilbert_nextinbox_bind<bitmask_t>,
        R"pbdoc(
        Determine the first point of a box after a given point to lie on a Hilbert curve.

        Inputs:
            nbits:     Number of bits/coordinate.
            find_prev: Is the previous point sought?
            coord1:    Array of ndims nbytes-byte coordinates - one corner of box
            coord2:    Array of ndims nbytes-byte coordinates - opposite corner
            point:     Array of ndims nbytes-byte coordinates - lower bound on point returned

        Output:
            if returns 1:
                coord1 and coord2 modified to refer to least point after "point" in box
            else returns 0:
                arguments unchanged; "point" is beyond the last point of the box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("find_prev"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true),
        py::arg("point").noconvert(true));

    m.def(
        "hilbert_nextinbox", &hilbert_nextinbox_bind<std::int32_t>,
        R"pbdoc(
        Determine the first point of a box after a given point to lie on a Hilbert curve.

        Inputs:
            nbits:     Number of bits/coordinate.
            find_prev: Is the previous point sought?
            coord1:    Array of ndims nbytes-byte coordinates - one corner of box
            coord2:    Array of ndims nbytes-byte coordinates - opposite corner
            point:     Array of ndims nbytes-byte coordinates - lower bound on point returned

        Output:
            if returns 1:
                coord1 and coord2 modified to refer to least point after "point" in box
            else returns 0:
                arguments unchanged; "point" is beyond the last point of the box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("find_prev"),
        py::arg("coord1").noconvert(true),
        py::arg("coord2").noconvert(true),
        py::arg("point").noconvert(true));

    m.def(
        "hilbert_nextinbox", &hilbert_nextinbox_bind<std::int64_t>,
        R"pbdoc(
        Determine the first point of a box after a given point to lie on a Hilbert curve.

        Inputs:
            nbits:     Number of bits/coordinate.
            find_prev: Is the previous point sought?
            coord1:    Array of ndims nbytes-byte coordinates - one corner of box
            coord2:    Array of ndims nbytes-byte coordinates - opposite corner
            point:     Array of ndims nbytes-byte coordinates - lower bound on point returned

        Output:
            if returns 1:
                coord1 and coord2 modified to refer to least point after "point" in box
            else returns 0:
                arguments unchanged; "point" is beyond the last point of the box

        Assumptions:
            nbits <= (sizeof bitmask_t) * (bits_per_byte)
       )pbdoc",
        py::arg("nbits"),
        py::arg("find_prev"),
        py::arg("coord1"),
        py::arg("coord2"),
        py::arg("point"));

    m.def(
        "hilbert_incr", [](unsigned int nbits, py::array_t<bitmask_t, py::array::c_style> coord_np) -> py::array_t<bitmask_t, py::array::c_style>
        {
            if (nbits > kAssumedBits<bitmask_t>)
            {
                throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
            }

            auto buf = coord_np.request();
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);

            auto ndims = static_cast<unsigned int>(buf.size);

            auto result_array = py::array(py::buffer_info(nullptr, sizeof(bitmask_t), py::format_descriptor<bitmask_t>::format(), 1, {ndims}, {sizeof(bitmask_t)}));
            auto result_buf = result_array.request();
            bitmask_t *result = static_cast<bitmask_t *>(result_buf.ptr);

            for (unsigned int i = 0; i < ndims; ++i)
                result[i] = coord[i];

            hilbert_incr(ndims, nbits, result);

            return result_array; },
        R"pbdoc(
        Advance from one point to its successor on a Hilbert curve.

        Inputs:
            nbits: Number of bits/coordinate.
            coord: Array of ndims nbits-bit coordinates

        Returns:
            Next point on Hilbert curve.

       )pbdoc",
        py::arg("nbits"),
        py::arg("coord"));

    m.def(
        "hilbert_incr", [](unsigned int nbits, py::array_t<std::int64_t, py::array::c_style> coord_np) -> py::array_t<bitmask_t, py::array::c_style>
        {
            if (nbits > kAssumedBits<bitmask_t>)
            {
                throw std::runtime_error("Assumptions: nbits <= (sizeof bitmask_t) * 8.");
            }

            auto buf = coord_np.request();
            bitmask_t *coord = static_cast<bitmask_t *>(buf.ptr);

            auto ndims = static_cast<unsigned int>(buf.size);

            auto result_array = py::array(py::buffer_info(nullptr, sizeof(bitmask_t), py::format_descriptor<bitmask_t>::format(), 1, {ndims}, {sizeof(bitmask_t)}));
            auto result_buf = result_array.request();
            bitmask_t *result = static_cast<bitmask_t *>(result_buf.ptr);

            for (unsigned int i = 0; i < ndims; ++i)
                result[i] = coord[i];

            hilbert_incr(ndims, nbits, result);

            return result_array; },
        R"pbdoc(
        Advance from one point to its successor on a Hilbert curve.

        Inputs:
            nbits: Number of bits/coordinate.
            coord: Array of ndims nbits-bit coordinates

        Returns:
            Next point on Hilbert curve.

       )pbdoc",
        py::arg("nbits"),
        py::arg("coord"));
}
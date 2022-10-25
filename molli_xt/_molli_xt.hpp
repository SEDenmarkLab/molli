// ================================================================================
// This file is part of
//      -----------
//      MOLLI 1.0.0
//      -----------
// (C) 2022 Alexander S. Shved and the Denmark laboratory
// University of Illinois at Urbana-Champaign, Department of Chemistry
// ================================================================================

/*
This file defines all necessary stuff for molli extensions
*/

#if !defined(MOLLI_EXTENSIONS)
#define MOLLI_EXTENSIONS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace molli
{
    // shorter namespace alias
    namespace py = pybind11;

    // C-contiguous array with force cast.
    // Useful type alias
    template <typename T>
    using carray = py::array_t<T, py::array::c_style | py::array::forcecast>;

    // short alias for signed size_t
    // python things
    using ssize_t = py::ssize_t;

    // My own little inline for square
    template <typename T>
    inline T square(const T number)
    {
        return number * number;
    }

    // Squared euclidean distance between two points
    template <typename T, ssize_t ND = 3>
    inline T euclidean2(const T *vec1, const T *vec2);

    // Euclidean distance between two points
    template <typename T, ssize_t ND = 3>
    inline T euclidean(const T *vec1, const T *vec2);

    // Matrix-to-maxtrix distance matrix
    // arr1[N,3]  arr2[M,3] --> dm[N,M]
    template <typename T, T (*distance_func)(const T *vec1, const T *vec2)>
    carray<T> cdist22(const carray<T> &arr1, const carray<T> &arr2);

    // Matrix-to-maxtrix MINIMAL distance matrix (over dimension X)
    // arr1[X,N,3]  arr2[M,3] --> dm[N,M]
    template <typename T, T (*distance_func)(const T *vec1, const T *vec2)>
    carray<T> cdist32(const carray<T> &arr1, const carray<T> &arr2);

    // module init functions
    void _init_distance(py::module_ &m);

} // namespace molli

#endif // MOLLI_EXTENSIONS

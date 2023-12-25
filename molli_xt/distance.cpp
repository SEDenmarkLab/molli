// ================================================================================
// This file is part of `molli 1.0`
// (https://github.com/SEDenmarkLab/molli)
//
// Developed by Alexander S. Shved <shvedalx@illinois.edu>
//
// S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
// https://denmarkgroup.illinois.edu/
//
// Copyright 2022-2023 The Board of Trustees of the University of Illinois.
// All Rights Reserved.
//
// Licensed under the terms MIT License
// The License is included in the distribution as LICENSE file.
// You may not use this file except in compliance with the License.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
// ================================================================================

/*
This file defines distance metrics and distance matrix calculating functions
*/

#include <_molli_xt.hpp>
#include <math.h>
using namespace molli;

// Squared euclidean distance between two points
template <typename T, ssize_t ND>
inline T molli::euclidean2(const T *vec1, const T *vec2)
{

    T dist = 0.0f;
    for (ssize_t i = 0; i < ND; i++)
        dist += square(vec1[i] - vec2[i]);
    return dist;
}

// Euclidean distance between two points
template <typename T, ssize_t ND>
inline T molli::euclidean(const T *vec1, const T *vec2)
{
    return sqrt(euclidean2(vec1, vec2));
}

template <typename T, T (*distance_func)(const T *arr1, const T *arr2)>
carray<T> molli::cdist22(const carray<T> &arr1, const carray<T> &arr2)
{
    // Array lengths
    const ssize_t L1 = arr1.shape(0), L2 = arr2.shape(0);

    // Resulting array
    carray<T> result = carray<T>({L1, L2});

    // Pointers to the data
    auto _arr1 = arr1.template unchecked<2>(), _arr2 = arr2.template unchecked<2>();
    auto _result = result.template mutable_unchecked<2>();
    const T *arr1_data_i;

    // Release GIL (Global Interpreter Lock). This is for parallelization purposes.
    py::gil_scoped_release release;

    for (ssize_t i = 0; i < L1; ++i)
    {
        arr1_data_i = _arr1.data(i, 0);
        for (ssize_t j = 0; j < L2; ++j)
            _result(i, j) = distance_func(arr1_data_i, _arr2.data(j, 0));
    }

    return result;
}

// Matrix-to-maxtrix MINIMAL distance matrix (over dimension X)
// arr1[X,N,3]  arr2[M,3] --> dm[X,N,M]
template <typename T, T (*distance_func)(const T *arr1, const T *arr2)>
carray<T> molli::cdist32(const carray<T> &arr1, const carray<T> &arr2)
{
    // Array lengths
    const ssize_t X1 = arr1.shape(0), L1 = arr1.shape(1), L2 = arr2.shape(0);

    // Resulting array
    carray<T> result = carray<T>({X1, L1, L2});

    // Pointers to the data
    auto _arr1 = arr1.template unchecked<3>();
    auto _arr2 = arr2.template unchecked<2>();
    auto _result = result.template mutable_unchecked<3>();
    const T *arr1_data_xi;

    // Release GIL (Global Interpreter Lock). This is for parallelization purposes.
    py::gil_scoped_release release;

    for (ssize_t x = 0; x < X1; ++x)
        for (ssize_t i = 0; i < L1; ++i)
        {
            arr1_data_xi = _arr1.data(x, i, 0);
            for (ssize_t j = 0; j < L2; ++j)
                _result(x, i, j) = distance_func(arr1_data_xi, _arr2.data(j, 0));
        }

    return result;
}

void molli::_init_distance(py::module_ &m)
{
    m.def("cdist22_eu", &cdist22<float, euclidean<float, 3>>, "Computes a Euclidean distance matrix between (M,3) and (N,3) in float32 precision");
    m.def("cdist22_eu", &cdist22<double, euclidean<double, 3>>, "Computes a Euclidean distance matrix between (M,3) and (N,3) in float64 precision");
    m.def("cdist22_eu2", &cdist22<float, euclidean2<float, 3>>, "Computes a Squared Euclidean distance matrix between (M,3) and (N,3) in float32 precision");
    m.def("cdist22_eu2", &cdist22<double, euclidean2<double, 3>>, "Computes a Squared Euclidean distance matrix between (M,3) and (N,3) in float64 precision");
    m.def("cdist32_eu", &cdist32<float, euclidean<float, 3>>, "Computes a Euclidean distance matrix between (X,M,3) and (N,3) in float32 precision");
    m.def("cdist32_eu", &cdist32<double, euclidean<double, 3>>, "Computes a Euclidean distance matrix between (X,M,3) and (N,3) in float64 precision");
    m.def("cdist32_eu2", &cdist32<float, euclidean2<float, 3>>, "Computes a Squared Euclidean distance matrix between (X,M,3) and (N,3) in float32 precision");
    m.def("cdist32_eu2", &cdist32<double, euclidean2<double, 3>>, "Computes a Squared Euclidean distance matrix between (X,M,3) and (N,3) in float64 precision");

    m.def("cdist22f_eu", &cdist22<float, euclidean<float, 3>>, "Computes a Euclidean distance matrix between (M,3) and (N,3) in float32 precision");
    m.def("cdist22d_eu", &cdist22<double, euclidean<double, 3>>, "Computes a Euclidean distance matrix between (M,3) and (N,3) in float64 precision");
    m.def("cdist22f_eu2", &cdist22<float, euclidean2<float, 3>>, "Computes a Squared Euclidean distance matrix between (M,3) and (N,3) in float32 precision");
    m.def("cdist22d_eu2", &cdist22<double, euclidean2<double, 3>>, "Computes a Squared Euclidean distance matrix between (M,3) and (N,3) in float64 precision");

    m.def("cdist32f_eu", &cdist32<float, euclidean<float, 3>>, "Computes a Euclidean distance matrix between (X,M,3) and (N,3) in float32 precision");
    m.def("cdist32d_eu", &cdist32<double, euclidean<double, 3>>, "Computes a Euclidean distance matrix between (X,M,3) and (N,3) in float64 precision");
    m.def("cdist32f_eu2", &cdist32<float, euclidean2<float, 3>>, "Computes a Squared Euclidean distance matrix between (X,M,3) and (N,3) in float32 precision");
    m.def("cdist32d_eu2", &cdist32<double, euclidean2<double, 3>>, "Computes a Squared Euclidean distance matrix between (X,M,3) and (N,3) in float64 precision");
}

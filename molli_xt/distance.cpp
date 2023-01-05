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
    m.def("cdist22_eu_f3", &cdist22<float, euclidean<float, 3>>, "A function that adds all elements of an array.");
    m.def("cdist22_eu_d3", &cdist22<double, euclidean<double, 3>>, "A function that adds all elements of an array.");
    m.def("cdist22_eu2_f3", &cdist22<float, euclidean2<float, 3>>, "A function that adds all elements of an array.");
    m.def("cdist22_eu2_d3", &cdist22<double, euclidean2<double, 3>>, "A function that adds all elements of an array.");

    m.def("cdist32_eu_f3", &cdist32<float, euclidean<float, 3>>, "A function that adds all elements of an array.");
    m.def("cdist32_eu_d3", &cdist32<double, euclidean<double, 3>>, "A function that adds all elements of an array.");
    m.def("cdist32_eu2_f3", &cdist32<float, euclidean2<float, 3>>, "A function that adds all elements of an array.");
    m.def("cdist32_eu2_d3", &cdist32<double, euclidean2<double, 3>>, "A function that adds all elements of an array.");
}
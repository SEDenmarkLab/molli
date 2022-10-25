#include <_molli_xt.hpp>

using namespace molli;

PYBIND11_MODULE(molli_xt, m)
{
    m.doc() = "molli_xt module (pybind11 c++ compiled extensions)";
    _init_distance(m);
}

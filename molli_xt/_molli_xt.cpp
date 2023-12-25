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
This defines the pybind11 module
*/

#include <_molli_xt.hpp>

using namespace molli;

PYBIND11_MODULE(molli_xt, m)
{
    m.doc() = "molli_xt module (pybind11 c++ compiled extensions)";
    _init_distance(m);
}

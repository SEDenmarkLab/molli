# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
`molli.aux` subpackage defines auxiliary helper functions
"""

from .misc import (
    timeit,
    ForeColor,
    unique_path,
    load_external_module,
    catch_interrupt,
)
from . import db
from .version import (
    assert_molli_version_min,
    assert_molli_version_max,
    assert_molli_version_in_range,
)
from .iterators import sglob, dglob, batched, len_batched
from .lock import rwlock

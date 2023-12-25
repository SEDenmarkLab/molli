# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
This file provides cached access to chemically relevant data defined in the *.yml files
in this directory.
"""

import os
from glob import glob
import re
from enum import Enum
from functools import wraps
import yaml

# determine data directories
DATA_ROOT = os.path.abspath(os.path.dirname(__file__))

# assemble
_data_file_paths = glob(f"{DATA_ROOT}/*.yml")

DATA_FILES = {}

for fp in _data_file_paths:
    base = os.path.basename(fp)
    category, dataset, _ = base.split(".")

    if category not in DATA_FILES:
        DATA_FILES[category] = {}

    if dataset not in DATA_FILES[category]:
        DATA_FILES[category][dataset] = {}

    DATA_FILES[category][dataset] = fp

__DATASET_CACHE = {}


def _get_dataset_cached(category: str, dataset: str):
    """
    Retrieve dataset in a cached mode
    example:
    ```python
    _data = molli.data.get_dataset_cached('element', 'vdw_radius')
    ```
    """

    if (category, dataset) not in __DATASET_CACHE:
        with open(DATA_FILES[category][dataset], "rt") as f:
            dat = yaml.safe_load(f)

        if not isinstance(dat, dict) or ("data" not in dat) or ("whatis" not in dat):
            raise SyntaxError(
                f"{DATA_FILES[category][dataset]} is not a valid molli dataset."
            )
        else:
            __DATASET_CACHE[(category, dataset)] = dat

    return __DATASET_CACHE[(category, dataset)]


def remove_from_cache(category: str, dataset: str):
    del __DATASET_CACHE[(category, dataset)]


def get(category: str, dataset: str, key, noexcept=False):
    """
    Retrieves data stored in
    """

    try:
        _dset = _get_dataset_cached(category, dataset)
        property = _dset["data"][key]
    except:
        if noexcept:
            property = None
        else:
            if (category, dataset) in __DATASET_CACHE:
                raise KeyError(
                    f"Cannot retrieve key <{key}> from {DATA_FILES[category][dataset]}::data"
                )
            else:
                raise
    return property


def whatis(category: str, dataset: str) -> dict:
    dataset = _get_dataset_cached(category, dataset)
    return dataset["whatis"]

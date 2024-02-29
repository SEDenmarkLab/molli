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
Test molli package. For most intents and purposes equivalent to `python -m unittest molli_test [args]`
"""
# Test molli installation by running molli test suite

import molli as ml
from pprint import pprint
from argparse import ArgumentParser
import unittest as ut
import molli_test

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser("molli test", description=__doc__, add_help=False)


def molli_main(args, verbosity=0, **kwargs):
    parsed, unknown = arg_parser.parse_known_args(args)
    if verbosity:
        v_arg = ("-" + "v" * verbosity,)
    else:
        v_arg = ()
    ut.main(module=molli_test, argv=["molli test", *unknown, *v_arg])

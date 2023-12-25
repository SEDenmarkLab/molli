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
Show a molecule in a GUI of choice
"""

raise NotImplementedError

import molli as ml
from pprint import pprint
from argparse import ArgumentParser

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli show",
    description=__doc__,
)

arg_parser.add_argument(
    "-l",
    "--library",
    action="store",
    help="Load the molecules to show from this library",
)

arg_parser.add_argument(
    "-c",
    "--command",
    action="store",
    default="avogadro {mol}",
    help="Run this command to get to gui",
)

arg_parser.add_argument(
    "molecules",
    action="store",
    nargs="*",
    help="Load all these molecules from this library",
)


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    print("Will show the following files:")
    for m in parsed.molecules:
        print(f"  > {m}")
    print(f"from library {parsed.library}")

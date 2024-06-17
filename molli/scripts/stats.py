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
Inspect a collection file, such as .mlib, .clib, or .cdxml file.
"""

from argparse import ArgumentParser
import molli as ml
from pathlib import Path
from sys import stderr, stdout
import warnings
from contextlib import nullcontext
from tqdm import tqdm
import math
import pandas as pd

arg_parser = ArgumentParser(
    "molli stats",
    description="Calculate statistics on the collection",
)

arg_parser.add_argument(
    "expression",
    help=(
        "What to count. Expression is evaluated with the local variable `m` that corresponds to the object."
    ),
)

arg_parser.add_argument(
    "input",
    help=(
        "Collection to inspect. If type is not specified, it will be deduced from file extensions"
        " or directory properties."
    ),
)

arg_parser.add_argument(
    "-t",
    "--type",
    default=None,
    choices=["mlib", "clib"],
    help="Collection type",
)

arg_parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output the results as a space-separated file",
)


def molli_main(args, verbose=False, **kwargs):
    parsed = arg_parser.parse_args(args)

    input_type = parsed.type or Path(parsed.input).suffix[1:]

    match input_type:
        case "mlib":
            library = ml.MoleculeLibrary(parsed.input, readonly=True)

        case "clib":
            library = ml.ConformerLibrary(parsed.input, readonly=True)

        case _:
            print(f"Unrecognized input type: {input_type}")
            exit(1)

    # We are going to assume homogeneity of the collection
    with library.reading(), warnings.catch_warnings():
        keys = sorted(library.keys())
        data = [eval(parsed.expression, None, {"m": library[k]}) for k in keys]
        series = pd.Series(data=data, index=keys)

        print(series.describe())

    if parsed.output is not None:
        series.to_csv(parsed.output, header=[parsed.expression])

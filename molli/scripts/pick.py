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
Invoke this script to get a subset of a UKV file
"""

from argparse import ArgumentParser
import molli as ml
from molli.external import openbabel
from itertools import (
    permutations,
    combinations_with_replacement,
    combinations,
    repeat,
    chain,
    product,
)
from typing import Callable
from tqdm import tqdm
import os
from pathlib import Path
import re

arg_parser = ArgumentParser(
    "molli pick",
    description="Merge several molli libraries into a single file",
)

arg_parser.add_argument(
    "source",
    type=Path,
    help="Library file",
)

arg_parser.add_argument(
    "patterns",
    nargs="*",
    default=None,
    help="List of regular expressions to match",
)

arg_parser.add_argument(
    "-f",
    "--file",
    type=Path,
    help="Specify the name patterns from file",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    help="Merge the files into this output file",
)


arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the target files if they exist (default is false)",
    default=False,
)

arg_parser.add_argument(
    "--sort",
    action="store_true",
    help="Sort the keys before writing",
    default=False,
)


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)
    from molli.storage.backends import UKVFile

    if parsed.patterns:
        patterns = parsed.patterns
    elif parsed.file:
        patterns = Path(parsed.file).read_text().splitlines()
    patterns = list(map(re.compile, patterns))

    with (UKVFile(parsed.source, "r") as source,):
        keys = list(map(bytes.decode, source.keys()))
        result = []
        for k in keys:
            for qry in patterns:
                if qry.match(k):
                    result.append(k)
                    break

    if parsed.output:
        with (
            source,
            UKVFile(parsed.output, "w" if parsed.overwrite else "x") as destination,
        ):
            for k in map(str.encode, result):
                destination.put(k, source.get(k))
    else:
        for k in result:
            print(k)

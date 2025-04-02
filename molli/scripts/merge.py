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
Invoke this script to merge several uKV files into one.
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
from glob import glob

arg_parser = ArgumentParser(
    "molli merge",
    description="Merge several molli libraries into a single file",
)

arg_parser.add_argument(
    "sources",
    nargs="+",
    help="List of library files (or glob patterns)",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    required=True,
    help="Merge the files into this output file",
)

arg_parser.add_argument(
    "-c",
    "--comment",
    action="store",
    default=None,
    help="Specify a text based comment to override the uKV default",
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
    files = []
    for source in parsed.sources:
        files.extend(glob(source))

    if files:
        print(f"Matched {len(files)} files for merging.")
    else:
        print("No suitable molecule files were found. Aborting.")
        exit(1)

    from molli.storage.backends import UKVFile

    if parsed.comment:
        h2 = str.encode(parsed.comment)
    else:
        h2 = None

    # Pass 1. Read the uKV keys and, optionally, sort them.
    ukvfiles = [UKVFile(f, "r") for f in files]
    try:
        all_keys = [(k, f) for f in ukvfiles for k in f.keys()]

        if parsed.sort:
            all_keys = sorted(all_keys, key=lambda x: x[0])

        with UKVFile(
            parsed.output, mode="w" if parsed.overwrite else "x", h2=h2
        ) as destn:
            for k, f in tqdm(all_keys, "Merging files"):
                destn.put(k, f.get(k))

    finally:
        # This is in case something goes terribly wrong
        for f in ukvfiles:
            f.close()

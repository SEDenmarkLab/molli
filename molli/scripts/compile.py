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
`molli compile` script is useful to compile a lot of files into a molli collection
"""

from argparse import ArgumentParser
import molli as ml
from importlib.machinery import SourceFileLoader
from glob import glob
from pathlib import Path
from tqdm import tqdm


arg_parser = ArgumentParser(
    "molli compile",
    description="Compile a lot of files into a molli collection",
)

arg_parser.add_argument(
    "sources",
    metavar="<file_or_glob.mol2>",
    action="store",
    type=str,
    nargs="*",
    help="New style collection to be made",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="MLI_FILE",
    action="store",
    type=str,
    default=...,
    help="New style collection to be made",
)

arg_parser.add_argument(
    "--name_as_file_stem",
    action="store_true",
    default=False,
    help="Renames the conformer ensemble to match the file stem",
)

arg_parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Increase the amount of output",
)


def molli_main(args, **kwargs):
    print("This routine will compile all requested mol2 files into a single collection")
    parsed = arg_parser.parse_args(args)
    files = []
    for source in parsed.sources:
        files.extend(glob(source))

    if parsed.verbose:
        for i, fn in enumerate(files):
            print(f"{i+1:>10} | {fn}")

    with ml.ConformerLibrary.new(parsed.output, overwrite=False) as lib:
        print("\nImporting mol2 files:")
        for fn in (pb := tqdm(files, dynamic_ncols=True)):
            fp = Path(fn)
            if parsed.name_as_file_stem:
                name = fp.stem
            else:
                name = None

            ens = ml.ConformerEnsemble.load_mol2(fn, name=name)

            if parsed.verbose:
                pb.write(f"{fn} --> {ens!r}")

            lib.append(ens.name, ens)

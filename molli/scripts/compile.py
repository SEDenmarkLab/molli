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
    description="Compile matching files into a molli collection. Both conformer libraries and molecule libraries are supported.",
)

arg_parser.add_argument(
    "sources",
    metavar="<file_or_glob>",
    action="store",
    type=str,
    nargs="*",
    help="List of source files or a glob pattern.",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="LIB_FILE",
    action="store",
    type=str,
    required=True,
    help="New style collection to be made",
)

arg_parser.add_argument(
    "-t",
    "--type",
    action="store",
    type=str.lower,
    default="molecule",
    choices=["molecule", "ensemble"],
    help="Type of object to be imported",
)

arg_parser.add_argument(
    "-p",
    "--parser",
    action="store",
    type=str.lower,
    default="molli",
    choices=["openbabel", "obabel", "molli"],
    help="Parser to be used to import the molecule object",
)

arg_parser.add_argument(
    "--stem",
    action="store_true",
    default=False,
    help="Renames the conformer ensemble to match the file stem",
)

arg_parser.add_argument(
    "-s",
    "--split",
    action="store_true",
    default=False,
    help="This is only compatible with the choice of type `molecule`. In this case all files are treated as multi-molecule files",
)

arg_parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Increase the amount of output",
)

arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite the destination collection",
)


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)
    files = []
    for source in parsed.sources:
        files.extend(glob(source))

    if files:
        print(f"Matched {len(files)} files for importing.")
    else:
        print("No suitable molecule files were found. Aborting.")
        exit(1)

    if parsed.split:
        raise NotImplementedError(
            "Splitting multimolecule files has not been implemented yet."
        )

    if parsed.type == "conformer":
        library = ml.ConformerLibrary(
            parsed.output,
            overwrite=parsed.overwrite,
            readonly=False,
        )

    elif parsed.type == "molecule":
        library = ml.MoleculeLibrary(
            parsed.output,
            overwrite=parsed.overwrite,
            readonly=False,
        )

    with library.writing():
        for fn in (pb := tqdm(files, dynamic_ncols=True, desc="Importing molecules")):
            fp = Path(fn)
            if parsed.stem:
                name = fp.stem
            else:
                name = None
            mol = ml.load(fn, parser=parsed.parser, otype=parsed.type, name=name)
            name = mol.name
            library[name] = mol

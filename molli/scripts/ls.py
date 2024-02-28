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

arg_parser = ArgumentParser(
    "molli ls",
    description="Read a molli library and list its contents.",
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
    choices=["mlib", "clib", "cdxml"],
    help="Collection type",
)


arg_parser.add_argument(
    "-a",
    "--attrib",
    default=list(),
    nargs="*",
    help=(
        "Attributes to report. At least one must be specified. Attributes are accessed via"
        " `getattr` function. Possible options: `n_atoms`, `n_bonds`, `n_attachment_points`,"
        " `n_conformers` `molecular_weight`, `formula`. If none specified, only the indexes will be"
        " returned."
    ),
)

# arg_parser.add_argument(
#     "-o",
#     "--output",
#     help="Type of output to produce. By default the output produced will be written to stdout.",
#     choices=["json", "json-compact", "yaml", "pandas"],
#     action="default",
#     default="pandas",
# )


def get_attribute(obj, key, sentinel):
    res = getattr(obj, key, sentinel)
    if res is not sentinel:
        return res
    elif key in obj.attrib:
        return obj.attrib[key]
    else:
        raise KeyError(f"{key} is neither an attribute of, nor in .attrib of {obj}")


def molli_main(args, verbose=False, **kwargs):
    parsed = arg_parser.parse_args(args)

    input_type = parsed.type or Path(parsed.input).suffix[1:]

    match input_type:
        case "mlib":
            library = ml.MoleculeLibrary(parsed.input, readonly=True)
            guard = library.reading

        case "clib":
            library = ml.ConformerLibrary(parsed.input, readonly=True)
            guard = library.reading

        case "cdxml":
            library = ml.CDXMLFile(parsed.input)
            guard = nullcontext

        case _:
            print(f"Unrecognized input type: {input_type}")
            exit(1)

    # We are going to assume homogeneity of the collection
    with guard(), warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore")

        L = len(library)
        _sentinel = object()
        if L > 0:
            D = int(math.log10(L)) + 1
            N = max(len(k) for k in library.keys()) + 1
            for i, k in tqdm(
                enumerate(library.keys()),
                total=L,
                disable=not verbose,
            ):
                obj = library[k]
                attrib = {a: get_attribute(obj, a, _sentinel) for a in parsed.attrib}
                s = f""" {i:>{D}}  {k:<{N}} """ + " ".join(
                    f"{x}={y!r}" for x, y in attrib.items()
                )
                tqdm.write(s)

# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Blake E. Ocampo
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
Convert molli library formats
"""

from argparse import ArgumentParser
import molli as ml
from tqdm import tqdm
import zipfile
import msgpack
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import numpy as np
from ..storage import Collection
from molli.external import openbabel as mob
from functools import partial
import sys

arg_parser = ArgumentParser(
    "molli recollect",
    description="Read old style molli collection and convert it to the new file format.",
)

arg_parser.add_argument(
    "-i",
    "--input",
    metavar="<PATH>",
    action="store",
    type=Path,
    help="This is the input path",
)

arg_parser.add_argument(
    "-it",
    "--input_type",
    choices=["mlib", "clib", "dir", "zip"],
    action="store",
    type=str.lower,
    default=None,
    help="This is the input type, including <mlib>, <.clib>, <.zip>, <.xml>, <.ukv>, or directory (<dir>)",
)

arg_parser.add_argument(
    "-iext",
    "--input_ext",
    action="store",
    type=str,
    default=None,
    help="This option is required if reading from a <zip> or directory to indicate the File Type being searched for (<mol2>, <xyz>, etc.)",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="<PATH>",
    action="store",
    type=Path,
    help="This is the output path",
)

arg_parser.add_argument(
    "-ot",
    "--output_type",
    choices=["mlib", "clib", "dir", "zip"],
    action="store",
    type=str,
    default=None,
    help="New style collection, either with or without conformers",
)

arg_parser.add_argument(
    "-oext",
    "--output_ext",
    action="store",
    default="mol2",
    type=str,
    help="This option is required if reading from a <zip> or directory to indicate the File Type being searched for (<mol2>, <xyz>, etc.)",
)

arg_parser.add_argument(
    "-l",
    "--library",
    choices=["molli", "obabel", "openbabel"],
    action="store",
    type=str.lower,
    default="molli",
    help="""This indicates the type of library to utilize, defaults to molli, but openbabel can be specified if non xyz/mol2 formats are used.
    In the event a file format without connectivity is utilized, such as xyz, the molli parser will not create/perceive connectivity, while the 
    openbabel parser will connect/perceive bond orders.
    """,
)

arg_parser.add_argument(
    "-cm",
    "--charge_mult",
    metavar=("0", "1"),
    action="store",
    type=int,
    nargs=2,
    default=[0, 1],
    help="Assign these charge and multiplicity to the imported molecules",
)

arg_parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase the amount of output",
)

arg_parser.add_argument(
    "-s",
    "--skip",
    action="store_true",
    default=False,
    help="This option enables skipping malformed files within old collections. Warnings will be printed.",
)

arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="This option enables overwriting the destination collection.",
)


def recollect(
    source: Collection,
    destination: Collection,
    dest_type: type = None,
    progress: bool = False,
    skip: bool = True,
):
    """This performs a recollection"""
    if dest_type is None:
        dest_type = lambda x: x

    fn = destination._path.name
    with source.reading(), destination.writing():
        for k in (
            pbar := tqdm(source.keys(), disable=not progress, desc=f"Writing into {fn}")
        ):
            try:
                src = source[k]
                res = dest_type(src)
            except Exception as xc:
                if skip:
                    pbar.write(f"Error in {k}: {xc}")
                else:
                    raise
            else:
                destination[k] = res

def recollect_legacy(
        source: ZipFile, 
        destination, 
        charge, 
        mult, 
        molecule,
        dest_type: type = None,
        progress: bool = False,
        skip: bool = True,):

    if dest_type is None:
        dest_type = lambda x: x

    fn = destination._path.name
    with destination.writing():
        for xml in (
            pbar := tqdm(source.namelist(), disable=not progress, desc=f'Writing into {fn}')
        ):
            if xml != '__molli__':
                try:
                    src = ml.chem.ensemble_from_molli_old_xml(source.open(xml), molecule=molecule)
                    src.charge = charge
                    src.mult = mult
                    res = dest_type(src)
                except Exception as xc:
                    if skip:
                        pbar.write(f"Error in {xml}: {xc}")
                    else:
                        raise
                else:
                    destination[src.name] = res

def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    inp = Path(parsed.input)
    input_type = parsed.input_type

    out = Path(parsed.output)
    output_type = parsed.output_type

    # if parsed.output is Ellipsis:
    #     out = inp.with_suffix(".mli")
    #     with ml.aux.ForeColor("yellow"):
    #         print(f"Defaulting to {out} as the output destination.")
    # else:
    #     out = Path(parsed.output)

    charge, mult = parsed.charge_mult

    print(f"Charge and multiplicity: {charge} {mult}")
    if parsed.verbose:
        print("Full paths to files:")
        print(" - Input  ", inp.absolute())
        print(" - Output ", out.absolute())

    # if parsed.skip:
    #     with ml.aux.ForeColor("yellow"):
    #         print(f"Enabled skipping malformed files.")

    if not input_type:
        # deduce the input type here
        if parsed.input.is_dir():
            input_type = "dir"
        else:
            input_type = parsed.input.suffix[1:]
        print(f"Recognized input type as {input_type!r}")

    if not output_type:
        # deduce the input type here
        if parsed.output.is_dir():
            output_type = "dir"
        else:
            output_type = parsed.output.suffix[1:]
        print(f"Recognized output type as {output_type!r}")

    converter = None
    legacy = False
    molecule = False
    
    match input_type:
        case "mlib":
            source = ml.MoleculeLibrary(parsed.input, readonly=True)

        case "clib":
            source = ml.ConformerLibrary(parsed.input, readonly=True)

        case "zip":
            if not is_zipfile(inp):
                raise ValueError(f'{inp} is not a valid zipfile!')
            else:
                with ZipFile(inp, mode='r') as zf:
                    if '__molli__' in zf.NameToInfo:
                        legacy = True
                        zf.close()
                    else:
                        zf.close()
                        suffixes = {Path(x).suffix for x in zf.namelist()}
                        assert len(suffixes) == 1, f'There are not uniform file types in this ZipFile: {suffixes}'

                        source = ml.storage.Collection[dict](
                            parsed.input,
                            ml.storage.ZipCollectionBackend,
                            ext=f'.{parsed.input_ext}',
                            value_decoder=partial(ml.loads, fmt=parsed.input_ext, parser=parsed.library, otype='molecule'),
                            readonly=True,
                            overwrite=False
                            )
        case "dir":
            source = ml.storage.Collection[dict](
                inp,
                ml.storage.DirCollectionBackend,
                ext=f".{parsed.input_ext}",
                value_decoder=partial(ml.loads, fmt=parsed.input_ext, parser=parsed.library, otype='molecule'),
                readonly=True,
                overwrite=False
            )

    match output_type:
        case "mlib":
            destination = ml.MoleculeLibrary(
                parsed.output,
                readonly=False,
                overwrite=parsed.overwrite,
            )
            if input_type == "clib":
                converter = lambda x: ml.Molecule(x[0])

        case "clib":
            destination = ml.ConformerLibrary(
                parsed.output,
                readonly=False,
                overwrite=parsed.overwrite,
            )
            if input_type == "mlib":
                converter = ml.ConformerEnsemble

        case "dir":
            destination = ml.storage.Collection[dict](
                parsed.output,
                ml.storage.DirCollectionBackend,
                ext=f".{parsed.output_ext}", 
                value_encoder=partial(ml.dumps, fmt=parsed.output_ext, writer=parsed.library),
                readonly=False,
                overwrite=True
            )

    if not legacy:
        recollect(source, destination, dest_type=converter, progress=True, skip=parsed.skip)
    else:

        if output_type == 'mlib':
            molecule = True

        with ZipFile(inp, mode='r') as source:
            recollect_legacy(source, destination, charge, mult, molecule=molecule, dest_type=converter, progress=True, skip=parsed.skip)
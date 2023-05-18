"""
This package parses chemical files, such as .cdxml, and creates a collection of molecules in .mlib format.
"""
import molli as ml
from pprint import pprint
from argparse import ArgumentParser
from tqdm import tqdm
from time import sleep
from pathlib import Path

arg_parser = ArgumentParser(
    f"molli parse",
    description=__doc__,
)

arg_parser.add_argument("file", action="store", help="File to be parsed.")

arg_parser.add_argument(
    "-i",
    "--input",
    action="store",
    metavar="<fpath>",
    default=None,
    help=(
        "Override the source file format. Defaults to the file extension. Supported types: 'cdxml'"
    ),
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<fpath>",
    default=None,
    help="Destination for .MLIB output",
)


arg_parser.add_argument(
    "--hadd",
    action="store_true",
    help=(
        "Add implicit hydrogen atoms wherever possible. By default this only affects elements in"
        " groups 13-17."
    ),
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    fpath = Path(parsed.file)
    cdxml_file = ml.CDXMLFile(fpath)

    with ml.MoleculeLibrary.new(fpath.with_suffix(".mlib"), overwrite=False) as lib:
        for k in tqdm(cdxml_file.keys()):
            mol = cdxml_file[k]
            if parsed.hadd:
                mol.add_implicit_hydrogens()
            lib.append(mol.name, mol)

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
    "-f",
    "--format",
    action="store",
    default=None,
    choices=[
        "cdxml",
    ],
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

arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the target files if they exist (default is false)",
    default=False,
)


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)
    fpath = Path(parsed.file)

    iformat = parsed.format or fpath.suffix[1:]
    opath = Path(parsed.output) if parsed.output else fpath.with_suffix(".mlib")
    match iformat:
        case "cdxml":
            input_file = ml.CDXMLFile(fpath)
        case _:
            with ml.aux.ForeColor("ltred"):
                print("Unknown input format: {iformat}")
            exit(1)

    library = ml.MoleculeLibrary(opath, readonly=False, overwrite=parsed.overwrite)

    with library.writing():
        for k in tqdm(input_file.keys(), desc=f"Parsing {fpath}"):
            try:
                mol = input_file[k]
                if parsed.hadd:
                    mol.add_implicit_hydrogens()
                library[mol.name] = mol
            except Exception as e:
                raise RuntimeError(f"Unable to parse {k}") from e

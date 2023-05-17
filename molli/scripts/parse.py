"""
This package parses chemical files, such as .cdxml, and creates a collection of molecules in .mlib format.
"""
import molli as ml
from pprint import pprint
from argparse import ArgumentParser
import tqdm
from time import sleep

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


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    for x in tqdm.tqdm(parsed.files, colour="green", dynamic_ncols=True):
        # print(x)
        sleep(0.5)

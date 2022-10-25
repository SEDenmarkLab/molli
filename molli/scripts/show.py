"""
Show a molecule in a GUI of choice
"""
import molli as ml
from pprint import pprint
from argparse import ArgumentParser

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli show",
    description=__doc__,
)

arg_parser.add_argument(
    "-l",
    "--library",
    action="store",
    help="Load the molecules to show from this library",
)

arg_parser.add_argument(
    "-c",
    "--command",
    action="store",
    default="avogadro {mol}",
    help="Run this command to get to gui",
)

arg_parser.add_argument(
    "molecules",
    action="store",
    nargs="*",
    help="Load all these molecules from this library",
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    print("Will show the following files:")
    for m in parsed.molecules:
        print(f"  > {m}")
    print(f"from library {parsed.library}")

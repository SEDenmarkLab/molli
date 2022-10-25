"""
Print information about current molli package
"""
import molli as ml
from pprint import pprint
from argparse import ArgumentParser

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli info",
    description=__doc__,
)


def molli_main(args, config=None, output=None, **kwargs):
    arg_parser.parse_args(args)

    with ml.aux.ForeColor("yellow"):
        print(ml.__doc__)
        print(f"MOLLI version {MOLLI_VERSION}\n\n")
        print(f"Data root: {ml.data.DATA_ROOT}")

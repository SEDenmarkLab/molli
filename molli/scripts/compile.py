"""
Compile a lot of files into a molli collection
"""

from argparse import ArgumentParser
import molli as ml
from importlib.machinery import SourceFileLoader


arg_parser = ArgumentParser(
    "molli compile",
    description="Compile a lot of files into a molli collection",
)

arg_parser.add_argument("source", action="store", type=str, help="Source file")


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

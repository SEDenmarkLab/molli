"""
Test molli installation. For most intents and purposes equivalent to `python -m unittest molli_test -vvv`
"""
# Test molli installation by running molli test suite

import molli as ml
from pprint import pprint
from argparse import ArgumentParser
import unittest as ut
import molli_test

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser("molli test", description=__doc__, add_help=False)


def molli_main(args, **kwargs):
    parsed, unknown = arg_parser.parse_known_args(args)
    ut.main(module=molli_test, argv=["molli test", *unknown])

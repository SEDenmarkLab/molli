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


def molli_main(args,  **kwargs):
    arg_parser.parse_args(args)
    with ml.aux.ForeColor("yellow"):
        print(ml.config.SPLASH)
    print(ml.__doc__)
    print("HOME:     ", ml.config.HOME)
    print("DATA_DIR: ", ml.config.DATA_DIR)


    

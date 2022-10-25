"""
This package parses chemical files, such as .cdxml, and creates a collection.
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

arg_parser.add_argument("files", nargs="*")


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    for x in tqdm.tqdm(parsed.files, colour="green", dynamic_ncols=True):
        # print(x)
        sleep(0.5)

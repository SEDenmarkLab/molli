"""
Run a custom script that defines `molli_main(args, config, output, **)`
"""
import molli as ml
from pprint import pprint
from argparse import ArgumentParser

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli run",
    description=__doc__,
    add_help=False,
)

arg_parser.add_argument("script", action="store", type=str, metavar="<script.py>")


def molli_main(args, config=None, output=None, **kwargs):
    parsed, unknown = arg_parser.parse_known_args(args)

    with ml.aux.ForeColor("yellow"):
        print(
            "WARNING: this routine can execute arbitrary code. Do not feed untrusted .py files into it!"
        )

    extm = ml.aux.load_external_module(parsed.script, "extm")
    extm.molli_main(unknown, config=config, output=output, **kwargs)

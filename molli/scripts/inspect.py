"""
Inspect a .mlib file.
"""

from argparse import ArgumentParser
import molli as ml

arg_parser = ArgumentParser(
    "molli inspect",
    description="Read a molli library and perform some basic inspections",
)

arg_parser.add_argument(
    "mlib",
    help="Library file to inspect",
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    try:
        lib = ml.ConformerLibrary(parsed.mlib, readonly=True)
        lib[0].n_atoms
    except:
        lib = ml.MoleculeLibrary(parsed.mlib, readonly=True)
    print(lib)
    for i, x in enumerate(lib):
        print(f"{i:>5} | name={x.name!r:<20} formula={x.formula!r}")

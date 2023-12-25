"""
Inspect a .mlib file.
"""

from argparse import ArgumentParser
import molli as ml
from pathlib import Path
from sys import stderr, stdout
import warnings
from contextlib import nullcontext
from tqdm import tqdm
import math

arg_parser = ArgumentParser(
    "molli inspect",
    description="Read a molli library and perform some basic inspections.",
)

arg_parser.add_argument(
    "input",
    help=(
        "Collection to inspect. If type is not specified, it will be deduced from file extensions"
        " or directory properties."
    ),
)

arg_parser.add_argument(
    "-t",
    "--type",
    default=None,
    choices=["mlib", "clib", "cdxml"],
    help="Collection type",
)

arg_parser.add_argument(
    "--timeout",
    default=1.0,
    type=float,
    help="Timeout for a reader lock acquisition",
)

arg_parser.add_argument(
    "-a",
    "--attrib",
    default=list(),
    nargs="*",
    help=(
        "Attributes to report. At least one must be specified. Attributes are accessed via"
        " `getattr` function. Possible options: `n_atoms`, `n_bonds`, `n_attachment_points`,"
        " `n_conformers` `molecular_weight`, `formula`. If none specified, only the indexes will be"
        " returned."
    ),
)

# arg_parser.add_argument(
#     "-o",
#     "--output",
#     help="Type of output to produce. By default the output produced will be written to stdout.",
#     choices=["json", "json-compact", "yaml", "pandas"],
#     action="default",
#     default="pandas",
# )


def get_attribute(obj, key, sentinel):
    res = getattr(obj, key, sentinel)
    if res is not sentinel:
        return res
    elif key in obj.attrib:
        return obj.attrib[key]
    else:
        raise KeyError(f"{key} is neither an attribute of, nor in .attrib of {obj}")


def molli_main(args, verbose=False, **kwargs):
    parsed = arg_parser.parse_args(args)

    input_type = parsed.type or Path(parsed.input).suffix[1:]

    match input_type:
        case "mlib":
            library = ml.MoleculeLibrary(parsed.input, readonly=True)
            guard = library.reading

        case "clib":
            library = ml.ConformerLibrary(parsed.input, readonly=True)
            guard = library.reading

        case "cdxml":
            library = ml.CDXMLFile(parsed.input)
            guard = nullcontext

        case _:
            print(f"Unrecognized input type: {input_type}")
            exit(1)

    # We are going to assume homogeneity of the collection
    with guard(timeout=parsed.timeout), warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore")

        L = len(library)
        _sentinel = object()
        if L > 0:
            D = int(math.log10(L)) + 1
            N = max(len(k) for k in library.keys()) + 1
            for i, k in tqdm(
                enumerate(library.keys()),
                total=L,
                disable=not verbose,
            ):
                obj = library[k]
                attrib = {a: get_attribute(obj, a, _sentinel) for a in parsed.attrib}
                s = f""" {i:>{D}}  {k:<{N}} """ + " ".join(
                    f"{x}={y!r}" for x, y in attrib.items()
                )
                tqdm.write(s)

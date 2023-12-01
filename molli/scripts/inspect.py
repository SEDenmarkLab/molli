"""
Inspect a .mlib file.
"""

from argparse import ArgumentParser
import molli as ml
from pathlib import Path
from sys import stderr, stdout
import warnings

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
    choices=["mlib", "cdxml"],
    help="Collection type",
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

arg_parser.add_argument(
    "-o",
    "--output",
    help="Type of output to produce. By default the output produced will be written to stdout.",
    choices=["json", "json-compact", "yaml", "pandas"],
    action="store",
    default="pandas",
)


def molli_main(args,  **kwargs):
    parsed = arg_parser.parse_args(args)

    input_type = parsed.type or Path(parsed.input).suffix[1:]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        match input_type:
            case "mlib":
                # Right now the library import is a bit syntactically ambiguous
                # Hopefully this will be resolved in later versions
                try:
                    lib = ml.ConformerLibrary(parsed.input, readonly=True)
                    lib[0].n_conformers
                except:
                    lib = ml.MoleculeLibrary(parsed.input, readonly=True)

            case "cdxml":
                lib = ml.CDXMLFile(parsed.input)

            case _:
                print(f"Unrecognized input type: {input_type}")
                exit(1)

        # We are going to assume homogeneity of the collection
        for a in parsed.attrib:
            assert hasattr(
                lib[0], a
            ), f"Unable to locate the following attribute in {lib[0].__class__.__name__}: {a}"

        header = {
            "_molli_collection_class": lib.__class__.__name__,
            "_molli_data_class": lib[0].__class__.__name__,
            "len": len(lib),
        }

        collection = {x: {"id": i} for i, x in enumerate(lib.keys())}

        for k in lib.keys():
            mol = lib[k]
            collection[k] |= {a: getattr(mol, a) for a in parsed.attrib}

        match parsed.output:
            case "json":
                import json

                json.dump({"header": header, "collection": collection}, stdout, indent=2)

            case "json-compact":
                import json

                json.dump({"header": header, "collection": collection}, stdout)

            case "pandas":
                import pandas as pd

                for h in header:
                    print(f"### {h}: {header[h]}")

                df = pd.DataFrame.from_dict(collection, orient="index")
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_colwidth", None)
                pd.set_option("display.width", None)
                print(df)

            case _:
                print("Unrecognized file format", file=stderr)
                return 2

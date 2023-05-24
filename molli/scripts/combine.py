"""
Inspect a .mlib file.
"""

from argparse import ArgumentParser
import molli as ml
from itertools import permutations, combinations_with_replacement, combinations
from tqdm import tqdm

arg_parser = ArgumentParser(
    "molli combine",
    description="Read a molli library and perform some basic inspections",
)

arg_parser.add_argument(
    "cores",
    help="Library file to inspect",
)

arg_parser.add_argument(
    "-s",
    "--substituents",
    action="store",
    metavar="<substituents.mlib>",
    help="Attach the same substituents to each attachment",
    required=True,
)

arg_parser.add_argument(
    "-m",
    "--mode",
    action="store",
    choices=["same", "permutations", "combinations", "combinations_with_replacement"],
    default="permutations",
    help="Attach the same substituents to each attachment",
)

arg_parser.add_argument(
    "-ap",
    "--attachment_points",
    action="store",
    default=None,
    help='Syntax for attachment point selector. Possible variants: "AP0", "AP1,AP2,H12".',
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    required=True,
    help="Attach the same substituents to each attachment",
)

arg_parser.add_argument(
    "-sep",
    "--separator",
    action="store",
    default="_",
    help="Name separator",
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)
    cores: list[ml.Molecule] = ml.MoleculeLibrary(parsed.cores)[:]
    substituents: list[ml.Molecule] = ml.MoleculeLibrary(parsed.substituents)[:]

    assert all(sub.n_attachment_points == 1 for sub in substituents)

    with ml.MoleculeLibrary.new(parsed.output, overwrite=False) as lib:
        for core in tqdm(cores, desc="Processing cores", position=0):
            n_ap = core.n_attachment_points
            ap_idx = core.get_atom_indices(*core.attachment_points)
            for substituents_combo in permutations(substituents, n_ap):
                deriv = ml.Molecule(core)
                for i, (ap_i, sub) in enumerate(zip(ap_idx, substituents_combo)):
                    deriv = ml.Molecule.join(
                        deriv, sub, ap_i - i, sub.attachment_points[0], optimize_rotation=True
                    )
                deriv.name = parsed.separator.join(
                    [core.name] + [sub.name for sub in substituents_combo]
                )
                lib.append(deriv.name, deriv)

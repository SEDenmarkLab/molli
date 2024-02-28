# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================

"""
Show a molecule in a GUI of choice
"""


import molli as ml
from pprint import pprint
from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli show",
    description=__doc__,
)

arg_parser.add_argument(
    "library_or_mol",
    action="store",
    help="This can be a molecule file or a Load all these molecules from this library",
)

arg_parser.add_argument(
    "key",
    action="store",
    nargs="?",
    const=None,
    help="Molecule to be shown. Only applies if the `library_or_mol` argument is a molli collection.",
)

arg_parser.add_argument(
    "-p",
    "--program",
    action="store",
    default="pyvista",
    type=str.lower,
    help="Run this command to get to gui. Special cases: `pyvista`, `3dmol.js`, `http-3dmol.js`. Others are interpreted as command path.",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    help="If any temporary visualization files are producted, they will be written in this destination. User is then responsible for destrying those. If not specified, temporary files will be created.",
)

arg_parser.add_argument(
    "-ot",
    "--otype",
    action="store",
    default=None,
    help="Output temporary file type. defaults to `mol2`",
)

arg_parser.add_argument(
    "--bgcolor",
    action="store",
    default="white",
    help="If the visualization software supports, set this color as background color.",
)

arg_parser.add_argument(
    "--port",
    action="store",
    default="8000",
    help="If the visualization protocol requires to fire up a server, this will be the port of choice.",
)

arg_parser.add_argument(
    "--parser",
    action="store",
    default="molli",
    help="If the visualization requires to load an arbitrary file, this parser will be used to parse out the file.",
)

arg_parser.add_argument(
    "--no_confs",
    action="store_true",
    help="Does not display all conformers of the molecule.",
)

# arg_parser.add_argument(
#     "--orientation",
#     action="store",
#     nargs=9,
#     default=None,
#     type=float,
#     metavar="#",
#     help="This will set the orientation matrix. When applicable.",
# )

f"""
<html encoding="u>
"""


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    path = Path(parsed.library_or_mol)

    libtype = parsed.otype or path.suffix[1:]

    if libtype == "mlib":
        lib = ml.MoleculeLibrary(path, readonly=True)
        with lib.reading():
            mol_or_ens = lib[parsed.key]

    elif libtype == "clib":
        lib = ml.ConformerLibrary(path, readonly=True)
        with lib.reading():
            mol_or_ens = lib[parsed.key]
    else:
        mol_or_ens = ml.load(path, parser=parsed.parser)

    match parsed.program:
        case "pyvista":
            import pyvista as pv
            from pyvista.plotting.plotter import Plotter
            from molli.visual import _pyvista

            plt = Plotter(polygon_smoothing=True)
            plt.set_background(parsed.bgcolor)

            if isinstance(mol_or_ens, ml.Molecule):
                _pyvista.draw_ballnstick(mol_or_ens, plt)
            else:
                _pyvista.draw_ballnstick(mol_or_ens[0], plt)
                if not parsed.no_confs:
                    for c, w in zip(mol_or_ens, mol_or_ens.weights):
                        _pyvista.draw_wireframe(
                            c, plt, opacity=min(w * 5, 1), color_darkness=0
                        )

            plt.show()

        case "3dmol.js":
            ...

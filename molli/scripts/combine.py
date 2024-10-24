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
`molli combine` script is useful when performing a combinatorial expansion of the library
"""

from functools import partial
from argparse import ArgumentParser
import molli as ml
from molli.external import openbabel
from itertools import (
    permutations,
    combinations_with_replacement,
    combinations,
    repeat,
    chain,
    product,
)
from typing import Callable
from tqdm import tqdm
import multiprocessing as mp
import os
from math import comb, perm, factorial
from joblib import Parallel, delayed

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli combine",
    description="Combines two lists of molecules together",
)

arg_parser.add_argument(
    "cores",
    help="Base library file to combine wth substituents",
)

arg_parser.add_argument(
    "-s",
    "--substituents",
    action="store",
    metavar="<substituents.mlib>",
    help="Substituents to add at each attachment of a core file",
    required=True,
)

arg_parser.add_argument(
    "-m",
    "--mode",
    action="store",
    choices=["same", "permutns", "combns", "combns_repl"],
    default="permutns",
    help="Method for combining substituents",
)

arg_parser.add_argument(
    "-a",
    "--attachment_points",
    action="append",
    default=None,
    help="Label used to find attachment points",
)

arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    metavar=1,
    default=1,
    type=int,
    help="Number of processes to be used in parallel",
)

arg_parser.add_argument(
    "-b",
    "--batchsize",
    action="store",
    metavar=1,
    default=256,
    type=int,
    help="Number of molecules to be processed at a time on a single core",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    required=True,
    help="File to be written to",
)

arg_parser.add_argument(
    "-sep",
    "--separator",
    action="store",
    default="_",
    help="Name separator",
)

arg_parser.add_argument(
    "--hadd",
    action="store_true",
    help=("Add implicit hydrogen atoms wherever possible."),
)

arg_parser.add_argument(
    "--obopt",
    nargs="*",
    metavar="ff maxiter tol disp",
    default=None,
    help=(
        "Perform openbabel optimization on the fly. This accepts up to 4 arguments. Arg 1: the"
        " forcefield (uff/mmff94/gaff/ghemical). Arg 2: is the max number of steps (default=500)."
        " Arg 3: energy convergence criterion (default=1e-4) Arg 4: geometry displacement"
        " (default=False) but values ~0.01-0.1 can help escape planarity."
    ),
)


arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the target files if they exist (default is false)",
    default=False,
)


@delayed
def _ml_assemble(
    core: ml.Molecule,
    core_aps: tuple[int],
    substituent_combos: list[tuple[ml.Molecule]],
    hadd: bool = True,
    obopt: Callable = None,
    separator: str = "_",
):
    results = {}
    for substituent_combo in substituent_combos:
        assert len(core_aps) == len(substituent_combo)

        deriv = ml.Molecule(core)
        for i, (ap_i, sub) in enumerate(zip(core_aps, substituent_combo)):
            deriv = ml.Molecule.join(
                deriv, sub, ap_i - i, sub.attachment_points[0], optimize_rotation=True
            )
        deriv.name = separator.join(
            [core.name] + [sub.name for sub in substituent_combo]
        )

        if hadd:
            deriv.add_implicit_hydrogens()

        if callable(obopt):
            obopt(deriv)

        results[deriv.name] = deriv

    return results


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)
    with (_lib := ml.MoleculeLibrary(parsed.cores)).reading():
        cores: list[ml.Molecule] = list(_lib.values())
    with (_lib := ml.MoleculeLibrary(parsed.substituents)).reading():
        substituents: list[ml.Molecule] = list(_lib.values())

    if parsed.obopt is not None:
        match parsed.obopt:
            case []:
                obopt = partial(openbabel.obabel_optimize, inplace=True)

            case [ff]:
                obopt = partial(
                    openbabel.obabel_optimize,
                    ff=ff,
                    inplace=True,
                )

            case [ff, maxiter]:
                obopt = partial(
                    openbabel.obabel_optimize,
                    ff=ff,
                    max_steps=int(maxiter),
                    inplace=True,
                )

            case [ff, maxiter, tol]:
                obopt = partial(
                    openbabel.obabel_optimize,
                    ff=ff,
                    max_steps=int(maxiter),
                    tol=float(tol),
                    inplace=True,
                )

            case [ff, maxiter, tol, disp]:
                obopt = partial(
                    openbabel.obabel_optimize,
                    ff=ff,
                    max_steps=int(maxiter),
                    tol=float(tol),
                    coord_displace=float(disp),
                    inplace=True,
                )

            case _:
                raise ValueError(
                    f"Unsupported arguments for openbabel optimize: {parsed.obopt}"
                )
    else:
        obopt = None
    # TODO: turn all assertions into more meaningful errors
    assert all(sub.n_attachment_points == 1 for sub in substituents)

    ap_indices = (
        [
            [
                core.index_atom(a)
                for lbl in parsed.attachment_points
                for a in core.yield_atoms_by_label(lbl)
            ]
            for core in cores
        ]
        if parsed.attachment_points
        else [list(map(core.index_atom, core.attachment_points)) for core in cores]
    )
    n_aps = len(ap_indices[0])

    # TODO: turn all assertions into more meaningful errors
    assert n_aps > 0, "Did not find any attachment points"
    assert all(
        len(aps) == n_aps for aps in ap_indices
    ), "Cores must have identical number of attachment points"

    n_cores = len(cores)
    n_subst = len(substituents)

    match parsed.mode:
        case "same":
            subst_iter = zip(*repeat(substituents, n_aps))
            n_subst_combos = n_subst
        case "permutns":
            # len = n! / (n - k)!
            subst_iter = permutations(substituents, n_aps)
            n_subst_combos = perm(n_subst, n_aps)
        case "combns":
            # len = n! / k! / (n - k)!
            subst_iter = combinations(substituents, n_aps)
            n_subst_combos = comb(n_subst, n_aps)
        case "combns_repl":
            # from docs: len = (n+r-1)! / r! / (n-1)!
            subst_iter = combinations_with_replacement(substituents, n_aps)
            n_subst_combos = perm(n_subst + n_aps - 1, n_aps) // factorial(n_aps)
        case _:
            raise NotImplementedError(f"Unknown mode: {parsed.mode}")

    lib_size = n_cores * n_subst_combos

    print(f"Will create a library of size {lib_size}")

    parallel = Parallel(n_jobs=parsed.nprocs, return_as="generator")

    library = ml.MoleculeLibrary(
        parsed.output, readonly=False, overwrite=parsed.overwrite
    )

    for results in tqdm(
        parallel(
            _ml_assemble(
                c,
                i,
                sb,
                hadd=parsed.hadd,
                obopt=obopt,
                separator=parsed.separator,
            )
            for (c, i), sb in product(
                zip(cores, ap_indices),
                ml.aux.batched(subst_iter, parsed.batchsize),
            )
        ),
        total=ml.aux.len_batched(n_subst_combos, parsed.batchsize) * n_cores,
    ):
        with library.writing():
            for k, v in results.items():
                library[k] = v

    # Now all the files will be concatenated

    # with ml.MoleculeLibrary.new(parsed.output, overwrite=False) as lib:
    #     for core in tqdm(cores, desc="Processing cores", position=0):
    #         n_ap = core.n_attachment_points
    #         ap_idx = core.get_atom_indices(*core.attachment_points)
    #         for substituents_combo in permutations(substituents, n_ap):
    #             deriv = ml.Molecule(core)
    #             for i, (ap_i, sub) in enumerate(zip(ap_idx, substituents_combo)):
    #                 deriv = ml.Molecule.join(
    #                     deriv, sub, ap_i - i, sub.attachment_points[0], optimize_rotation=True
    #                 )
    #             deriv.name = parsed.separator.join(
    #                 [core.name] + [sub.name for sub in substituents_combo]
    #             )
    #             lib.append(deriv.name, deriv)

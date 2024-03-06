# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Elena S. Burlova
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
`molli align` script is useful to align a library of molecules or conformer ensembles
to a uniform orientation given by a query molecule
"""

from argparse import ArgumentParser
from typing import Callable, Tuple
import molli as ml
import os
from tqdm import tqdm
import scipy.spatial
import rmsd
import pandas as pd

import numpy as np
from pathlib import Path

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli align",
    description="Read a conformer library and align it across given query",
)

arg_parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="ConformerLibrary/MoleculeLibrary file to align",
)

arg_parser.add_argument(
    "-q",
    "--query",
    metavar="query_mol.mol2",
    help="Mol2 file with the reference query structure",
    required=True,
)

arg_parser.add_argument(
    "--rmsd",
    choices=["rmsd", "scipy"],
    default="rmsd",
    help="Method of rmsd calculation. Available are the default and scipy",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="<aligned>",
    default=None,
    help="Output file path and name w/o extension",
)

# arg_parser.add_argument(
#     "-v",
#     "--verbose",
#     metavar="0",
#     default="0",
#     type=int,
#     choices=[0, 1, 2],
#     help="how verbose the log is. 0 - nothing is printed, 1 - some information, 2 - alignment statistics will be returned",
# )

arg_parser.add_argument(
    "-s",
    "--stats",
    default=False,
    type=bool,
    help="True/False flag to save alignment statistics in the separate file. Defaults to False.",
)


def rmsd_kabsch_wrapper(P, Q):
    rotation, _, rmsd_ = rmsd.kabsch_weighted(P, Q)
    return rotation, rmsd_


def scipy_kabsch_wrapper(P, Q):
    rotation, rmsd_ = scipy.spatial.transform.Rotation.align_vectors(P, Q)
    return rotation.as_matrix(), rmsd_


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    # -------------------Reading arguments--------------------------#

    # check for the correct format of the query file
    query_path = Path(parsed.query)
    if query_path.suffix == ".mol2":
        with open(query_path) as f:
            query_mol = ml.chem.Molecule.load_mol2(f, name="query").heavy
    else:
        raise FileNotFoundError("*.mol2 file is required")

    # read Moleculeibrary/ConformerLibrary
    input_path = Path(parsed.input)
    ext = input_path.suffix
    output_path = (
        Path(parsed.output)
        if parsed.output
        else str(input_path.with_suffix("")) + "_aligned" + ext
    )
    print(f"The output path is {output_path}")

    # choosing rmsd calculation method:
    rmsd_func = rmsd_kabsch_wrapper if parsed.rmsd == "rmsd" else scipy_kabsch_wrapper
    # defining save_stats flag:
    save_stats = parsed.stats

    processMoleculeLibrary = True
    match ext:
        case ".mlib":
            source = ml.MoleculeLibrary(input_path, readonly=True)
            destination = ml.MoleculeLibrary(
                output_path,
                readonly=False,
                overwrite=True,
            )
        case ".clib":
            source = ml.ConformerLibrary(input_path, readonly=True)
            destination = ml.ConformerLibrary(
                output_path,
                readonly=False,
                overwrite=True,
            )
            processMoleculeLibrary = False
        case _:
            raise ValueError(
                "This operation is only meant to work on .mlib/.clib files. Received something else"
            )
    # -------------------Performing alignment--------------------------#

    vec = query_mol.centroid()
    query_mol.translate(-vec)

    matched_query_indices = {}

    print("matching structures:")
    with source.reading():
        for elem_name in tqdm(source):
            elem = source[elem_name]
            matched_query_indices[elem_name] = list(elem.get_substr_indices(query_mol))

    align_stats = {}

    reference_subgeometry = query_mol

    reference_subgeometry.translate(-reference_subgeometry.centroid())

    print("bringing to an optimal rotation:")
    with source.reading(), destination.writing():
        for element_name in tqdm(source):
            element = source[element_name]
            rmsd_val = element.align_to_ref_coords(
                rmsd_func,
                matched_query_indices[element_name],
                reference_subgeometry,
                vec,
            )
            align_stats[element_name] = rmsd_val
            element.attrib["name"] = element.name
            element.attrib["coords"] = element.coords
            destination[element_name] = element

    query_mol.translate(vec)

    if processMoleculeLibrary:
        total_rmsds = [round(rmsd_val, 3) for rmsd_val in align_stats.values()]
    else:
        total_rmsds = [
            round(cf_rmsd, 3)
            for cf_rmsd_list in align_stats.values()
            for cf_rmsd in cf_rmsd_list
        ]

    print(
        f"Conformers checked: {len(total_rmsds)}, maximal rmsd: {max(total_rmsds)}, minimal rmsd: {min(total_rmsds)}"
    )
    print(f"Median rmsd: {np.median(total_rmsds)}")

    if save_stats:
        df = pd.DataFrame.from_dict(align_stats, orient="index")
        df.round(3).to_csv(str(input_path.with_suffix("")) + "_stats.csv")

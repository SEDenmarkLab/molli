# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Blake E. Ocampo
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
# `molli.descriptor.cov_cn`

This defines the covalent coordination number calculator.
"""

from ..chem import Molecule, Atom, AtomLike, Substructure
import numpy as np


def dftd_coordination_number(mol: Molecule, a: AtomLike):
    """
    This is the coordination number defined by Grimme in "https://doi.org/10.1063/1.3382344".
    All covalent radii for metals will be scaled down by 10%, and is built to error if the element used is not 1-94.

    This defines a coordination number for individual atoms based on the location of all atoms

    Could use some additional boundary cases, but is currently functioning the way it was implemented by Grimme.
    """

    k1 = 16
    k2 = 4 / 3

    a1 = mol.get_atom(a)

    # assert 0 < max(elem_list := [a.element.z for a in mol.atoms]) < 94, f'There are elements for in this list that do not fall between 1 to 94! {elem_list}'
    assert all(
        0 < a.element < 94 for a in mol.atoms
    ), f"There are elements for in this list that do not fall between 1 to 94!"

    ra1_cov = a1.element.cov_radius_grimme
    # This creates an "atom2" array where the atom is not equal to atom1
    a2_list: list[Atom] = [atom for atom in mol.atoms if atom != a1]

    # This creates an array matching grimme covalent radii to each atom in atom2
    ra2_cov_arr = np.array([atom.element.cov_radius_grimme for atom in a2_list])
    # This is an array of all of the magnitudes of the vector a1 -> a2 multiplied by the constant k2
    # Coordinates of atom1 (shape = (1,3) ) and array of atom2 (shape = (n_atoms-1,3) )
    a1_coords = mol.coord_subset([a1])
    a2_coord_arr = mol.coord_subset(a2_list)

    # This line finds the magnitude of each row with shape of (n_atoms-1,1) (new radius vector)
    rab = np.linalg.norm(
        a2_coord_arr - a1_coords, axis=1
    )  # this can potentially be rewritten using molli_xt (~10x faster)
    inn = -k1 * (k2 * (ra1_cov + ra2_cov_arr) / rab - 1)

    e = 1 + np.exp(inn)
    cn = np.sum(1 / e)

    return cn

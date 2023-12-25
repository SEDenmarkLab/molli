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
# `molli.chem`
Main subpackage of the molli library. 
Contains all fundamental chemistry data structures: `Atom`, `Bond`, `Molecule`, `ConformerEnsemble`, etc.
"""

from .atom import (
    Element,
    ElementLike,
    Atom,
    AtomLike,
    AtomType,
    AtomStereo,
    AtomGeom,
    Promolecule,
    PromoleculeLike,
)

from .bond import (
    Bond,
    Connectivity,
    BondStereo,
    BondType,
)

from .geometry import CartesianGeometry, DistanceUnit
from .structure import Structure, Substructure
from .molecule import Molecule, StructureLike
from .ensemble import ConformerEnsemble, Conformer

from .library import ConformerLibrary, MoleculeLibrary
from .legacy import ensemble_from_molli_old_xml

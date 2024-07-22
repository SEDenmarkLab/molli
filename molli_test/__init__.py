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
# `molli_test`
This package provides unit tests for the required molli functionality
"""

from .test_install import BasicInstallTC
from .test_chem_promolecule import PromoleculeTC
from .test_chem_connectivity import ConnectivityTC
from .test_chem_geometry import GeometryTC
from .test_chem_structure import StructureTC
from .test_chem_molecule import MoleculeTC
from .test_cdxml_parse import CDXMLParserTC
from .test_conformer_ensemble import ConformerEnsembleTC
from .test_descriptor import DescriptorTC
from .test_external_openbabel import OpenbabelTC
from .test_external_rdkit import RDKitTC
from .test_collections import CollectionsTC
from .test_molli_extensions import ExtensionTC
from .test_read_write import ReadWriteTC

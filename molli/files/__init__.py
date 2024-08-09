# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Alexander S. Shved
#               Blake E. Ocampo
#               Casey L. Olen
#               Elena S. Burlova
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
This module defines paths to sample files that can be used for testing or initial development of some code.
"""


from pathlib import Path

ROOT = Path(__file__).parent.absolute()


dendrobine_mol2 = ROOT / "dendrobine.mol2"
nanotube_mol2 = ROOT / "nanotube.mol2"
hadd_test_mol2 = ROOT / "hadd_test.mol2"
pdb_4a05_mol2 = ROOT / "pdb_4a05.mol2"
pentane_confs_mol2 = ROOT / "pentane_confs.mol2"
zincdb_fda_mol2 = ROOT / "zincdb_fda.mol2"
dummy_mol2 = ROOT / "dummy.mol2"
fxyl_mol2 = ROOT / "fxyl.mol2"
dmf_mol2 = ROOT / "dmf.mol2"
bpa_backbone_mol2 = ROOT / "bpa_core.mol2"
box_backbone_mol2 = ROOT / "box_alignment_core.mol2"
isornitrate_mol2 = ROOT / "isornitrate.mol2"

benzene_mol2 = ROOT / "benzene.mol2"
dmf_mol2 = ROOT / "dmf.mol2"


cinchonidine_query = ROOT / "cinchonidine_query.mol2"
cinchonidine_mcs = ROOT / "cinchonidine_mcs.mol2"


dendrobine_xyz = ROOT / "dendrobine.xyz"
pentane_confs_xyz = ROOT / "pentane_confs.xyz"
dummy_xyz = ROOT / "dummy.xyz"

# SDF V3000 MOL molecules
dendrobine_molv3 = ROOT / "dendrobine.mol"
zincdb_fda_molv3 = ROOT / "zincdb_fda.mol"
pentane_confs_molv3 = ROOT / "pentane_confs.mol"

substituents_cdxml = ROOT / "substituents.cdxml"
parser_demo_cdxml = ROOT / "parser_demo.cdxml"
parser_demo2_cdxml = ROOT / "parser_demo2.cdxml"
charges_mult_cdxml = ROOT / "charges_mult.cdxml"

oldstyle_xml = ROOT / "oldstyle.xml"

tiny_bpa_raw_conf = ROOT / "tiny_test_bpa_raw_conf.mlib"
box_no_conf = ROOT / "box_ligands.mlib"
cinchonidine_no_conf = ROOT / "cinchonidine.mlib"
cinchonidine_rd_conf = ROOT / "cinchonidine_rdconfs.clib"
fletcher_phosphoramidite = ROOT / "fletcher_phosphoramidite_cats.mlib"
test_mol2_zip = ROOT / "test_mol2s.zip"
test_mol2_ml02_zip = ROOT / "test_mol2s_ml02.zip"

BOX_4_position = ROOT / "BOX_4position_fragments.cdxml"
BOX_cores = ROOT / "BOX_cores.cdxml"
BOX_bridge = ROOT / "BOX_bridging_fragments.cdxml"

# This module provides easier access to miscellaneous files

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
cinchonidine_query = ROOT / "cinchonidine_query.mol2"
cinchonidine_mcs = ROOT / "cinchonidine_mcs.mol2"

dendrobine_xyz = ROOT / "dendrobine.xyz"
pentane_confs_xyz = ROOT / "pentane_confs.xyz"
dummy_xyz = ROOT / "dummy.xyz"


substituents_cdxml = ROOT / "substituents.cdxml"
parser_demo_cdxml = ROOT / "parser_demo.cdxml"

oldstyle_xml = ROOT / "oldstyle.xml"

tiny_bpa_raw_conf = ROOT / "tiny_test_bpa_raw_conf.mlib"
box_no_conf = ROOT / "box_ligands.mlib"
cinchonidine_no_conf = ROOT / "cinchonidine.mlib"
cinchonidine_rd_conf = ROOT / "cinchonidine_rdconfs.clib"
fletcher_phosphoramidite = ROOT / "fletcher_phosphoramidite_cats.mlib"
test_mol2_zip = ROOT / "test_mol2s.zip"
test_mol2_ml02_zip = ROOT / "test_mol2s_ml02.zip"
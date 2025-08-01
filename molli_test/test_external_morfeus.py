# ================================================================================
# This file is part of `molli`
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
Testing the external functionality related to `OpenBabel`
"""

import unittest as ut

import molli as ml
import importlib.util
import shutil

def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None

class MorfeusTC(ut.TestCase):
    def setUp(self) -> None:
        self.root = ml.config.SCRATCH_DIR 
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    @ut.skipUnless(is_package_installed("morfeus"), "Morfeus is not installed")
    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is required for this test and is not installed")
    def test_buriedvolume(self):
        import morfeus as mf
        import numpy as np
        from molli.external import rdkit as mrd
        from molli.external import morfeus as mmf
        from rdkit.Chem.PropertyMol import PropertyMol
        import os

        mlib = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        with mlib.reading():

            for k in mlib:
                mlmol = mlib[k]
                mlatom_arr = np.array(mlmol.atoms)
                xyz_path = self.root / f'{mlmol.name}.xyz'
                with open(xyz_path, 'w') as f:
                    mlmol.dump_xyz(f)

                #This finds the cinchonidine portion of the atom
                rdmol = mrd.to_rdmol(mlmol, via='mol2', remove_hs=False)
                self.assertIsInstance(rdmol, PropertyMol)
                afmol = mrd.atom_filter(rdmol)

                #cinchonidine nitrogen
                mbool:np.ndarray = afmol.nitrogen_type() & ~afmol.aromatic_type()
                mwhere = np.where(mbool)[0]
                self.assertTrue(np.count_nonzero(mbool) == 1)
                # assert np.count_nonzero(mbool) == 1, f'More than one nitrogen found'

                #Z atom defined as not a part of the quinuclidine 
                zbool = afmol.carbon_type() & afmol.smarts_query("[N]C") & ~afmol.aromatic_type() & ~afmol.in_ring()
                zwhere = np.where(zbool)[0]
                self.assertTrue(np.count_nonzero(zbool) == 1)

                #3 atoms a part of the quinuclidine
                xzbool = afmol.carbon_type() & afmol.smarts_query("NC") & ~zbool
                xzwhere = np.where(xzbool)[0]
                self.assertTrue(np.count_nonzero(xzbool) == 3)
                
                #Morfeus starts from 1
                midx = mwhere[0] + 1
                matom = mlatom_arr[mwhere][0]

                zidx = zwhere + 1
                zatom = mlatom_arr[zwhere]

                xzidx = xzwhere + 1
                xzatom = mlatom_arr[xzwhere]

                #Does the morfeus BuriedVolume calculation
                elements, coordinates = mf.read_xyz(xyz_path)
                bv = mf.BuriedVolume(elements, coordinates, midx, z_axis_atoms=zidx, xz_plane_atoms=xzidx)

                bv.compute_distal_volume(method='sasa', octants=False)

                #Creates dictionary matching the assigned attributes
                bv_res = {
                        'percent_buried_volume': bv.fraction_buried_volume,
                        'buried_volume': bv.buried_volume,
                        'free_volume': bv.free_volume,
                        'distal_volume': bv.distal_volume,
                        'molecular_volume': bv.molecular_volume
                    }
                bv.octant_analysis()
                bv_res['Octants'] = bv.octants
                bv_res['Quadrants'] = bv.quadrants

                #Does the morfeus BuriedVolume calculation
                ml_bv = mmf.buried_volume(mlmol, matom, round_coords=True, z_axis_atoms=zatom, xz_plane_atoms=xzatom, calc_distal_vol=True)

                self.assertTrue(bv_res == ml_bv.attrib['BuriedVolume'])
                os.remove(xyz_path)
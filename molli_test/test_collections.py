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
Testing the collections classes functionality
"""

import unittest as ut
import molli as ml
from tempfile import mkdtemp
from pathlib import Path
import msgpack
import shutil
import os
import gc
from time import sleep
import joblib
from io import UnsupportedOperation
from numpy.testing import assert_allclose


def _write1(i: int, collection: ml.storage.Collection):
    """This writes a simple test dictionary into the collection"""
    d = {"name": (n := f"item{i+1:0>2}"), "value": i**2}
    with collection.writing():  # this ensures proper flushes
        collection[n] = d


class CollectionsTC(ut.TestCase):
    def setUp(self) -> None:
        self.root = ml.config.SCRATCH_DIR / "_molli_test" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def test_dir(self):
        """Testing the most primitive form of data storage"""

        collect = ml.storage.Collection[dict](
            self.root / "collection",
            ml.storage.DirCollectionBackend,
            ext=".dat",
            value_encoder=msgpack.dumps,
            value_decoder=msgpack.loads,
            readonly=False,
        )

        with collect.writing():
            for i in range(50):
                d = {"name": (n := f"item{i+1:0>2}"), "value": i**2}
                collect[n] = d

        with collect.reading():
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

    def test_dir_parallel(self):
        collect = ml.storage.Collection[dict](
            self.root / "collection",
            ml.storage.DirCollectionBackend,
            ext=".dat",
            value_encoder=msgpack.dumps,
            value_decoder=msgpack.loads,
            readonly=False,
        )

        # Yeah, I know, this is probably going to be pretty slow.
        # But it's the principle that we are trying to test here.
        joblib.Parallel(n_jobs=4)(
            joblib.delayed(_write1)(i, collect) for i in range(50)
        )

        with collect.reading():
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

    def test_ukvfile_io(self):
        """This tests the fundamentals"""
        from molli.storage.ukvfile import UKVFile

        comment = (
            b"This is a somewhat lengthy comment that is going to be split in multiple lines."
            b"This is just to test the principle that UKV files can (and should be!) annotated"
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )

        b0 = b"Schema"

        new_entry = b"This is some other data that might replace the old entry."

        # This is the
        with UKVFile(
            self.root / "ukv_write_test.ukv",
            mode="w",
            h1=b"UKVTEST",
            h2=comment,
            b0=b0,
        ) as f:
            for i in range(50):
                f[f"entry_{i+1}".encode()] = comment

            # In this implementation, replacement of keys is not allowed.
            # Should we?
            with self.assertRaises(KeyError):
                f[b"entry_7"] = new_entry

        # Reading the same file should reproduce the data stored
        with UKVFile(
            self.root / "ukv_write_test.ukv",
            mode="r",
        ) as f:
            self.assertEqual(f.h1, b"UKVTEST\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            self.assertEqual(f.h2, comment)
            self.assertEqual(f.b0, b0)
            self.assertEqual(f[b"entry_35"], comment)
            self.assertEqual(f[b"entry_7"], comment)

            self.assertSetEqual(
                set(f.keys()), {f"entry_{i+1}".encode() for i in range(50)}
            )

            # this should raise an exception as it is a read-only stream
            with self.assertRaises(UnsupportedOperation):
                f[b"any_key"] = b"anything"

    def test_ukv(self):
        """Tests a collection with a ukv file backend"""

        collect = ml.storage.Collection[dict](
            self.root / "ukv_collection_test.ukv",
            ml.storage.UkvCollectionBackend,
            value_encoder=msgpack.dumps,
            value_decoder=msgpack.loads,
            readonly=False,
        )

        with collect.writing():
            for i in range(50):
                d = {"name": (n := f"item{i+1:0>2}"), "value": i**2}
                collect[n] = d

        with collect.reading():
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

    def test_ukv_parallel(self):
        """Tests parallel safety of a collection with a ukv file backend"""
        collect = ml.storage.Collection[dict](
            self.root / "collection.ukv",
            ml.storage.UkvCollectionBackend,
            value_encoder=msgpack.dumps,
            value_decoder=msgpack.loads,
            readonly=False,
        )

        joblib.Parallel(n_jobs=4)(
            joblib.delayed(_write1)(i, collect) for i in range(50)
        )

        with collect.reading():
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

    def test_molecule_lib(self):
        source = ml.MoleculeLibrary(ml.files.fletcher_phosphoramidite, readonly=True)
        destination = ml.MoleculeLibrary(
            self.root / "test.mlib",
            readonly=False,
            comment="This is intended to test the molecule collection with implicit conversion between version 1 and 2",
        )

        with source.reading(), destination.writing():
            for mol_name in source:
                mol = source[mol_name]  # Read from source

                # Save some attributes just so the benefits of the new style collections are
                # immediately apparent
                mol.attrib["name"] = mol.name  # strings are serializable by default
                mol.attrib["coords"] = mol.coords  # coordinates are trickier
                destination[mol_name] = mol  # Write into the destination

        with destination.reading():
            self.assertSetEqual(source.keys(), destination.keys())

            for mol_name in destination:
                mol = destination[mol_name]
                self.assertEqual(mol.attrib["name"], mol.name)
                assert_allclose(mol.attrib["coords"], mol.coords)

    def test_ensemble_lib(self):
        source = ml.ConformerLibrary(ml.files.cinchonidine_rd_conf, readonly=True)
        destination = ml.ConformerLibrary(
            self.root / "test.clib",
            readonly=False,
            comment="This is intended to test the ensemble collection with implicit conversion between version 1 and 2",
        )

        with source.reading(), destination.writing():
            for ens_name in source:
                ens = source[ens_name]  # Read from source
                # Save some attributes just so the benefits of the new style collections are
                # immediately apparent
                ens.attrib["name"] = ens.name  # strings are serializable by default
                ens.attrib["coords"] = ens.coords  # coordinates are trickier
                destination[ens_name] = ens  # Write into the destination

        with destination.reading():
            self.assertSetEqual(source.keys(), destination.keys())

            for ens_name in destination:
                ens = destination[ens_name]
                self.assertEqual(ens.attrib["name"], ens.name)
                assert_allclose(ens.attrib["coords"], ens.coords)

    def test_conformer_to_lib(self):
        source = ml.ConformerLibrary(ml.files.cinchonidine_rd_conf, readonly=True)
        destination = ml.MoleculeLibrary(
            self.root / "test.clib",
            overwrite=True,
            readonly=False,
            comment="This is intended to test conformer from an ensemble to Molecule Library",
        )

        with source.reading(), destination.writing():
            for ens_name in source:
                ens = source[ens_name]  # Read from source
                destination[ens_name] = ens[0]

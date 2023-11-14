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


def _write1(i: int, collection: ml.storage.CollectionBase):
    """This writes a simple test dictionary into the collection"""
    d = {"name": (n := f"item{i+1:0>2}"), "value": i**2}
    with collection:  # this ensures proper flushes
        collection[n] = d


class CollectionsTC(ut.TestCase):
    def setUp(self) -> None:
        self.root = ml.config.SCRATCH_DIR / "_molli_test" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def test_dircollect1(self):
        """Testing the most primitive form of data storage"""

        collect = ml.storage.CollectionBase[dict](
            self.root / "collection",
            ml.storage.DirCollectionBackend,
            ext=".dat",
            value_encoder=msgpack.dumps,
            value_decoder=msgpack.loads,
            readonly=False,
        )

        with collect:
            for i in range(50):
                d = {"name": (n := f"item{i+1:0>2}"), "value": i**2}
                collect[n] = d

        with collect:
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

    def test_dircollect_parallel(self):
        collect = ml.storage.CollectionBase[dict](
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

        with collect:
            self.assertEqual(len(collect), 50)
            for n in collect.keys():
                d = collect[n]
                self.assertEqual(d["value"], (int(n[4:]) - 1) ** 2)

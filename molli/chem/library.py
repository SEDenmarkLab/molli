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
# `molli.chem.library`
Defines two essential classes: `MoleculeLibrary` and `ConformerLibrary` that are
used for efficient binary storage of chemical data
"""


from pathlib import Path
from typing import Callable, Iterator, Type
from molli.storage.backends import CollectionBackendBase
from . import Molecule, ConformerEnsemble
from .io import (
    # V1 object notation
    _serialize_mol_v1,
    _serialize_ens_v1,
    _deserialize_mol_v1,
    _deserialize_ens_v1,
    # V2 object notation
    _serialize_mol_v2,
    _serialize_ens_v2,
    _deserialize_mol_v2,
    _deserialize_ens_v2,
    DESCRIPTOR_ENS_V2,
    DESCRIPTOR_MOL_V2,
)
from ..storage import Collection, UkvCollectionBackend
from ..config import VERSION
import msgpack
import gc


class MoleculeLibrary(Collection[Molecule]):
    """This class is an overarching storage system for molecules
    for that allows for both serialization and deserialization.
    Reading and writing takes place using keys as placeholders for
    retrieval

    Examples
    -------
    A MoleculeLibrary can be serialized from existing Molecules
        >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
        >>> mlib = ml.MoleculeLibrary('test.mlib', readonly=False)
        >>> with mlib.writing():
        >>>     lib[dendrobine.name] = dendrobine
        >>> mlib
        MoleculeLibrary(backend=UkvCollectionBackend('test.mlib'), n_items=1)
    A MoleculeLibrary can be deserialized and iterated over
        >>> mlib = ml.MoleculeLibrary('test.mlib')
        >>> with mlib.reading():
        >>>     for key in mlib:
        >>>         mol = mlib[key]
        >>> mol
        Molecule(name='dendrobine', formula='C16 H25 N1 O2')
        >>> mlib
        MoleculeLibrary(backend=UkvCollectionBackend('test.mlib'), n_items=1)
    These actions can be done at the same time to transfer structures
        >>> mlib1 = ml.MoleculeLibrary('test1.mlib')
        >>> mlib2 = ml.MoleculeLibrary('test2.mlib', readonly=False)
        >>> with mlib1.reading(), mlib2.writing():
        >>>     for key in mlib1:
        >>>         mol = mlib1[key]
        >>>         mlib2[mol.name] = mol

    Notes
    -----
    The default behavior with writing will be appending structures, which
    can be changed with the use of `overwrite`
    """

    def __init__(
        self,
        path: Path | str,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        encoding: str = "utf8",
        bufsize: int = -1,
        comment: str = None,
        **kwargs,
    ) -> None:
        # Figure out the correct version of the library

        _v = 2

        if Path(path).is_file():
            try:
                with open(path, "rb") as f:
                    header: bytes = f.read(16)
            except:
                _v = 2
            else:
                if header.startswith(b"ML10Library"):
                    _v = 1

        if _v == 1:
            self._serializer = _serialize_mol_v1
            self._deserializer = _deserialize_mol_v1
            self.descriptor = None
        elif _v == 2:
            self._serializer = _serialize_mol_v2
            self._deserializer = _deserialize_mol_v2
            self.descriptor = msgpack.dumps(DESCRIPTOR_MOL_V2)

        super().__init__(
            path,
            UkvCollectionBackend,
            value_encoder=self._molecule_encoder,
            value_decoder=self._molecule_decoder,
            overwrite=overwrite,
            readonly=readonly,
            encoding=encoding,
            bufsize=bufsize,
            comment=comment,
            b0=self.descriptor,
            **kwargs,
        )
        self._backend: UkvCollectionBackend

    def _molecule_encoder(self, mol: Molecule) -> bytes:
        return msgpack.dumps(self._serializer(mol), use_single_float=True)

    def _molecule_decoder(self, molb: bytes) -> Molecule:
        return self._deserializer(msgpack.loads(molb, use_list=False))


class ConformerLibrary(Collection[ConformerEnsemble]):
    """This class is an overarching storage system for ConformerEnsembles
    for that allows for both serialization and deserialization.
    Reading and writing takes place using keys as placeholders for
    retrieval

    Examples
    -------
    A ConformerLibrary can be serialized from existing Molecules
        >>> ens = ml.Molecule.load_mol2(ml.files.ens_mol2)
        >>> clib = ml.ConformerLibrary('test.clib', readonly=False)
        >>> with clib.writing():
        >>>     lib[ens.name] = ens
        >>> clib
        ConformerLibrary(backend=UkvCollectionBackend('test.clib'), n_items=1)
    A ConformerLibrary can be deserialized and iterated over
        >>> clib = ml.ConformerLibrary('test.clib')
        >>> with clib.reading():
        >>>     for key in clib:
        >>>         ens = clib[key]
        >>> ens
        ConformerEnsemble(name='pentane', formula='C5 H12')
        >>> clib
        ConformerLibrary(backend=UkvCollectionBackend('test.clib'), n_items=1)
    These actions can be done at the same time to transfer structures
        >>> clib1 = ml.ConformerLibrary('test1.clib')
        >>> clib2 = ml.ConformerLibrary('test2.clib', readonly=False)
        >>> with clib1.reading(), clib2.writing():
        >>>     for key in clib1:
        >>>         ens = clib1[key]
        >>>         clib2[ens.name] = ens

    Notes
    -----
    The default behavior with writing will be appending structures, which
    can be changed with the use of `overwrite`
    """

    def __init__(
        self,
        path: Path | str,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        encoding: str = "utf8",
        bufsize: int = -1,
        comment: str = None,
        **kwargs,
    ) -> None:
        # Figure out the correct version of the library
        _v = 2

        if Path(path).is_file():
            try:
                with open(path, "rb") as f:
                    header: bytes = f.read(16)
            except:
                _v = 2
            else:
                if header.startswith(b"ML10Library"):
                    _v = 1

        if _v == 1:
            self._serializer = _serialize_ens_v1
            self._deserializer = _deserialize_ens_v1
            self.descriptor = None
        elif _v == 2:
            self._serializer = _serialize_ens_v2
            self._deserializer = _deserialize_ens_v2
            self.descriptor = msgpack.dumps(DESCRIPTOR_ENS_V2)

        super().__init__(
            path,
            UkvCollectionBackend,
            value_encoder=self._ensemble_encoder,
            value_decoder=self._ensemble_decoder,
            overwrite=overwrite,
            readonly=readonly,
            encoding=encoding,
            bufsize=bufsize,
            comment=comment,
            b0=self.descriptor,
            **kwargs,
        )
        self._backend: UkvCollectionBackend

    def _ensemble_encoder(self, ens: ConformerEnsemble) -> bytes:
        return msgpack.dumps(self._serializer(ens), use_single_float=True)

    def _ensemble_decoder(self, ensb: bytes) -> ConformerEnsemble:
        return self._deserializer(msgpack.loads(ensb, use_list=False))

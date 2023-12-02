# # A library is a dict-like object with a cached access to the elements
from pathlib import Path
from typing import Callable, Type
from molli.storage.backends import CollectionBackendBase
from . import Molecule, ConformerEnsemble
from .io import (
    _serialize_mol_v2,
    _serialize_ens_v2,
    _deserialize_mol_v2,
    _deserialize_ens_v2,
)
from ..storage import Collection, UkvCollectionBackend
import msgpack
import gc


class MoleculeLibrary(Collection[Molecule]):
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
            **kwargs,
        )
        self._backend: UkvCollectionBackend

    @staticmethod
    def _molecule_encoder(mol: Molecule) -> bytes:
        return msgpack.dumps(_serialize_mol_v2(mol), use_single_float=True)

    @staticmethod
    def _molecule_decoder(mb: bytes) -> Molecule:
        return _deserialize_mol_v2(msgpack.loads(mb, use_list=False))


class ConformerLibrary(Collection[ConformerEnsemble]):
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
            **kwargs,
        )
        self._backend: UkvCollectionBackend

    @staticmethod
    def _ensemble_encoder(ens: ConformerEnsemble) -> bytes:
        return msgpack.dumps(_serialize_ens_v2(ens), use_single_float=True)

    @staticmethod
    def _ensemble_decoder(ensb: bytes) -> ConformerEnsemble:
        return _deserialize_ens_v2(msgpack.loads(ensb, use_list=False))

# # A library is a dict-like object with a cached access to the elements
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
        if Path(path).is_file():
            try:
                with open(path, "rb") as f:
                    header: bytes = f.read(16)
            except:
                _v = 2
            else:
                if header.startswith(b"ML10Library"):
                    _v = 1
        else:
            _v = 2

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
        if Path(path).is_file():
            try:
                with open(path, "rb") as f:
                    header: bytes = f.read(16)
            except:
                _v = 2
            else:
                if header.startswith(b"ML10Library"):
                    _v = 1
        else:
            _v = 2

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

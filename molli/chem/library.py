# A library is a dict-like object with a cached access to the elements
from __future__ import annotations
from . import ConformerEnsemble, Molecule
from ..storage import _Library
from typing import Generic, TypeVar, Iterable, List, Literal
from pathlib import Path
import msgpack


class MoleculeLibrary(_Library[Molecule]):
    ...


class ConformerLibrary(_Library[ConformerEnsemble]):
    @staticmethod
    def default_encoder(ens: ConformerEnsemble) -> bytes:
        return msgpack.dumps(ens.serialize())

    @staticmethod
    def default_decoder(bts: bytes) -> ConformerEnsemble:
        return ConformerEnsemble.deserialize(msgpack.loads(bts))

    @classmethod
    def new(
        cls: type[_Library],
        path: Path | str,
        overwrite: bool = False,
    ) -> ConformerLibrary:

        return super().new(path, overwrite=overwrite)

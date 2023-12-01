# A library is a dict-like object with a cached access to the elements
from __future__ import annotations
import io
from . import ConformerEnsemble, Molecule
from ..storage import _Library
from typing import Generic, TypeVar, Iterable, List, Literal
from pathlib import Path
import msgpack

class MoleculeLibrary(_Library[Molecule]):
    @staticmethod
    def default_encoder(mol: Molecule) -> bytes:
        return msgpack.dumps(mol.serialize(), use_single_float=True)

    @staticmethod
    def default_decoder(bts: bytes) -> Molecule:
        return Molecule.deserialize(msgpack.loads(bts, use_list=False))

    @classmethod
    def new(
        cls: type[_Library],
        path: Path | str,
        overwrite: bool = False,
    ) -> MoleculeLibrary:
        return super().new(path, overwrite=overwrite)


class ConformerLibrary(_Library[ConformerEnsemble]):
    @staticmethod
    def default_encoder(ens: ConformerEnsemble) -> bytes:
        return msgpack.dumps(ens.serialize(), use_single_float=True)

    @staticmethod
    def default_decoder(bts: bytes) -> ConformerEnsemble:
        return ConformerEnsemble.deserialize(msgpack.loads(bts, use_list=False))

    @classmethod
    def new(
        cls: type[_Library],
        path: Path | str,
        overwrite: bool = False,
    ) -> ConformerLibrary:
        return super().new(path, overwrite=overwrite)

    def render(
        self,
        which_ens: list[str] | list[int] | slice | str,
        which_cf: list[str] | list[int] | slice | str,
    ):
        assert isinstance(which_ens, str | slice | list), "which_ens must be a list or slice or str"
        assert isinstance(which_cf, str | slice | list), "which_cf must be a list or slice or str"

        v = py3Dmol.view(style="line")

        for ens in self[which_ens]:
            with io.StringIO() as stream:
                for cf in ens[which_cf]:
                    cf.heavy.dump_xyz(stream, fmt="6.3f")

                v.addModels(stream.getvalue())

        v.setBackgroundColor(None)
        v.show()

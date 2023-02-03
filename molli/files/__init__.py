# This module provides easier access to miscellaneous files

from pathlib import Path
from enum import Enum, auto

ROOT = Path(__file__).parent.absolute()


class _FileEnum(Enum):
    @property
    def path(self) -> Path:
        return ROOT / f"""{self.name}.{self.__class__.__name__}"""

    def open(self, mode: str = "t"):
        match mode:
            case "t":
                return open(self.path, "rt")
            case "b":
                return open(self.path, "rb")
            case _:
                raise ValueError("mode must be 't' or 'b'")


class mol2(_FileEnum):
    dendrobine = auto()
    nanotube = auto()
    pdb_4a05 = auto()
    pentane_confs = auto()
    zincdb_fda = auto()
    dummy = auto()


class xyz(_FileEnum):
    dendrobine = auto()
    pentane_confs = auto()
    dummy = auto()
    # nanotube = auto()


class cdxml(_FileEnum):
    substituents = auto()


class xml(_FileEnum):
    oldstyle = auto()

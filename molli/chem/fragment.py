from . import Atom, AtomLike, AtomType, Structure
from numpy.typing import ArrayLike


class Substituent(Structure):
    """
    Substituent is a structure with only one(!) predefined attachment point
    """

    def __init__(
        self,
        other: Structure = None,
        /,
        *,
        n_atoms: int = 0,
        name: str = None,
        coords: ArrayLike = None,
        copy_atoms: bool = False,
        charge: int = None,
        mult: int = None,
        attachment_point=AtomLike,
        **kwds,
    ):
        super().__init__(
            other,
            n_atoms=n_atoms,
            name=name,
            coords=coords,
            copy_atoms=copy_atoms,
            charge=charge,
            mult=mult,
            **kwds,
        )

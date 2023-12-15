from . import Atom, AtomLike, AtomType, Structure
from numpy.typing import ArrayLike


class Substituent(Structure):
    """
    Substituent is a structure with only one(!) predefined attachment point

    Args:
        other (Structure, optional): Another Structure object to initialize from.
        n_atoms (int, optional): The number of atoms in the substituent.
        name (str, optional): The name of the substituent.
        coords (ArrayLike, optional): The coordinates of the substituent's atoms.
        copy_atoms (bool, optional): Flag indicating whether to copy the atoms from `other`.
        charge (int, optional): The charge of the substituent.
        mult (int, optional): The multiplicity of the substituent.
        attachment_point (AtomLike): The predefined attachment point of the substituent.
        kwds: Additional keyword arguments to pass to the parent class.
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

from . import Atom, AtomLike, AtomType, Structure
from numpy.typing import ArrayLike


class Substituent(Structure):
    """
    Substituent is a structure with only one(!) predefined attachment point

    :param other: Another Structure object to initialize from.
    :type other: Structure, optional
    :param n_atoms: The number of atoms in the substituent.
    :type n_atoms: int, optional
    :param name: The name of the substituent.
    :type name: str, optional
    :param coords: The coordinates of the substituent's atoms.
    :type coords: ArrayLike, optional
    :param copy_atoms: Flag indicating whether to copy the atoms from `other`.
    :type copy_atoms: bool, optional
    :param charge: The charge of the substituent.
    :type charge: int, optional
    :param mult: The multiplicity of the substituent.
    :type mult: int, optional
    :param attachment_point: The predefined attachment point of the substituent.
    :type attachment_point: AtomLike
    :param kwds: Additional keyword arguments to pass to the parent class.
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

from . import Atom, AtomLike, AtomType, Structure
from numpy.typing import ArrayLike


class Substituent(Structure):
    """Substituent is a structure with only one(!) predefined attachment point"""

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
        """
        Parameters
        ----------
        other : Structure, optional
            Another Structure object to initialize from, by default None
        n_atoms : int, optional
            The number of atoms in the substituent, by default 0
        name : str, optional
            The name of the substituent, by default None
        coords : ArrayLike, optional
            The coordinates of the substituent's atoms, by default None
        copy_atoms : bool, optional
            Flag indicating whether to copy the atoms from `other`,
            by default False
        charge : int, optional
            The charge of the substituent, by default None
        mult : int, optional
            The multiplicity of the substituent, by default None
        attachment_point : _type_, optional
            The predefined attachment point of the substituent, by default AtomLike
        """
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

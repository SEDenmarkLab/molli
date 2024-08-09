# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Alexander S. Shved <shvedalx@illinois.edu>,
#               Elena S. Burlova
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
# `molli.chem.ensemble`

This submodule defines the `ConformerEnsemble` class.
"""

from __future__ import annotations
from typing import IO, Iterable, Iterator, List, Callable, Generator, Tuple
from . import (
    Molecule,
    Atom,
    AtomLike,
    Element,
    Promolecule,
    Bond,
    Connectivity,
    CartesianGeometry,
    Structure,
    Substructure,
    StructureLike,
    PromoleculeLike,
)

# from .geometry import _nans
from .structure import RE_MOL_NAME, RE_MOL_ILLEGAL
from ..parsing import read_mol2

import numpy as np
from numpy.typing import ArrayLike
from warnings import warn
from io import StringIO
from pathlib import Path
from deprecated import deprecated


class ConformerEnsemble(Connectivity):
    """This is a fundamental class of Molli that employs methods that work on
    a collection of conformers. This is built to treat all conformers as having
    a single Connectivity and various coordinates associated with the
    atoms of each Conformer. The ensemble supports iteration and collective
    transformations.

    Examples
    -------
    ConformerEnsemble can be initialized from a multi-mol2 file
        >>> ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
    It can also be initialized from a list of molecules
        >>> mol_list = ml.Molecule.load_all_mol2(ml.files.pentane_confs_mol2)
        >>> ml.ConformerEnsemble(mol_list)
        ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
    """

    def __init__(
        self,
        other: ConformerEnsemble = None,
        /,
        n_conformers: int = 0,
        n_atoms: int = 0,
        *,
        name: str = None,
        charge: int = None,
        mult: int = None,
        coords: ArrayLike = None,
        weights: ArrayLike = None,
        atomic_charges: ArrayLike = None,
        copy_atoms: bool = False,
        **kwds,
    ):
        # TODO: revise the constructor

        if isinstance(other, list) and all(isinstance(o, Structure) for o in other):
            super().__init__(
                other[0],
                name=other[0].name,
                charge=other[0].charge,
                mult=other[0].mult,
                **kwds,
            )
            n_conformers = len(other)

            self._coords = np.full((n_conformers, self.n_atoms, 3), np.nan)
            self._atomic_charges = np.zeros((n_conformers, self.n_atoms))
            self._weights = np.ones((n_conformers,))

            self.atomic_charges = [c.atomic_charges for c in other]
            self.coords = [c.coords for c in other]
        else:
            super().__init__(
                other,
                n_atoms=n_atoms,
                name=name,
                copy_atoms=copy_atoms,
                charge=charge,
                mult=mult,
                **kwds,
            )
            self._coords = np.full((n_conformers, self.n_atoms, 3), np.nan)
            self._atomic_charges = np.zeros((n_conformers, self.n_atoms))
            self._weights = np.ones((n_conformers,))

        if isinstance(other, ConformerEnsemble):
            self._atomic_charges = np.array(other.atomic_charges)
            self._coords = np.array(other.coords)
            self._weights = np.array(other.weights)

        if isinstance(other, Molecule):
            self._coords = np.full((n_conformers or 1, self.n_atoms, 3), np.nan)
            self._atomic_charges = np.zeros((n_conformers or 1, self.n_atoms))
            self._weights = np.ones((n_conformers or 1,))

        if coords is not None:
            self.coords = coords

        if atomic_charges is not None:
            self.atomic_charges = atomic_charges

        if weights is not None:
            self.weights = weights

    @property
    def coords(self) -> np.ndarray:
        """Set of atomic positions in shape (n_confs, n_atoms, 3)

        Returns
        -------
        np.ndarray
            Returns array of coordinates

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.coords
            array([[[-2.8045e+00,  3.9964e+00, -1.4128e+00],...
        """
        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        self._coords[:] = other

    @property
    def weights(self) -> np.ndarray:
        """The weights of conformers in the ensemble in shape (n_confs,)

        Returns
        -------
        np.ndarray
            Returns array of weights

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.weights
            array([1., 1., 1., 1., 1., 1., 1.])
        """

        return self._weights

    @weights.setter
    def weights(self, other: ArrayLike):
        self._weights[:] = other

    @property
    def atomic_charges(self) -> np.ndarray:
        """The atomic charges of the ensemble in shape (n_confs,n_atoms)

        Returns
        -------
        np.ndarray
            Returns the atomic charges of the ensemble

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.atomic_charges
            array([[-0.0653, -0.0559, ...], [-0.0653, -0.0559, ...], ...
        """

        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        self._atomic_charges[:] = other

    @classmethod
    @deprecated("This function has been replaced with `load_mol2`", version="1.0.0")
    def from_mol2(
        cls: type[ConformerEnsemble],
        input: str | StringIO,
        /,
        name: str = None,
        charge: int = None,
        mult: int = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """ """
        mol2io = StringIO(input) if isinstance(input, str) else input
        mols = Molecule.load_all_mol2(mol2io, name=name, source_units=source_units)

        return cls(mols)

    @classmethod
    def load_mol2(
        cls: type[ConformerEnsemble],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Loads mol2 from a file path, string, or stream

        Parameters
        ----------
        cls : type[ConformerEnsemble]
            Class to be loaded into
        input : str | Path | IO
            File path, string, or stream
        name : str, optional
            Name for ConformerEnsemble, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        ConformerEnsemble
            Returns ConformerEnsemble

        Examples
        -------
            >>> ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
        """

        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            mols = Molecule.load_all_mol2(stream, source_units=source_units)

        return cls(mols, name=name)

    @classmethod
    def loads_mol2(
        cls: type[ConformerEnsemble],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Loads mol2 from a string

        Parameters
        ----------
        cls : type[ConformerEnsemble]
            Class to be loaded into
        input : str
            Mol2 block as string
        name : str, optional
            Name for ConformerEnsemble, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        ConformerEnsemble
            Returns ConformerEnsemble

        Examples
        -------
            >>> with open(ml.files.pentane_confs_mol2, 'r') as f:
            >>>     ml.ConformerEnsemble.loads_mol2(f.read())
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
        """

        stream = StringIO(input)
        return cls.load_mol2(stream)

    @classmethod
    def load_xyz(
        cls: type[ConformerEnsemble],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Loads xyz from a file path, string, or stream

        Parameters
        ----------
        cls : type[ConformerEnsemble]
            Class to be loaded into
        input : str | Path | IO
            File path, string, or stream
        name : str, optional
            Name for ConformerEnsemble, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        ConformerEnsemble
            Returns ConformerEnsemble

        Examples
        -------
            >>> ml.ConformerEnsemble.load_xyz(ml.files.pentane_confs_xyz)
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
        """

        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            mols = Molecule.load_all_xyz(stream, source_units=source_units)

        return cls(mols, name=name)

    @classmethod
    def loads_xyz(
        cls: type[ConformerEnsemble],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Loads xyz from a string

        Parameters
        ----------
        cls : type[ConformerEnsemble]
            Class to be loaded into
        input : str
            xyz block as string
        name : str, optional
            Name for ConformerEnsemble, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        ConformerEnsemble
            Returns ConformerEnsemble

        Examples
        -------
            >>> with open(ml.files.pentane_confs_xyz, 'r') as f:
            >>>     ml.ConformerEnsemble.loads_xyz(f.read())
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=7)
        """

        stream = StringIO(input)
        return cls.load_xyz(stream)

    def dump_mol2(self, stream: StringIO) -> None:
        """Dumps the multi-mol2 block into the output stream

        Parameters
        ----------
        stream : StringIO, optional
            Output stream, by default None

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> with open('test.mol2', 'w') as f:
            >>>     ens.dump_mol2(f)
            # Produced with molli package
            @<TRIPOS>MOLECULE
            pentane
            ...
        """

        for conf in self:
            conf.dump_mol2(stream)

    def dumps_mol2(self) -> str:
        """Dumps the multi-mol2 block as a string

        Returns
        -------
        str
            The multi-mol2 block

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.dumps_mol2()
            # Produced with molli package
            @<TRIPOS>MOLECULE
            pentane
            ...
        """

        with StringIO() as stream:
            self.dump_mol2(stream)
            return stream.getvalue()

    def dump_xyz(self, stream: StringIO):
        """Dumps the .xyz file into the stream

        Parameters
        ----------
        stream : StringIO, optional
            Stream into which the xyz values are written.

        """
        for conf in self:
            conf.dump_xyz(stream)

    def dumps_xyz(self):
        """Returns the string of the .xyz file

        Returns
        -------
        str
            String containing .xyz file of multi-conformer molecule.
        """
        stream = StringIO()
        self.dump_xyz(stream)
        return stream.getvalue()

    @property
    def n_conformers(self) -> int:
        """
        Returns
        -------
        int
            Returns number of conformers in the ensemble

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.n_conformers
            7
        """

        return self._coords.shape[0]

    def __iter__(self):
        self._current_mol_index = 0
        return self

    def __next__(self) -> Conformer:
        idx = self._current_mol_index
        if idx in range(self.n_conformers):
            m = self[idx]
            self._current_mol_index += 1
        else:
            raise StopIteration
        return m

    def __getitem__(self, locator: int | slice) -> Conformer | List[Conformer]:
        match locator:
            case int() as _i:
                return Conformer(self, _i)

            case slice() as _s:
                return [
                    Conformer(self, _i) for _i in range(*_s.indices(self.n_conformers))
                ]

            case _:
                raise ValueError("Cannot use this locator")

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = (
            f"ConformerEnsemble(name='{self.name}', formula='{_fml}',"
            f" n_conformers={self.n_conformers})"
        )
        return s

    def filter(
        self,
        fx: Callable[[Conformer], bool],
    ): ...

    def serialize(self):
        atom_id_map = {a: i for i, a in enumerate(self.atoms)}

        # atoms = [
        #     (a.element.z, a.label, a.isotope, a.dummy, a.stereo) for a in self.atoms
        # ]
        # bonds = [
        #     (atom_index[b.a1], atom_index[b.a2], b.order, b.aromatic, b.stereo)
        #     for b in self.bonds
        # ]

        atoms = [a.as_tuple() for a in self.atoms]
        bonds = [b.as_tuple(atom_id_map=atom_id_map) for b in self.bonds]

        return (
            self.name,
            self.n_conformers,
            self.n_atoms,
            atoms,
            bonds,
            self.charge,
            self.mult,
            self.coords.astype(">f4").tobytes(),
            self.weights.astype(">f4").tobytes(),
            self.atomic_charges.astype(">f4").tobytes(),
        )

    @classmethod
    def deserialize(cls: type[ConformerEnsemble], struct: tuple):
        (
            _name,
            _nc,
            _na,
            _atoms,
            _bonds,
            _charge,
            _mult,
            _coords,
            _weights,
            _atomic_charges,
        ) = struct

        coords = np.frombuffer(_coords, dtype=">f4").reshape((_nc, _na, 3))
        weights = np.frombuffer(_weights, dtype=">f4")
        atomic_charges = np.frombuffer(_atomic_charges, dtype=">f4")

        atoms = []
        for i, a in enumerate(_atoms):
            atoms.append(Atom(*a))

        res = cls(
            atoms,
            n_conformers=_nc,
            n_atoms=_na,
            name=_name,
            charge=_charge,
            mult=_mult,
            coords=coords,
            weights=weights,
            atomic_charges=atomic_charges,
        )

        for i, b in enumerate(_bonds):
            a1, a2, *b_attr = b
            res.append_bond(Bond(atoms[a1], atoms[a2], *b_attr))

        return res

    def extend(self, others: ConformerEnsemble | Iterable[CartesianGeometry]) -> None:
        """Extends ConformerEnsemble

        Parameters
        ----------
        others : ConformerEnsemble | Iterable[CartesianGeometry]
            Iterable object with a `coords` property

        Examples
        -------
            >>> ens1 = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens2 = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens1.extend(ens2)
            >>> ens1
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=14)
        """
        if isinstance(others, ConformerEnsemble):
            other_coords = others.coords
        else:
            other_coords = np.array([g.coords for g in others])

        self._coords = np.append(self._coords, other_coords, axis=0)

    def append(self, other: CartesianGeometry) -> None:
        """_summary_

        Parameters
        ----------
        other : CartesianGeometry
            Object with a `coords` property

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> mol = ens[0]
            >>> ens.append(mol)
            >>> ens
            ConformerEnsemble(name='pentane', formula='C5 H12', n_conformers=8)
        """
        if self._coords.shape == (0, 0, 3):
            self._coords = other.coords[np.newaxis, :]
        else:
            self._coords = np.append(self._coords, [other.coords], axis=0)

    def scale(self, factor: float, allow_inversion=False) -> None:
        """Scale the coordinates by a factor. This also scales the atomic charges

        Parameters
        ----------
        factor : float
            Factor to scale by
        allow_inversion : bool, optional
            Allows inversion of coordinates, by default False

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.coords
            array([[[-2.8045e+00,  3.9964e+00, -1.4128e+00],...
            >>> ens.scale(0.5)
            >>> ens.coords
            array([[[-1.40225e+00,  1.99820e+00, -7.06400e-01],...
        """

        if factor < 0 and not allow_inversion:
            raise ValueError(
                "Scaling with a negative factor can only be performed with explicit `scale(factor,"
                " allow_inversion = True)`"
            )

        if factor == 0:
            raise ValueError("Scaling with a factor == 0 is not allowed.")

        self._coords *= factor

    def invert(self) -> None:
        """Coordinates are inverted wrt the origin. This also inverts
        inverts the absolute stereochemistry

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.coords
            array([[[-2.8045e+00,  3.9964e+00, -1.4128e+00],...
            >>> ens.invert
            >>> ens.coords
            array([[[ 2.8045e+00, -3.9964e+00,  1.4128e+00],...
        """

        self.scale(-1, allow_inversion=True)

    def get_substr_indices(
        self, pattern: Connectivity
    ) -> Generator[list[int], None, None]:
        """
        Yields all possible combinations of substructure indices that matched
        with the given pattern.

        Parameters:
        -----------
        pattern: Connectivity

        Returns:
        --------
        Generator over list of all possible mappings to pattern


        If only one variation of substructure indices is needed, use
        next(ens.get_substr_indices(pattern))

        Examples
        -------
        >>> for ens in tqdm(library):
        >>>    for mapping in ens.get_substr_indices(pattern):
        >>>        ...
        """
        mappings = self.match(
            pattern,
            node_match=Connectivity._node_match,
            edge_match=Connectivity._edge_match,
        )
        ens_atom_idx = {a: i for i, a in enumerate(self.atoms)}

        for mapping in mappings:
            yield [ens_atom_idx[mapping[x]] for x in pattern.atoms]

    # NOTE: this function is different to translate function from geometry! (explain why)
    def translate(self, vector: ArrayLike):
        """Translates coordinates by a set amount

        Parameters
        ----------
        vector : ArrayLike
            Array for translation

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.coords
            array([[[-2.8045e+00,  3.9964e+00, -1.4128e+00],...
            >>> ens.translate([1,1,1])
            >>> ens.coords
            array([[[-1.8045,  4.9964, -0.4128],

        """
        v = np.array(vector)
        match v.ndim:
            case 1:
                self.coords += v
            case 2:
                self.coords += v[:, np.newaxis, :]
            case _:
                raise ValueError("wrong shape of vector")

    def rotate(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Rotates coordinates by a set rotation matrix

        Parameters
        ----------
        rotation_matrix : np.ndarray
            Rotation matrix for ConformerEnsemble

        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens.coords
            array([[[-2.8045e+00,  3.9964e+00, -1.4128e+00],...
            >>> ens.rotate(np.array([[1,0,0],[0,0,-1],[0,1,0]])) #90 deg Rot X-axis
            >>> ens.coords
            array([[[-2.8045e+00, -1.4128e+00, -3.9964e+00],
        """
        self.coords = self.coords @ rotation_matrix

    def center_at_atom(self, _a: Atom):
        """Translates coordinates of ensemble placing an atom at the origin

        Parameters
        ----------
        _a : Atom
            Atom to center coordinates around
        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> a = ens.get_atom(0)
            >>> ens.center_at_atom(a)
            array([[[0,0,0],...
        """

        atom_ind = self.index_atom(_a)

        self.translate(-self.coords[:, atom_ind])

        # previous working version:
        # for cf in self:
        #     atom_coord = cf.coords[atom_ind]
        #     cf.coords -= atom_coord

    def center_at_core(self, substructure_indices: list[int]):
        """
        Centers ensemble at its substructure so that the coordinates of centroid of the
        this substructure are at the origin.
        """
        centroids = np.array(
            [cf.substructure(substructure_indices).centroid() for cf in self]
        )
        # self.coords -= centroids[:, np.newaxis]
        self.translate(-centroids)

        # previous working version:
        # for cf in self:
        #     cnf_subgeom = cf.substructure(substructure_indices)
        #     cf.coords -= cnf_subgeom.centroid()

    def optimal_rotation_to_ref_coords(
        self,
        func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        substr_indices: list[list[int]],
        reference_subgeometry: Substructure,
    ) -> Tuple[list[float], np.ndarray]:
        """
        This is the inner part of the main function for Conformer alignment. For each conformer in the ConformerEnsemble,
        it does the following:
        1. Finds optimal rotation using symmetry corrected rmsd. It calculates rmsd (root mean squared deviation) for
        every possible mapping with reference coordinates, then picks the lowest rmsd and corresponding rotation matrix.
        2. Rotates conformer with the resulting rotation matrix.
        3. Returns list with the lowest rmsd values for each conformer.

        Parameters:
        -----------
        func: callable
            function that calculates rmsd value and rotation matrix.
            It accepts two ndarrays of coordinates and returns optimal rotation
            matrix as ndarray and minimal rmsd value as float.
        substr_indices: list
            list of all possible mappings to the alignment core
        reference_subgeometry: ml.chem.Substructure
            Referentce coordinates for the alignment

        Returns:
        --------
        rmsds: list
            List of rmsd values for each conformer in the given ConformerEnsemble

        Notes:
        ------
        If core_indices list has only one element, the algorithm will perform the usual (non symmetry corrected) alignment
        """

        rmsds = []
        opt_rot_ms = []

        for cf in self:
            smallest_rmsd = 100.0
            optimal_rot_matrix = None

            for idx in substr_indices:
                cnf_subgeom = cf.substructure(idx)

                rotation, rmsd_ = func(cnf_subgeom.coords, reference_subgeometry.coords)

                if rmsd_ < smallest_rmsd:
                    smallest_rmsd = rmsd_
                    optimal_rot_matrix = rotation

            rmsds.append(smallest_rmsd)
            opt_rot_ms.append(optimal_rot_matrix)
            # cf.coords = cf.coords @ optimal_rot_matrix

        return rmsds, np.array(opt_rot_ms)

    def align_to_ref_coords(
        self,
        func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        substructure_indices: list[list[int]],
        reference_subgeometry: Substructure,
        vec: list = None,
    ) -> list:
        """
        This is the outer part of the main function for the ConformerEnsemble alignment.

        Parameters:
        -----------
        func: Callable
            function that calculates rmsd value and rotation matrix.
            It accepts two ndarrays of coordinates and returns optimal rotation
            matrix as ndarray and minimal rmsd value as float.
        substr_indices: list
            list of all possible mappings to the alignment core
        reference_subgeometry: Substructure
            Referential coordinates for the alignment
        vec: list = None
            translation vector. If vec is not None, the whole ConformerEnsemble
            will be translated on that vector.

        Returns:
        --------
        rmsds: list
            List of rmsd values for each conformer in the ConformerEnsemble

        Notes:
        ------
        Reference_subgeometry should be centered at the origin before calling
        this function.
        """

        # Most alignment algorithms require centering the ensemble first
        self.center_at_core(substructure_indices[0])

        # Finding optimal rotation
        rmsds, rot_matrix = self.optimal_rotation_to_ref_coords(
            func, substructure_indices, reference_subgeometry
        )

        self.rotate(rot_matrix)

        # bringing ensemble coordinates from origin back to referential coordinates (if necessary)
        if vec is not None:
            self.translate(vec)
            # for cf in self:
            #     cf.coords += vec

        return rmsds


class Conformer(Molecule):
    """Conformer class behaves like a molecule, yet is completely virtual.
    More documentation about the use of this class can be found
    in the `Molecule` documentation

    Examples
    -------
        >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        >>> ens[0]
        Conformer(name='pentane', conf_id=0)
    """

    def __init__(self, parent: ConformerEnsemble, conf_id: int):
        self._parent = parent
        self._conf_id = conf_id

    @property
    def name(self) -> str:
        """Name of the Conformer

        Returns
        -------
        str
            The name
        Examples
        -------
            >>> ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
            >>> ens[0].name
            pentane
        """

        return self._parent.name

    @property
    def _atoms(self):
        return self._parent.atoms

    @property
    def _bonds(self):
        return self._parent.bonds

    @property
    def _coords(self):
        return self._parent._coords[self._conf_id]

    @property
    def _atomic_charges(self):
        return self._parent._atomic_charges[self._conf_id]

    @property
    def attrib(self):
        return self._parent.attrib

    @_coords.setter
    def _coords(self, other):
        self._parent._coords[self._conf_id] = other

    @property
    def charge(self):
        return self._parent.charge

    @property
    def mult(self):
        return self._parent.mult

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"Conformer(name='{self.name}', conf_id={self._conf_id})"
        return s

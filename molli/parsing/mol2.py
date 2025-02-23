# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
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
This is a parser for .mol2 files
"""

from dataclasses import dataclass, field
from typing import List, Generator
from io import StringIO
from warnings import warn
from ._reader import LineReader
from collections import deque
import re


@dataclass(slots=True, init=True)
class MOL2Atom:
    _idx: str
    label: str
    _x: str
    _y: str
    _z: str
    mol2_type: str = None
    _subst_idx: str = None
    subst_name: str = None
    _charge: str = None
    status_bit: str = None
    unknown: str = None
    attrib: dict = field(default_factory=dict)

    @property
    def element(self):
        return self.typ.split(".")[0]

    @property
    def idx(self):
        return int(self._idx)

    @property
    def xyz(self):
        return [float(self._x), float(self._y), float(self._z)]

    @property
    def subst_idx(self):
        return int(self._subst_idx)

    @property
    def charge(self):
        return float(self._charge)


@dataclass(slots=True, init=True)
class MOL2Bond:
    _idx: str
    _a1: str
    _a2: str
    mol2_type: str
    status_bit: str = None
    unknown: str = None
    attrib: dict = field(default_factory=dict)

    @property
    def idx(self):
        return int(self._idx)

    @property
    def a1(self):
        return int(self._a1)

    @property
    def a2(self):
        return int(self._a2)


@dataclass(slots=True, init=True)
class MOL2Header:
    name: str = None
    mol_type: str = None
    chrg_type: str = None
    n_atoms: int = None
    n_bonds: int = None
    n_substructs: int = None
    n_feats: int = None
    n_sets: int = None
    comment: str = None


@dataclass(slots=True, init=True)
class MOL2Block:
    header: MOL2Header
    atoms: List[MOL2Atom]
    bonds: List[MOL2Bond]


class MOL2SyntaxError(SyntaxError):
    """Error is raised when a syntax error is detected within MOL2 file"""

    ...


RE_COMMENT = re.compile(r"#.*")
RE_TRIPOS = re.compile(r"@<TRIPOS>([A-Z_]+)")

MOL2_EMPTY_STR = "****"


def read_mol2(input: StringIO) -> Generator[MOL2Block, None, None]:
    reader = LineReader(input, str.strip)
    skip_lines = False
    skipped = 0

    parsed_header = None
    parsed_atoms = None
    parsed_bonds = None
    # current_substructures = None

    while (line := reader.next_noexcept()) is not None:
        match line:
            case "":
                # Empty lines should be ignored
                continue

            case _ if RE_COMMENT.match(line):
                # These are the comment lines
                continue

            case _ if m := RE_TRIPOS.match(line):
                # =========== MAIN MATCHING PROCEDURE ============
                skip_lines = False

                match m[1]:
                    case "MOLECULE":
                        if parsed_header is not None:
                            yield MOL2Block(parsed_header, parsed_atoms, parsed_bonds)

                        mol_name = next(reader)
                        mol_record_counts = next(reader)
                        mol_type = next(reader)
                        chrg_type = next(reader)
                        # So the tricky thing is that status bits are not necessarily going to be there...
                        # in this case they should match RE_TRIPOS
                        status_bits = next(reader)

                        _mrc = list(map(int, mol_record_counts.split()))

                        n_atoms = None
                        n_bonds = None
                        n_substructs = None
                        n_feats = None
                        n_sets = None

                        match _mrc:
                            case [na]:
                                n_atoms = na

                            case [na, nb]:
                                n_atoms = na
                                n_bonds = nb

                            case [na, nb, ns, *rest]:
                                n_atoms = na
                                n_bonds = nb
                                n_substructs = ns

                            case _:
                                raise MOL2SyntaxError(
                                    "Invalid molecule record counts (line"
                                    f" {reader.pos})"
                                )

                        if RE_TRIPOS.match(status_bits):
                            reader.put_back(status_bits)
                            comment = None
                        elif status_bits == MOL2_EMPTY_STR:
                            comment = next(reader)
                        else:
                            comment = None

                        parsed_header = MOL2Header(
                            name=mol_name,
                            mol_type=mol_type,
                            chrg_type=chrg_type,
                            n_atoms=n_atoms,
                            n_bonds=n_bonds,
                            n_substructs=n_substructs,
                            n_feats=n_feats,
                            n_sets=n_sets,
                            comment=comment,
                        )

                    case "ATOM":
                        parsed_atoms: list[MOL2Atom] = []
                        for i in range(n_atoms):
                            atom_def = next(reader)
                            # (
                            #     _id,  # atom id (same as atom index)
                            #     _lbl,  # atom label
                            #     _x,
                            #     _y,
                            #     _z,
                            #     _type,
                            #     _subid,
                            #     _sub,
                            #     _chrg,
                            # ) = atom_def.strip().split(maxsplit=9)

                            # _element = str.capitalize(_type.split(".")[0])

                            # parsed_atoms.append(
                            #     MOL2Atom(
                            #         idx=int(_id),
                            #         label=_lbl if _lbl.upper() != _element.upper() else _lbl.capitalize()+_id,
                            #         x=float(_x),
                            #         y=float(_y),
                            #         z=float(_z),
                            #         typ=_type,
                            #         subst_idx=int(_subid),
                            #         subst_name=_sub,
                            #         charge=float(_chrg),
                            #     )
                            # )
                            parsed_atoms.append(MOL2Atom(*atom_def.split(maxsplit=10)))

                    case "BOND":
                        parsed_bonds = []
                        for _ in range(n_bonds):
                            bond_def = next(reader)
                            # _id, _a1, _a2, _type = bond_def.split()
                            # # a1, a2 = mol.yield_atoms(int(_a1) - 1, int(_a2) - 1)

                            # order = 1
                            # match _type:
                            #     case "ar":
                            #         order = 1.5
                            #     case "_":
                            #         order = float(_type)

                            # parsed_bonds.append(
                            #     MOL2Bond(
                            #         idx=int(_id), a1=int(_a1), a2=int(_a2), typ=_type
                            #     )
                            # )
                            parsed_bonds.append(MOL2Bond(*bond_def.split(maxsplit=5)))

                    # case "SUBSTRUCTURE":
                    #     ...

                    case "UNITY_ATOM_ATTR":
                        # This is TRIPOS mol2 way of storing atomic attributes.
                        while True:
                            line = next(reader)
                            if RE_TRIPOS.match(line):
                                reader.put_back(line)
                                break

                            atom, n_attr = map(int, line.split())
                            for _ in range(n_attr):
                                attr, value = next(reader).split()
                                parsed_atoms[atom - 1].attrib[attr] = value

                    case "UNITY_BOND_ATTR":
                        while True:
                            line = next(reader)
                            if RE_TRIPOS.match(line):
                                reader.put_back(line)
                                break

                            bond, n_attr = map(int, line.split())
                            for _ in range(n_attr):
                                attr, value = next(reader).split()
                                parsed_bonds[bond - 1].attrib[attr] = value

                    case _:
                        warn(
                            f"TRIPOS block '{m[1]}' is not implemented in"
                            " current version of MOLLI!. Lines will be"
                            " skipped.",
                        )
                        skip_lines = True

            case _:
                if not skip_lines:
                    raise MOL2SyntaxError(
                        f"Unexpected syntax at line {reader.pos}: '{line}'"
                    )
                else:
                    skipped += 1

    # if skipped > 0:
    #     warn(
    #         "This mol2 stream contained lines within TRIPOS blocks not"
    #         f" supported by MOLLI. skipped {skipped} lines",
    #     )

    yield MOL2Block(parsed_header, parsed_atoms, parsed_bonds)

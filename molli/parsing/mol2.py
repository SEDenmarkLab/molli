from dataclasses import dataclass
from typing import List, Generator
from io import StringIO
from warnings import warn
from ._reader import LineReader
from collections import deque
import re


@dataclass(slots=True, init=True)
class MOL2Atom:
    idx: int = None
    label: str = None
    x: float = None
    y: float = None
    z: float = None
    typ: str = None
    subst_idx: int = None
    subst_name: str = None
    charge: float = None
    # status_bit: str = None

    @property
    def element(self):
        return self.typ.split(".")[0]


@dataclass(slots=True, init=True)
class MOL2Bond:
    idx: int = None
    a1: int = None
    a2: int = None
    typ: str = None

    @property
    def order(self):
        try:
            ord = float(self.typ)
        except:
            if self.typ == "ar":
                return 1.5
            else:
                return -1
        else:
            return ord


@dataclass(slots=True, init=True)
class MOL2Header:
    name: str = None
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
RE_TRIPOS = re.compile(r"@<TRIPOS>([A-Z]+)")


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
                        charge_type = next(reader)
                        status_bits = next(reader)

                        if "***" in status_bits:
                            comment = next(reader)
                        else:
                            comment = None

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
                                    f"Invalid molecule record counts (line {reader.pos})"
                                )

                        parsed_header = MOL2Header(
                            name=mol_name,
                            n_atoms=n_atoms,
                            n_bonds=n_bonds,
                            n_substructs=n_substructs,
                            n_feats=n_feats,
                            n_sets=n_sets,
                            comment=comment,
                        )

                    case "ATOM":
                        parsed_atoms = []
                        for i in range(n_atoms):
                            atom_def = next(reader)
                            (
                                _id,  # atom id (same as atom index)
                                _lbl,  # atom label
                                _x,
                                _y,
                                _z,
                                _type,
                                _subid,
                                _sub,
                                _chrg,
                            ) = atom_def.strip().split(maxsplit=9)

                            _element = str.capitalize(_type.split(".")[0])

                            parsed_atoms.append(
                                MOL2Atom(
                                    idx=int(_id),
                                    label=_lbl if _lbl.upper() != _element.upper() else _lbl.capitalize()+_id,
                                    x=float(_x),
                                    y=float(_y),
                                    z=float(_z),
                                    typ=_type,
                                    subst_idx=int(_subid),
                                    subst_name=_sub,
                                    charge=float(_chrg),
                                )
                            )

                    case "BOND":
                        parsed_bonds = []
                        for _ in range(n_bonds):
                            bond_def = next(reader)
                            _id, _a1, _a2, _type = bond_def.split()
                            # a1, a2 = mol.yield_atoms(int(_a1) - 1, int(_a2) - 1)

                            order = 1
                            match _type:
                                case "ar":
                                    order = 1.5
                                case "_":
                                    order = float(_type)

                            parsed_bonds.append(
                                MOL2Bond(
                                    idx=int(_id), a1=int(_a1), a2=int(_a2), typ=_type
                                )
                            )

                    # case "SUBSTRUCTURE":
                    #     ...

                    case _:
                        warn(
                            f"TRIPOS block '{m[1]}' is not implemented in current version of MOLLI!. Lines will be skipped.",
                        )
                        skip_lines = True

            case _:
                if not skip_lines:
                    raise MOL2SyntaxError(
                        f"Unexpected syntax at line {reader.pos}: '{line}'"
                    )
                else:
                    skipped += 1

    if skipped > 0:
        warn(
            f"This mol2 stream contained lines within TRIPOS blocks not supported by MOLLI. skipped {skipped} lines",
        )

    yield MOL2Block(parsed_header, parsed_atoms, parsed_bonds)

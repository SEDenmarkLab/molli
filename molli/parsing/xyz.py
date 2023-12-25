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
This is a parser for .xyz files
"""

from dataclasses import dataclass
from typing import List, Generator
from io import StringIO
from ._reader import LineReader


@dataclass(slots=True, init=True)
class XYZAtom:
    symbol: str
    x: float
    y: float
    z: float


@dataclass(slots=True, init=True)
class XYZBlock:
    n_atoms: int
    comment: str
    atoms: List[XYZAtom]

    @property
    def symbols(self):
        return [a.symbol for a in self.atoms]

    @property
    def coords(self):
        return [(a.x, a.y, a.z) for a in self.atoms]


class XYZSyntaxError(SyntaxError):
    """Error is raised when a syntax error is detected within XYZ file"""

    ...


def read_xyz(input: StringIO) -> Generator[XYZBlock, None, None]:
    """
    Parse .xyz file with strict syntax
    """
    reader = LineReader(input)

    while line := reader.next_noexcept():
        try:
            n_atoms = int(line)
        except ValueError:
            raise XYZSyntaxError(
                f"Expected number of atoms at line {reader.pos - 1}, received: {line}"
            )

        try:
            comment = str.strip(next(reader))
        except StopIteration:
            raise XYZSyntaxError(f"Expected comment at line {reader.pos}, not EOF")

        atoms = []

        try:
            for _ in range(n_atoms):
                atom_line = next(reader)
                a, x, y, z = atom_line.split()
                atoms.append(XYZAtom(a, float(x), float(y), float(z)))

        except StopIteration:
            raise XYZSyntaxError(
                f"Expected atom definitions at line {reader.pos}, not EOF"
            )

        except:
            raise XYZSyntaxError(f"Unknown syntax at line {reader.pos}:\n\t{atom_line}")

        yield XYZBlock(n_atoms=n_atoms, comment=comment, atoms=atoms)

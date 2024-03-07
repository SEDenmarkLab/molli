# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Casey L. Olen
#               Alexander S. Shved
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
This defines the way that molli can launch XTB jobs.
"""

import os
import re
import shlex
from math import degrees
from pathlib import Path
from pprint import pprint
from subprocess import PIPE, run
from tempfile import TemporaryDirectory, mkstemp
from typing import Any, Callable, Generator, Iterable
from io import StringIO

import attrs
import msgpack
import numpy as np
from joblib import Parallel, delayed

from ..chem import AtomLike, ConformerEnsemble, Molecule
from .driver import DriverBase
from .job import Job, JobInput, JobOutput


class XTBDriver(DriverBase):
    default_executable = "xtb"

    @Job(return_files=("xtbopt.xyz",)).prep
    def optimize_m(
        self,
        M: Molecule,
        charge: int = None,
        mult: int = None,
        method: str = "gff",
        crit: str = "loose",
        xtbinp: str = "",
        maxiter: int = 500,
        misc: str = None,
    ):
        """Optimize a molecule with XTB"""
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"
        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} input.xyz --{method} --opt {crit} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1}"""
                    f""" --iterations {maxiter} {"--input param.inp" if xtbinp else ""} -P {self.nprocs} {misc or ""}""",
                    "xtb",
                ),
            ],
            files={
                "input.xyz": M.dumps_xyz().encode(),
            },
            return_files=self.return_files,
        )

        return inp

    @optimize_m.post
    def optimize_m(
        self,
        out: JobOutput,
        M: Molecule,
        **kwargs,
    ):
        if res := out.files["xtbopt.xyz"]:
            xyz = res.decode()

            # the second line of the xtb output is not needed - it is the energy line
            xyz_coords = (
                xyz.split("\n")[0] + "\n" + "\n" + "\n".join(xyz.split("\n")[2:])
            )
            optimized = Molecule(M, coords=Molecule.loads_xyz(xyz_coords).coords)

            return optimized

    optimize_ens = Job.vectorize(optimize_m)

    @optimize_ens.reduce
    def optimize_ens(
        self,
        outputs: Iterable[Molecule],
        ens: ConformerEnsemble,
        *args,
        **kwargs,
    ):
        """Optimize a conformer ensemble with XTB"""
        newens = ConformerEnsemble(ens)
        for new_conf, old_conf in zip(outputs, newens):
            old_conf.coords = new_conf.coords
        return newens

    @Job().prep
    def energy_m(
        self,
        M: Molecule,
        charge: int = None,
        mult: int = None,
        method: str = "gfn2",
        accuracy: float = 0.5,
        xtbinp: str = "",
        maxiter: int = 2000,
        misc: str = None,
    ):
        # assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} input.xyz --{method} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1} --acc {accuracy:0.2f}"""
                    f""" --iterations {maxiter} {"--input param.inp" if xtbinp else ""} -P {self.nprocs} {misc or ""}""",
                    "xtb",
                )
            ],
            files={
                "input.xyz": M.dumps_xyz().encode(),
            },
            return_files=self.return_files,
        )

        return inp

    @energy_m.post
    def energy(self, out: JobOutput, M: Molecule, **kwargs):
        if res := out.stdouts[self.executable]:
            for l in res.split("\n")[::-1]:
                if m := re.match(
                    r"\s+\|\s+TOTAL ENERGY\s+(?P<eh>[0-9.-]+)\s+Eh\s+\|.*", l
                ):
                    M.attrib["energy"] = float(m["eh"])
                    return M

    @Job(return_files=()).prep
    def atom_properties_m(
        self,
        M: Molecule,
        charge: int = None,
        mult: int = None,
        method: str = "gfn2",
        accuracy: float = 0.5,
        xtbinp: str = "",
        maxiter: int = 500,
        misc: str = None,
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} input.xyz --{method} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1} --acc {accuracy:0.2f} --vfukui"""
                    f""" --iterations {maxiter} {"--input param.inp" if xtbinp else ""} -P {self.nprocs} {misc or ""}""",
                    "xtb",
                )
            ],
            files={
                "input.xyz": M.dumps_xyz().encode(),
            },
            return_files=self.return_files,
        )

        return inp

    @atom_properties_m.post
    def atom_properties_m(self, out: JobOutput, M: Molecule, **kwargs):
        from molli.parsing.xtbout import extract_xtb_atomic_properties

        if res := out.stdouts["xtb"]:
            outdf = extract_xtb_atomic_properties(res)
            for i, a in enumerate(M.atoms):
                for j, property in enumerate(outdf.columns):
                    a.attrib[property] = outdf.iloc[i, j]
            return M

    @Job().prep
    def scan_dihedral(
        self,
        M: Molecule,
        dihedral_atoms: tuple[AtomLike],
        method: str = "gfn2",
        accuracy: float = 0.5,
        range_deg: tuple[float] = (0.0, +360.0),
        n_steps: int = 72,
        maxiter_per_step: int = 16,
        force_const: float = 0.5,
        charge: int = None,
        mult: int = None,
        **kwargs,
    ):
        d0 = degrees(M.dihedral(*dihedral_atoms)) + range_deg[0]
        d1 = d0 + range_deg[1]
        idx = ",".join(str(x + 1) for x in M.get_atom_indices(*dihedral_atoms))

        inp_file = (
            f"$constrain\n"
            f"  force constant={force_const}\n"
            f"  dihedral: {idx},{d0:0.3f}\n"
            f"$scan\n"
            f"  1: {d0},{d1},{n_steps}\n"
            f"$opt\n"
            f"  maxcycle={maxiter_per_step}\n"
            f"$end\n\n"
        )

        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} mol.xyz --{method} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1} --acc {accuracy:0.2f} --opt --input scan.inp -P {self.nprocs} """,
                    "xtb",
                )
            ],
            files={
                "mol.xyz": M.dumps_xyz().encode(),
                "scan.inp": inp_file.encode(),
            },
            return_files=("xtbscan.log",),
        )

        return inp

    @scan_dihedral.post
    def scan_dihedral(
        self,
        out: JobOutput,
        M: Molecule,
        dihedral_atoms: tuple[AtomLike],
        method: str = "gfn2",
        accuracy: float = 0.5,
        range_deg: tuple[float] = (0.0, +360.0),
        n_steps: int = 72,
        maxiter_per_step: int = 16,
        force_const: float = 0.5,
        charge: int = None,
        mult: int = None,
        **kwargs,
    ):
        scan = out.files["xtbscan.log"].decode()
        scan_io = StringIO(scan)

        result = ConformerEnsemble(M, n_conformers=n_steps)

        energies = []
        i = 0
        while n := next(scan_io, None):
            assert int(n) == M.n_atoms
            _, nrg, _ = next(scan_io).split(maxsplit=2)
            energies.append(float(nrg))

            for j in range(M.n_atoms):
                _, x, y, z = next(scan_io).split()
                result.coords[i, j] = [float(x), float(y), float(z)]

            i += 1

        result.attrib["XTB/Conformer_Energies"] = np.array(energies)

        return result

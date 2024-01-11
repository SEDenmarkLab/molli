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
from pathlib import Path
from pprint import pprint
from subprocess import PIPE, run
from tempfile import TemporaryDirectory, mkstemp
from typing import Any, Callable, Generator, Iterable

import attrs
import msgpack
import numpy as np
from joblib import Parallel, delayed

from ..chem import Molecule, ConformerEnsemble
from .job import Job, JobInput, JobOutput
from .driver import DriverBase


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
        if pls := out.files["xtbopt.xyz"]:
            xyz = pls.decode()

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
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            command=f"""{self.executable} input.xyz --{method} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1} --acc {accuracy:0.2f}""",
            files={"input.xyz": M.dumps_xyz().encode()},
        )

        return inp

    @energy_m.post
    def energy(self, out: JobOutput, M: Molecule, **kwargs):
        if pls := out.stdout:
            for l in pls.split("\n")[::-1]:
                if m := re.match(
                    r"\s+\|\s+TOTAL ENERGY\s+(?P<eh>[0-9.-]+)\s+Eh\s+\|.*", l
                ):
                    return float(m["eh"])

    @Job(return_files=()).prep
    def atom_properties_m(
        self,
        M: Molecule,
        charge: int = None,
        mult: int = None,
        method: str = "gfn2",
        accuracy: float = 0.5,
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --charge {charge or M.charge} --uhf {(mult or M.mult) - 1} --acc {accuracy:0.2f} --vfukui""",
            files={"input.xyz": M.dumps_xyz().encode()},
        )

        return inp

    @atom_properties_m.post
    def atom_properties_m(self, out: JobOutput, M: Molecule, **kwargs):
        from molli.parsing.xtbout import extract_xtb_atomic_properties

        if pls := out.stdout:
            # print(pls)

            outdf = extract_xtb_atomic_properties(pls)
            for i, a in enumerate(M.atoms):
                for j, property in enumerate(outdf.columns):
                    a.attrib[property] = outdf.iloc[i, j]
            return M

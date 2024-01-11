# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Casey L. Olen
#               Alexander S. Shved
#               Ian Rinehart
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
from ..config import SCRATCH_DIR


class NWChemDriver(DriverBase):
    default_executable = "nwchem"

    # the nwchem .esp file contains BOTH optimized coordinates and esp charges
    @Job(return_files=(r"*.esp",)).prep
    def optimize_atomic_esp_charges(
        self,
        M: Molecule,
        basis: str = "def2-svp",
        functional: str = "b3lyp",
        maxiter: int = 100,
        memory_total: int = 500, # memory in MB,
        optimize_geometry: bool = True, # optimize geometry before ESP calc?
    ):
        """(Optionally) optimize molecule geometry with NWChem, then calculate atomic esp with NWChem esp module. ESPs are assigned as atom 'nwchem_esp_charge' Atom attributes in the returned Molecule object."""
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"
        
        full_xyz = M.dumps_xyz()
        xyz_block = "\n".join(full_xyz.split("\n")[2:])[:-3]

        _inp = (f"""title "{M.name} ESP Calculation"\n"""
                f"""memory total {memory_total} mb\n"""
                """echo\n"""
                """start\n"""
                """\n"""
                """geometry units angstroms\n"""
                f"""{xyz_block}\n"""
                """end\n"""
                f"""charge {M.charge}\n"""
                """\n"""
                """driver\n"""
                f"""    maxiter {maxiter}\n"""
                """end\n"""
                """\n"""
                """basis\n"""
                f"""* library {basis}\n"""
                """end\n"""
                """\n"""
                """dft\n"""
                f"""    xc {functional}\n"""
                f"""    maxiter {maxiter}\n"""
                """end\n"""
                """\n"""
                f"""{'task dft optimize' if optimize_geometry else ''}\n""" # optimize geometry
                """\n"""
                """esp\n"""
                """    recalculate\n"""
                """    range 0.2\n"""
                """    probe 0.1\n"""
                """    spacing 0.025\n"""
                """end\n"""
                """\n"""
                """task esp\n"""
                """\n"""
                """\n"""
            )


        inp = JobInput(
            M.name,
            commands=[
                (   
                    f"""mpirun -np {self.nprocs} {self.executable} esp.inp""",
                    "nwchem",
                ),
            ],
            files={
                f"esp.inp": _inp.encode(),
            },
            return_files=self.return_files,
        )

        return inp

    @optimize_atomic_esp_charges.post
    def optimize_atomic_esp_charges(
        self,
        out: JobOutput,
        M: Molecule,
        update_geometry: bool = True # if we want to update our geometry to the optimized coordinates used for esp calculation
    ):
        if pls := out.files[f"esp.esp"]:
            xyz_esp = pls.decode()

            # split up the lines
            xyz_esp_lines = xyz_esp.split('\n')
            # get our xyx block
            xyz_block_lines = [i.strip() for i in xyz_esp_lines[0:2]] + [' '.join(line.split(' ')[:-1]).strip() for line in xyz_esp_lines[2:] if len(line) > 0] # ignore empty lines
            xyz_block = '\n'.join(xyz_block_lines)

            # get esp charges
            esp_charges = [line.split(' ')[-1].strip() for line in xyz_esp_lines[2:-1]]
            esp_charges = [float(i) for i in esp_charges]

            # check a few things
            assert len(xyz_block_lines) == M.n_atoms + 2 # first two lines aren't atoms
            assert len(esp_charges) == M.n_atoms

            if update_geometry:
                rtn = Molecule(M, coords = Molecule.loads_xyz(xyz_block).coords) # create a new molecule with the updated coordinates
            else:
                rtn = Molecule(M) # just copy, don't update coords

            # assign esp charges as atomic properties
            for i, atom in enumerate(rtn.atoms):
                atom.attrib['nwchem_esp_charge'] = esp_charges[i]

            return rtn

    

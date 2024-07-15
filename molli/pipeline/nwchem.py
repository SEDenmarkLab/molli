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


NWCHEM_INPUT_TEMPLATE_1 = """
title "{name} ESP Calculation"
memory total {memory} mb
echo
start

geometry units angstroms {noautoz}
{xyz}

end
charge {charge}

driver
    maxiter {maxiter}
end

basis
{basis_input}
end

dft
    xc {functional}
    maxiter {maxiter}
end

task dft {optimize}

esp
    recalculate
    range {range}
    probe {probe}
    spacing {spacing}
end

task esp

"""


class NWChemDriver(DriverBase):
    default_executable = "nwchem"

    # the nwchem .esp file contains BOTH optimized coordinates and esp charges
    @Job(return_files=("esp.esp", "esp.grid")).prep
    def optimize_atomic_esp_charges_m(
        self,
        M: Molecule,
        basis_input: list[str] = ["* library 6-311G*"],
        functional: str = "b3lyp",
        maxiter: int = 100,
        optimize: bool = True,  # optimize geometry before ESP calc?
        noautoz: bool = False,
        charge: int = None,
        range: float = 0.2,  # nwchem esp params
        probe: float = 0.1,  # nwchem esp params
        spacing: float = 0.025,  # nwchem esp params
        espminmax: bool = True,
        **kwargs,
    ) -> Molecule:
        """Calculates charges for each atom, along with ESPmin and ESPmax descriptors
        This function can optionally update coordinates upon DFT calculation. ESPmin/max
        are where the electrostatic potential/charge calculated is at a minimum and maximum
        charge for the entire grid.

        Parameters
        ----------
        M : Molecule
            Molecule to be calculated
        basis_input : list[str], optional
            List of basis set changes to be used, by default ["* library 6-311G*"]
        functional : str, optional
            DFT Functional used, by default "b3lyp"
        maxiter : int, optional
            Maximum number of iterations, by default 100
        optimize_geometry : bool, optional
            Optimize geometry using DFT, by default True
        noautoz : bool, optional
            Disables the use of internal coordinates, by default False
        charge : int, optional
            Override for Molecule's net charge, by default None
        range : float, optional
            Maximum distance in nm between a grid point and any of the atomic centers (nm), by default 0.2
        probe : float, optional
            Radius of probes (nm), by default 0.1
        spacing : float, optional
            Grid spacing for the regularly spaced points (nm), by default 0.025

        Returns
        -------
        Molecule
            Returns Molecule with updated coordinates if requested and new properties
            "nwchem_espmin" and "nwchem_espmax", as well as ESPs assigned for each
            atom as "nwchem_esp_charge
        """

        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        xyz_block = M.dumps_xyz(write_header=False)

        noautoz = "noautoz" if noautoz else ""
        optimize = "optimize" if optimize else ""

        _inp = NWCHEM_INPUT_TEMPLATE_1.format(
            name=M.name,
            memory=self.memory,
            noautoz=noautoz,
            xyz=xyz_block,
            charge=charge or M.charge,
            maxiter=maxiter,
            basis_input="\n".join(basis_input),
            functional=functional,
            optimize=optimize,
            range=range,
            probe=probe,
            spacing=spacing,
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

    @optimize_atomic_esp_charges_m.post
    def optimize_atomic_esp_charges_m(
        self,
        out: JobOutput,
        M: Molecule,
        espminmax: bool = True,
        optimize: bool = True,  # if we want to update our geometry to the optimized coordinates used for esp calculation
        **kwargs,
    ):

        if res := out.files.get(f"esp.esp", None):
            xyz_esp = res.decode()

            # split up the lines
            xyz_esp_lines = xyz_esp.split("\n")
            # get our xyx block
            xyz_block_lines = [i.strip() for i in xyz_esp_lines[0:2]] + [
                " ".join(line.split(" ")[:-1]).strip()
                for line in xyz_esp_lines[2:]
                if len(line) > 0
            ]  # ignore empty lines
            xyz_block = "\n".join(xyz_block_lines)

            if optimize:
                rtn = Molecule(
                    M, coords=Molecule.loads_xyz(xyz_block).coords
                )  # create a new molecule with the updated coordinates
            else:
                rtn = Molecule(M)  # just copy, don't update coords

            # get esp charges
            esp_charges = [line.split(" ")[-1].strip() for line in xyz_esp_lines[2:-1]]
            esp_charges = [float(i) * 2625.5 for i in esp_charges]  # kJ

            # check a few things
            assert len(xyz_block_lines) == M.n_atoms + 2  # first two lines aren't atoms
            assert len(esp_charges) == M.n_atoms

            # assign esp charges as atomic properties
            for i, atom in enumerate(rtn.atoms):
                atom.attrib["nwchem_esp_charge"] = esp_charges[i]

        # ESPMin/Max Calculation
        if res := out.files.get("esp.grid", None):
            grid = res.decode()

            gc = np.loadtxt(grid.splitlines(), skiprows=1, usecols=(3)) * 2625.5  # kJ

        if espminmax:
            # assign ESPmin and ESPmax
            rtn.attrib["nwchem_espmin"] = np.min(gc)
            rtn.attrib["nwchem_espmax"] = np.max(gc)

            return rtn

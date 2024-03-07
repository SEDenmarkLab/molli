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
    @Job(return_files=("esp.esp",)).prep
    def optimize_atomic_esp_charges_m(
        self,
        M: Molecule,
        basis: str = "def2-svp",
        functional: str = "b3lyp",
        maxiter: int = 100,
        optimize_geometry: bool = True,  # optimize geometry before ESP calc?
        range: float = 0.2,  # nwchem esp params
        probe: float = 0.1,  # nwchem esp params
        spacing: float = 0.025,  # nwchem esp params
    ) -> Molecule:
        """Calculates charges for each atom, along with ESPmin and ESPmax descriptors
        This function can optionally update coordinates upon DFT calculation. ESPmin/max
        are where the electrostatic potential/charge calculated is at a minimum and maximum
        charge for the entire grid.

        Parameters
        ----------
        M : Molecule
            Molecule to be calculated
        basis : str, optional
            Basis set of functional used, by default "def2-svp"
        functional : str, optional
            DFT Functional used, by default "b3lyp"
        maxiter : int, optional
            Maximum number of iterations, by default 100
        optimize_geometry : bool, optional
            Optimize geometry using DFT, by default True

        Returns
        -------
        Molecule
            Returns Molecule with updated coordinates if requested and new properties
            "nwchem_espmin" and "nwchem_espmax", as well as ESPs assigned for each
            atom as "nwchem_esp_charge
        """

        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        full_xyz = M.dumps_xyz()
        xyz_block = "\n".join(full_xyz.split("\n")[2:])

        _inp = (
            f"""title "{M.name} ESP Calculation"\n"""
            f"""memory total {self.memory} mb\n"""
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
            f"""task dft {'optimize' if optimize_geometry else ''}\n"""  # optimize geometry
            """\n"""
            """esp\n"""
            """    recalculate\n"""
            f"""    range {range}\n"""
            f"""    probe {probe}\n"""
            f"""    spacing {spacing}\n"""
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

    @optimize_atomic_esp_charges_m.post
    def optimize_atomic_esp_charges_m(
        self,
        out: JobOutput,
        M: Molecule,
        update_geometry: bool = True,  # if we want to update our geometry to the optimized coordinates used for esp calculation
    ):
        if res := out.files[f"esp.esp"]:
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

            # get esp charges
            esp_charges = [line.split(" ")[-1].strip() for line in xyz_esp_lines[2:-1]]
            esp_charges = [float(i) * 2625.5 for i in esp_charges]  # kJ

            # check a few things
            assert len(xyz_block_lines) == M.n_atoms + 2  # first two lines aren't atoms
            assert len(esp_charges) == M.n_atoms

            if update_geometry:
                rtn = Molecule(
                    M, coords=Molecule.loads_xyz(xyz_block).coords
                )  # create a new molecule with the updated coordinates
            else:
                rtn = Molecule(M)  # just copy, don't update coords

            # assign esp charges as atomic properties
            for i, atom in enumerate(rtn.atoms):
                atom.attrib["nwchem_esp_charge"] = esp_charges[i]

        # ESPMin/Max Calculation
        if res := out.files[f"esp.grid"]:
            grid = res.decode()

            gc = np.loadtxt(grid.splitlines(), skiprows=1, usecols=(3)) * 2625.5  # kJ

        # assign ESPmin and ESPmax
        rtn.attrib["nwchem_espmin"] = np.min(gc)
        rtn.attrib["nwchem_espmax"] = np.max(gc)

        return rtn

    # the nwchem .esp file contains BOTH optimized coordinates and esp charges
    # the nwchem .grid file contains all grid points and charges at each point
    @Job(return_files=("esp.esp", "esp.grid")).prep
    def calc_espmin_max_m(
        self,
        M: Molecule,
        basis: str = "def2-svp",
        functional: str = "b3lyp",
        maxiter: int = 100,
        optimize_geometry: bool = True,  # optimize geometry before ESP calc?
        range: float = 0.2,  # nwchem esp params
        probe: float = 0.1,  # nwchem esp params
        spacing: float = 0.025,  # nwchem esp params
        **kwargs,
    ) -> Molecule:
        """Calculates ESPmin and ESPmax descriptors and can update coordinates. This
        is where the electrostatic potential/charge calculated is at a minimum and maximum
        charge for the entire grid

        Parameters
        ----------
        M : Molecule
            Molecule to be calculated
        basis : str, optional
            Basis set of functional used, by default "def2-svp"
        functional : str, optional
            DFT Functional used, by default "b3lyp"
        maxiter : int, optional
            Maximum number of iterations, by default 100
        optimize_geometry : bool, optional
            Optimize geometry using DFT, by default True

        Returns
        -------
        Molecule
            Returns Molecule with updated coordinates if requested and new properties
            "nwchem_espmin" and "nwchem_espmax"
        """
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        full_xyz = M.dumps_xyz()
        xyz_block = "\n".join(full_xyz.split("\n")[2:])

        _inp = (
            f"""title "{M.name} ESP Calculation"\n"""
            f"""memory total {self.memory} mb\n"""
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
            f"""task dft {'optimize' if optimize_geometry else ''}\n"""  # optimize geometry
            """\n"""
            """esp\n"""
            """    recalculate\n"""
            f"""    range {range}\n"""
            f"""    probe {probe}\n"""
            f"""    spacing {spacing}\n"""
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

    @calc_espmin_max_m.post
    def calc_espmin_max_m(
        self,
        out: JobOutput,
        M: Molecule,
        update_geometry: bool = True,
        **kwargs,  # if we want to update our geometry to the optimized coordinates used for esp calculation
    ):

        if res := out.files[f"esp.esp"]:
            if update_geometry:
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

                rtn = Molecule(
                    M, coords=Molecule.loads_xyz(xyz_block).coords
                )  # create a new molecule with the updated coordinates
            else:
                rtn = Molecule(M)  # just copy, don't update coords

        # ESPMin/Max Calculation
        if res := out.files[f"esp.grid"]:
            grid = res.decode()

            gc = np.loadtxt(grid.splitlines(), skiprows=1, usecols=(3)) * 2625.5  # kJ

        # assign ESPmin and ESPmax
        rtn.attrib["nwchem_espmin"] = np.min(gc)
        rtn.attrib["nwchem_espmax"] = np.max(gc)

        return rtn

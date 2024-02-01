# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Blake E. Ocampo
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
This file provides necessary functionality to interface with ORCA
"""


from typing import Any, Generator, Callable, Iterable
import molli as ml
from subprocess import run, PIPE
from pathlib import Path
import attrs
import shlex
from tempfile import TemporaryDirectory, mkstemp
import msgpack
from pprint import pprint
from joblib import delayed, Parallel
import numpy as np
from dataclasses import dataclass

from ..chem import Molecule, ConformerEnsemble
from .job import Job, JobInput, JobOutput
from .driver import DriverBase


class Orca_Out_Recognize:
    """
    This builds a quick Orca object that is used with the Orca driver
    """

    def __init__(
        self,
        mlmol: ml.Molecule,
        calc_type: str,
        output_file: str,
        hess_file: str,
        gbw_file: str,
        # nbo_file: str
    ):
        self.mlmol = mlmol
        self.calc_type = calc_type
        self.output_file = output_file
        self.hess_file = hess_file
        self.gbw_file = gbw_file
        # self.nbo_file = nbo_file

        if output_file is not None:
            self.end_line_list = output_file.split("\n")[-11:]
            self.fixed_err = [f"{x}\n" for x in self.end_line_list]
            self.end_lines = "".join(self.fixed_err)

            if any("ORCA TERMINATED NORMALLY" in x for x in self.end_line_list):
                self.orca_failed = False
            else:
                print(
                    f"{self.mlmol.name} has not converged correctly, here are the last few lines:\n{self.end_lines}"
                )
                self.orca_failed = True
        else:
            self.end_line_list = None
            self.fixed_err = None
            self.end_lines = None
            self.orca_failed = None

    def final_xyz(self):
        """
        Will return an xyz block only if the optimization has converged
        """

        lines = self.output_file.split("\n")
        start_idx = end_idx = None

        for idx, line in enumerate(lines):
            if "*** FINAL ENERGY EVALUATION AT THE STATIONARY POINT ***" in line:
                start_idx = idx + 6
            elif "CARTESIAN COORDINATES (A.U.)" in line and start_idx is not None:
                end_idx = idx - 3
                break

        if start_idx and end_idx:
            # Build the XYZ block string
            xyz_block = "\n".join(lines[start_idx : end_idx + 1])
            return f"{len(xyz_block.splitlines())}\n{self.mlmol.name}\n{xyz_block}"
        else:
            print("No Final XYZ detected")
            return None

    def search_freqs(self, num_of_freqs: int):
        """
        Will return a dictionary of number and frequency associated with number (in cm**-1) starting at 6, i.e. {6: 2.82, 7: 16.77 ...} based on the number of frequencies requested
        """
        if self.orca_failed is None:
            print(
                "Orca failed calculation, no vibrational frequencies are registered. Returning None"
            )
            return None

        if "freq" not in self.calc_type:
            print(
                "Current Orca calculation does not contain frequency calculation. Returning None"
            )
            return None
        reversed_all_out_lines = self.output_file.split("\n")[::-1]
        # starts and indexes at the end of the file
        for idx, line in enumerate(reversed_all_out_lines):
            if "VIBRATIONAL FREQUENCIES" == line:
                first_freq = idx - 10
                break
        final_freq = first_freq - num_of_freqs

        freq_requested = reversed_all_out_lines[final_freq:first_freq]
        freq_dict = dict()
        for line in freq_requested[::-1]:
            no_spaces = line.replace(" ", "")
            freq_num = no_spaces.split(":")[0]
            freq_value = float(no_spaces.split(":")[1].split("cm**-1")[0])

            freq_dict.update({freq_num: freq_value})

        return freq_dict

    def nbo_parse(self):
        """
        This is a prototype currently meant to try and parse the NBO population analysis. This returns 4 pieces of data in the following format:

        - orb_homo_lumo = (homo, lumo)

        - nat_charge_dict = {atom number : natural charge}

        """

        if "nbo" not in self.calc_type:
            print(
                "Current Orca calculation does not contain nbo analysis. Returning None"
            )
            return None

        nbo_file_split = self.output_file.split("\n")

        # Find the indices corresponding to unique pieces of the nbo output_file
        for idx, line in enumerate(nbo_file_split):
            if line == "ORBITAL ENERGIES":
                first_orb = idx + 4
                continue
            if line == "Now starting NBO....":
                last_orb = idx - 1
            if line == " Summary of Natural Population Analysis:":
                first_charge = idx + 6
            if line == "                                 Natural Population":
                last_charge = idx - 4
            if (
                line
                == " SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS"
            ):
                first_pert = idx + 8
            if line == " NATURAL BOND ORBITALS (Summary):":
                last_pert = idx - 3
                first_orb_energy = idx + 7
            if line == " $CHOOSE":
                last_orb_energy = idx - 9

        # Orbital Parsing
        orb_list = nbo_file_split[first_orb : last_orb + 1]
        orb_dict = dict()
        for line in orb_list:
            orb_nums, orb_occs, orb_ehs, orb_evs = " ".join(line.split()).split(" ")
            orb_num, orb_occ, orb_ehs, orb_ev = (
                int(orb_nums),
                float(orb_occs),
                float(orb_ehs),
                float(orb_evs),
            )
            orb_dict[orb_num] = orb_ev
            if float(orb_occ) == 0:
                homo = orb_dict[orb_num - 1]
                lumo = orb_dict[orb_num]
                orb_homo_lumo = (homo, lumo)
                break

        # Natural Charge Parsing
        nat_charge_list = nbo_file_split[first_charge : last_charge + 1]
        nat_charge_dict = dict()
        for line in nat_charge_list:
            atom, atom_nums, nat_charges, core, valence, rydberg, tot = " ".join(
                line.split()
            ).split(" ")
            atom_num, nat_charge = int(atom_nums) - 1, float(nat_charges)
            nat_charge_dict[atom_num] = nat_charge

        return orb_homo_lumo, nat_charge_dict


class ORCADriver(DriverBase):
    default_executable = "orca"

    @Job(
        return_files=(
            "m_orca.hess",
            "m_orca.gbw",
        )
    ).prep
    def basic_calc_m(
        self,
        M: Molecule,
        ram_setting: str = "900",
        kohn_sham_type: str = "rks",
        method: str = "b3lyp",
        basis_set: str = "def2-svp",
        calc_type: str = "sp",
        addtl_settings: str = "rijcosx def2/j tightscf nopop miniprint",
    ):
        """Creates an Orca Input File and runs the calculations on a Molecule.
        Here are the updates performed depending on the molecule:

        Any Calculation
        - `orca_success` (Attribute of Molecule as bool)

        Optimization (opt)
        - Updates coordinates of molecule if job is successful

        Frequency (freq)
        - `freq_dict` (Attribute of Molecule as dictionary of first 10 frequencies:
          {'6': float, '7': float, ...})

        NBO (nbo)
        - `nbo_homo_lumo` (Attribute of Molecule as float)
        - `nbo_nat_charge` (Attribute of individual atoms as float)

        Parameters
        ----------
        M : Molecule
            Molecule used for calculation
        ram_setting : str, optional
            Allocation of RAM per core used in MB, by default "900"
        kohn_sham_type : str, optional
            Indication of theory type such as Kohn-Sham or Hartree Fock, by default "rks"
        method : str, optional
            Level of Theory for DFT, by default "b3lyp"
        basis_set : str, optional
            Basis Set used in DFT calculation, by default "def2_svp"
        calc_type : str, optional
            Calculation type such as SP, OPT, OPT FREQ, or OPT FREQ, by default "sp"
        addtl_settings : str, optional
            Other changeable settings, by default "rijcosx def2/j tightscf nopop miniprint"
        """

        xyz_block = M.dumps_xyz(write_header=False)

        _inp = f"""#{str.upper(calc_type)} {M.name}

%maxcore {ram_setting}

%pal nprocs {self.nprocs} end

!{kohn_sham_type} {method} {basis_set} {calc_type} {addtl_settings}

*xyz {M.charge} {M.mult}
{xyz_block}*


"""
        inp = JobInput(
            M.name,
            commands=[(f"""{self.executable} m_orca.inp""", "orca")],
            files={
                "m_orca.inp": _inp.encode(),
            },
            return_files=self.return_files,
        )

        return inp

    @basic_calc_m.post
    def basic_calc_m(self, out: JobOutput, M: Molecule, calc_type: str, **kwargs):
        if _hess := out.files[f"m_orca.hess"]:
            hess = _hess.decode()
        else:
            hess = None
        if _gbw := out.files[f"m_orca.gbw"]:
            gbw = _gbw
        else:
            gbw = None

        orca_obj = Orca_Out_Recognize(
            mlmol=M,
            calc_type=calc_type,
            output_file=out.stdouts["orca"],
            hess_file=hess,
            gbw_file=gbw,
        )

        # Creates a copy of the existing molecule and attributes
        new_M = ml.Molecule(M)

        if orca_obj.orca_failed:
            new_M.attrib["orca_success"] = False
        else:
            new_M.attrib["orca_success"] = True

        if "opt" in calc_type.lower():
            if orca_obj.orca_failed:
                pass
            else:
                new_M.coords = ml.Molecule.loads_xyz(
                    orca_obj.final_xyz(), name="obj"
                ).coords

        if hess:
            new_M.attrib["freq_dict"] = orca_obj.search_freqs(10)

        if "nbo" in calc_type.lower():
            homo_lumo, nat_charge_dict = orca_obj.nbo_parse()
            new_M.attrib["nbo_homo_lumo"] = homo_lumo
            for i, a in enumerate(M.atoms):
                a.attrib["nbo_nat_charge"] = nat_charge_dict[i]

        return new_M

    basic_calc_ens = Job.vectorize(basic_calc_m)

    @basic_calc_ens.reduce
    def basic_calc_ens(
        self, outputs: Iterable[Molecule], ens: ConformerEnsemble, *args, **kwargs
    ):
        """
        General Orca Driver to Create an Orca Input File and run
        calculations on a ConformerEnsemble.
        """
        newens = ConformerEnsemble(ens)
        for i, (new_conf, old_conf) in enumerate(zip(outputs, newens)):
            old_conf.coords = new_conf.coords
            newens.attrib[f"orca_conf_{i}"] = new_conf.attrib
        return newens

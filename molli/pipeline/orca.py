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


from typing import Any, Generator, Callable
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
from .job import Job, JobInput, JobOutput


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

    def search_freqs(self, num_of_freqs: int):
        """
        Will return a dictionary of number and frequency associated with number (in cm**-1) starting at 6, i.e. {6: 2.82, 7: 16.77 ...} based on the number of frequencies requested
        """
        if self.orca_failed is None:
            print(
                "Orca failed calculation, no vibrational frequencies are registered. Returning None"
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
            freq_num = int(no_spaces.split(":")[0])
            freq_value = float(no_spaces.split(":")[1].split("cm**-1")[0])

            freq_dict.update({freq_num: freq_value})

        return freq_dict

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


class ORCADriver:
    def __init__(self, nprocs: int = 1) -> None:
        self.nprocs = nprocs
        self.backup_dir = ml.config.BACKUP_DIR
        self.scratch_dir = ml.config.SCRATCH_DIR
        self.backup_dir.mkdir(exist_ok=True)
        self.scratch_dir.mkdir(exist_ok=True)
        self.cache = Cache(self.backup_dir)
        # self.cache = {}

    def get_cache(self, k):
        # if k not in self.cache:
        #     self.cache[k] = dict()
        # return self.cache[k]
        return self.cache

    @Job(
        return_files=(
            r"*.hess",
            r"*.gbw",
            r"*.out",
        )
    ).prep
    def orca_basic_calc(
        self,
        M: ml.Molecule,
        orca_path: str = "/opt/share/orca/5.0.2/orca",
        ram_setting: str = "900",
        kohn_sham_type="rks",
        method: str = "b3lyp",
        basis_set: str = "def2_tzvp",
        calc_type="sp",
        addtl_settings="rijcosx def2/j tightscf nopop miniprint",
    ):
        """
        General Orca Driver to Create a File and run calculations. Currently usable for the following calculations: "sp","opt","freq", "opt freq".
        This currently cannot recognize different calculation types in a backup directory since the files are built from the "Molecule Object" name.
        Consider doing different calculations in different folders to prevent loading incorrect files.
        """
        # Corrects xyz to be usable in Orca
        full_xyz = M.dumps_xyz()
        xyz_block = "\n".join(full_xyz.split("\n")[2:])[:-3]

        _inp = f"""#{str.upper(calc_type)} {M.name}

%maxcore {ram_setting}

%pal nprocs {self.nprocs} end

!{kohn_sham_type} {method} {basis_set} {calc_type} {addtl_settings}

*xyz {M.charge} {M.mult}
{xyz_block}
*


"""
        inp = JobInput(
            M.name,
            command=f"""{orca_path} {M.name}_{calc_type}.inp""",
            files={f"{M.name}_{calc_type}.inp": _inp.encode()},
        )

        return inp

    @orca_basic_calc.post
    def orca_basic_calc(self, out: JobOutput, M: ml.Molecule, calc_type: str, **kwargs):
        if _hess := out.files[f"{M.name}_{calc_type}.hess"]:
            hess = _hess.decode()
        else:
            hess = None
        if _gbw := out.files[f"{M.name}_{calc_type}.gbw"]:
            gbw = _gbw.decode()
        else:
            gbw = None
        if _out := out.files[f"{M.name}_{calc_type}.out"]:
            out = _out.decode()
        else:
            out = None

        # final_dict = {
        #     'mlmol': M,
        #     'calc_type': calc_type,
        #     'hess': hess,
        #     'gbw': gbw,
        #     'out': out
        # }
        orca_obj = Orca_Out_Recognize(
            mlmol=M,
            calc_type=calc_type,
            output_file=out,
            hess_file=hess,
            gbw_file=gbw,
        )
        return orca_obj


# if __name__ == "__main__":
#     """
#     To set the directories for scratch and backup, for now, use the following commands in the terminal as an example:

#     export MOLLI_SCRATCH_DIR="/home/blakeo2/new_molli/molli_dev/orca_testing/scratch_dir"
#     export MOLLI_BACKUP_DIR="/home/blakeo2/new_molli/molli_dev/orca_testing/backup_dir"

#     Otherwise it defaults based to what is shown in the config.py file (~/.molli/*)

#     """

#     ml.config.configure()

#     print(f"Scratch files writing to: {ml.config.SCRATCH_DIR}")
#     print(f"Backup files writing to: {ml.config.BACKUP_DIR}")

#     mlib = ml.MoleculeLibrary("orca_cinchona_test.mli")

#     orca = ORCADriver(nprocs=32)

#     # Cinchonidine Charges = 1
#     for m in mlib:
#         m.charge = 1

#     # Cinchonidine Spin Multiplicity = 1
#     for m in mlib:
#         m.mult = 1

#     # Note, currently the cache is based on the molecule name
#     res = Parallel(n_jobs=4, verbose=50)(
#         # delayed(crest.conformer_search)(m) for m in ml1_mols
#         delayed(orca.orca_basic_calc)(
#             M=m,
#             orca_path="/opt/share/orca/5.0.2/orca",
#             ram_setting="900",
#             kohn_sham_type="rks",
#             method="vwn5",
#             basis_set="def2-svp",
#             calc_type="sp",
#             addtl_settings="rijcosx def2/j tightscf nopop miniprint",
#         )
#         for m in mlib
#     )

#     # with ml.ConformerLibrary.new(f'./final.cli') as lib:
#     print(res)
#     for orca_obj in res:
#         print(orca_obj.final_xyz())
#         raise ValueError()

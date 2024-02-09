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
from joblib import delayed, Parallel
import numpy as np
from dataclasses import dataclass

from ..chem import Molecule, ConformerEnsemble
from .job import Job, JobInput, JobOutput
from .driver import DriverBase

# class Orca_Out_Recognize:
#     """
#     This builds a quick Orca object that is used with the Orca driver
#     """

#     def __init__(
#         self,
#         mlmol: ml.Molecule,
#         calc_type: str,
#         output_file: str,
#         hess_file: str,
#         gbw_file: str,
#         # nbo_file: str
#     ):
#         self.mlmol = mlmol
#         self.calc_type = calc_type
#         self.output_file = output_file
#         self.hess_file = hess_file
#         self.gbw_file = gbw_file
#         # self.nbo_file = nbo_file

#         if output_file is not None:
#             self.end_line_list = output_file.split("\n")[-11:]
#             self.fixed_err = [f"{x}\n" for x in self.end_line_list]
#             self.end_lines = "".join(self.fixed_err)

#             if any("ORCA TERMINATED NORMALLY" in x for x in self.end_line_list):
#                 self.orca_failed = False
#             else:
#                 print(
#                     f"{self.mlmol.name} has not converged correctly, here are the last few lines:\n{self.end_lines}"
#                 )
#                 self.orca_failed = True
#         else:
#             self.end_line_list = None
#             self.fixed_err = None
#             self.end_lines = None
#             self.orca_failed = None

#     def final_xyz(self):
#         """
#         Will return an xyz block only if the optimization has converged
#         """

#         lines = self.output_file.split("\n")
#         start_idx = end_idx = None

#         for idx, line in enumerate(lines):
#             if "*** FINAL ENERGY EVALUATION AT THE STATIONARY POINT ***" in line:
#                 start_idx = idx + 6
#             elif "CARTESIAN COORDINATES (A.U.)" in line and start_idx is not None:
#                 end_idx = idx - 3
#                 break

#         if start_idx and end_idx:
#             # Build the XYZ block string
#             xyz_block = "\n".join(lines[start_idx : end_idx + 1])
#             return f"{len(xyz_block.splitlines())}\n{self.mlmol.name}\n{xyz_block}"
#         else:
#             print("No Final XYZ detected")
#             return None

#     def search_freqs(self, num_of_freqs: int):
#         """
#         Will return a dictionary of number and frequency associated with number (in cm**-1) starting at 6, i.e. {6: 2.82, 7: 16.77 ...} based on the number of frequencies requested
#         """
#         if self.orca_failed is None:
#             print(
#                 "Orca failed calculation, no vibrational frequencies are registered. Returning None"
#             )
#             return None

#         if "freq" not in self.calc_type:
#             print(
#                 "Current Orca calculation does not contain frequency calculation. Returning None"
#             )
#             return None
#         reversed_all_out_lines = self.output_file.split("\n")[::-1]
#         # starts and indexes at the end of the file
#         for idx, line in enumerate(reversed_all_out_lines):
#             if "VIBRATIONAL FREQUENCIES" == line:
#                 first_freq = idx - 10
#                 break
#         final_freq = first_freq - num_of_freqs

#         freq_requested = reversed_all_out_lines[final_freq:first_freq]
#         freq_dict = dict()
#         for line in freq_requested[::-1]:
#             no_spaces = line.replace(" ", "")
#             freq_num = no_spaces.split(":")[0]
#             freq_value = float(no_spaces.split(":")[1].split("cm**-1")[0])

#             freq_dict.update({freq_num: freq_value})

#         return freq_dict

#     def nbo_parse(self):
#         """
#         This is a prototype currently meant to try and parse the NBO population analysis. This returns 4 pieces of data in the following format:

#         - orb_homo_lumo = (homo, lumo)

#         - nat_charge_dict = {atom number : natural charge}

#         """

#         if "nbo" not in self.calc_type:
#             print(
#                 "Current Orca calculation does not contain nbo analysis. Returning None"
#             )
#             return None

#         nbo_file_split = self.output_file.split("\n")

#         # Find the indices corresponding to unique pieces of the nbo output_file
#         for idx, line in enumerate(nbo_file_split):
#             if line == "ORBITAL ENERGIES":
#                 first_orb = idx + 4
#                 continue
#             if line == "Now starting NBO....":
#                 last_orb = idx - 1
#             if line == " Summary of Natural Population Analysis:":
#                 first_charge = idx + 6
#             if line == "                                 Natural Population":
#                 last_charge = idx - 4
#             if (
#                 line
#                 == " SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS"
#             ):
#                 first_pert = idx + 8
#             if line == " NATURAL BOND ORBITALS (Summary):":
#                 last_pert = idx - 3
#                 first_orb_energy = idx + 7
#             if line == " $CHOOSE":
#                 last_orb_energy = idx - 9

#         # Orbital Parsing
#         orb_list = nbo_file_split[first_orb : last_orb + 1]
#         orb_dict = dict()
#         for line in orb_list:
#             orb_nums, orb_occs, orb_ehs, orb_evs = " ".join(line.split()).split(" ")
#             orb_num, orb_occ, orb_ehs, orb_ev = (
#                 int(orb_nums),
#                 float(orb_occs),
#                 float(orb_ehs),
#                 float(orb_evs),
#             )
#             orb_dict[orb_num] = orb_ev
#             if float(orb_occ) == 0:
#                 homo = orb_dict[orb_num - 1]
#                 lumo = orb_dict[orb_num]
#                 orb_homo_lumo = (homo, lumo)
#                 break

#         # Natural Charge Parsing
#         nat_charge_list = nbo_file_split[first_charge : last_charge + 1]
#         nat_charge_dict = dict()
#         for line in nat_charge_list:
#             atom, atom_nums, nat_charges, core, valence, rydberg, tot = " ".join(
#                 line.split()
#             ).split(" ")
#             atom_num, nat_charge = int(atom_nums) - 1, float(nat_charges)
#             nat_charge_dict[atom_num] = nat_charge

#         return orb_homo_lumo, nat_charge_dict

import pyparsing as pp
from warnings import warn
import numpy as np

dashes = pp.Suppress(pp.Word("-"))

INT = pp.Word(pp.nums).set_parse_action(lambda tks: int(tks[0]))

LNBEG = pp.Suppress(pp.LineStart())
LNEND = pp.Suppress(pp.LineEnd())

dashed_line = LNBEG + dashes + LNEND
delim = LNBEG + pp.Suppress("#") + dashes + LNEND

strkey = pp.Word(pp.alphanums + "_")

header = LNBEG + dashes + pp.Suppress("!") + strkey + pp.Suppress("!") + dashes + LNEND
geom_header = LNBEG + dashes + pp.Suppress("!GEOMETRY!") + dashes + LNEND

h1 = dashed_line + header + dashed_line + delim
h2 = header + delim

prop_block = (
    pp.Group(
        LNBEG
        + pp.Suppress("$")
        + strkey
        + LNEND
        + pp.Suppress("description:")
        + pp.SkipTo(pp.LineEnd())
        + pp.Suppress("geom. index:")
        + pp.SkipTo(pp.LineEnd())
        + pp.Suppress("prop. index:")
        + pp.SkipTo(pp.LineEnd())
        + LNEND
        + pp.SkipTo(delim)
    )
    + delim
)

geom_block = geom_header + pp.SkipTo(geom_header ^ pp.StringEnd())

prop_file_grammar = pp.Dict(
    pp.Group(h1 + pp.OneOrMore(prop_block)) + pp.Group(h2 + pp.OneOrMore(geom_block))
)

fp = pp.Word(pp.nums + "-+.").set_parse_action(lambda t: float(t[0]))

fp3 = pp.Group(fp * 3, aslist=True)

nxyz = pp.Group(pp.OneOrMore(pp.Group(INT + pp.Word(pp.alphas) + fp3)))

geom_block_grammar = (
    pp.Suppress("Number of atoms:")
    + INT
    + pp.Suppress("Geometry Index:")
    + INT
    + pp.Suppress("Coordinates:")
    + nxyz
)

scf_energy_grammar = pp.Suppress("SCF Energy:") + fp
vdw_correction_grammar = pp.Suppress("Van der Waals Correction:") + fp

thermo_key = pp.Word(pp.alphas + "()")[1, ...].setParseAction(" ".join) + pp.FollowedBy(
    ":"
)
thermo_oneliner = LNBEG + thermo_key + pp.Suppress(":") + pp.Word(pp.printables) + LNEND
thermo_multiline = (
    LNBEG + thermo_key + pp.Suppress(":") + LNEND + pp.SkipTo(thermo_oneliner)
)
thermo_grammar = pp.OneOrMore(thermo_multiline ^ thermo_oneliner)


def parse_orca_props(
    props: str,
    prefix="ORCA/",
    keep_full_hess=False,
) -> list[ml.Molecule]:
    """This function is used to parse the <basename>_property.txt file that is
    produced by ORCA calculations. This file is significantly more compact compared to the full output file

    Parameters
    ----------
    fname : str
        name of the file to parse
    prefix : str, optional
        _description_, by default "ORCA/"
    keep_full_hess : bool, optional
        _description_, by default False

    Returns
    -------
    list[ml.Molecule]
        List of molecules that
    """
    parsed = prop_file_grammar.parse_string(props).as_dict()

    mols: list[ml.Molecule] = []

    for i, geom_blk in enumerate(parsed["GEOMETRIES"]):
        n_atoms, geom_idx, nxyz_blk = geom_block_grammar.parse_string(geom_blk)
        n_atoms = int(n_atoms)
        assert int(geom_idx) - 1 == i

        m = ml.Molecule(n_atoms=n_atoms)
        for i, elt, xyz in nxyz_blk:
            m.atoms[i].element = elt
            m.coords[i] = xyz

        mols.append(m)

    for prop, descript, geom_idx, prop_idx, prop_text in parsed["PROPERTIES"]:
        # At this stage geom_idx can be used to retrieve the molecule from the list of mols
        #
        PNAME = prefix + prop
        match prop:
            case "SCF_Energy":
                # This is the KS or HF energy as computed by ORCA
                mols[int(geom_idx) - 1].attrib[PNAME] = scf_energy_grammar.parse_string(
                    prop_text
                )[0]

            case "Mayer_Pop":
                # Mayer population analyses are default
                # Return properties of the atoms and bonds within mols
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

            case "MDCI_Energies":
                # This module is invoved when a coupled cluster calculation is run
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

            case "SCF_Electric_Properties":
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

            case "MDCI_Electric_Properties":
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

            case "DFT_Energy":
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

            case "VdW_Correction":
                mols[int(geom_idx) - 1].attrib[
                    PNAME
                ] = vdw_correction_grammar.parse_string(prop_text)[0]

            case "Hessian":
                if keep_full_hess:
                    warn(
                        f"Current parser does not know what to do with the property {prop!r}."
                        " If this is of interest to you, please inform molli developers or submit a pull request."
                        " This property will be ignored for scraping purposes."
                    )

            case "THERMOCHEMISTRY_Energies":
                allv = thermo_grammar.parse_string(prop_text)
                assert len(allv) % 2 == 0
                data = {}
                while allv:
                    k, v = allv.pop(0).strip(), allv.pop(0).strip()
                    match k:
                        case "Number of frequencies":
                            data[k] = int(v)

                        case "Is Linear":
                            data[k] = bool(v)

                        case "Vibrational frequencies":
                            freqs = [
                                float(l.split()[1])
                                for l in v.splitlines(keepends=False)[1:]
                            ]
                            data[k] = np.array(freqs)

                        case _:
                            data[k] = float(v)

                mols[int(geom_idx) - 1].attrib[PNAME] = data

            case _:
                warn(
                    f"Current parser does not know what to do with the property {prop!r}."
                    " If this is of interest to you, please inform molli developers or submit a pull request."
                    " This property will be ignored for scraping purposes."
                )

    return mols


# This is a default template
ORCA_INPUT_TEMPLATE_1 = """
%pal nprocs {nprocs} end
%maxcore {maxcore}

! {keywords}

{input1}

*xyz {charge} {mult}
{xyz}
*

{input2}

"""


class ORCADriver(DriverBase):
    default_executable = "orca"

    @Job().prep
    def basic_calc_m(
        self,
        M: Molecule,
        keywords: str = "rks b97-3c energy",
        input1: str = None,
        input2: str = None,
        charge: int = None,
        mult: int = None,
        **kwargs,
    ):
        """
        Creates an Orca Input File and runs the calculations on a Molecule.
        Here are the updates performed depending on the molecule:
        TODO: Rewrite the documentation to better reflect the current state of things

        """

        xyz_block = M.dumps_xyz(write_header=False)

        _inp = ORCA_INPUT_TEMPLATE_1.format(
            xyz=xyz_block,
            nprocs=self.nprocs,
            maxcore=self.memory // self.nprocs,
            keywords=keywords,
            input1=input1 or "",
            input2=input2 or "",
            charge=charge or M.charge,
            mult=mult or M.mult,
        )

        inp = JobInput(
            M.name,
            commands=[(f"""{self.executable} m_orca.inp""", "orca")],
            files={
                "m_orca.inp": _inp.encode(),
            },
            return_files=(
                "m_orca.xyz",  # Final optimized geometry
                "m_orca_property.txt",  # Property file that will be the main source of information
            ),
        )

        return inp

    @basic_calc_m.post
    def basic_calc_m(
        self,
        out: JobOutput,
        M: Molecule,
        keywords: str = None,
        charge=None,
        mult=None,
        **kwargs,
    ):
        props = out.files["m_orca_property.txt"].decode()
        M_upd = parse_orca_props(props)[-1]
        new_mol = ml.Molecule(
            M,
            coords=M_upd.coords,
            charge=charge or M_upd.charge,
            mult=mult or M_upd.mult,
        )

        new_mol.attrib |= M_upd.attrib
        new_mol.attrib["ORCA/_keywords"] = keywords

        return new_mol

    basic_calc_ens = Job.vectorize(basic_calc_m)

    # @basic_calc_ens.reduce
    # def basic_calc_ens(
    #     self,
    #     outputs: Iterable[Molecule],
    #     ens: ConformerEnsemble,
    #     *args,
    #     **kwargs,
    # ):
    #     """
    #     General Orca Driver to Create an Orca Input File and run
    #     calculations on a ConformerEnsemble.
    #     """
    #     newens = ConformerEnsemble(ens)
    #     for i, (new_conf, old_conf) in enumerate(zip(outputs, newens)):
    #         old_conf.coords = new_conf.coords
    #         newens.attrib[f"orca_conf_{i}"] = new_conf.attrib
    #     return newens

    optimize_ens = Job.vectorize(basic_calc_m)

    @optimize_ens.reduce
    def optimize_ens(
        self,
        outputs: Iterable[Molecule],
        ens: ConformerEnsemble,
        *args,
        freq_threshold=-10.0,
        rmsd_threshold=0.1,
        **kwargs,
    ):
        from rmsd import rmsd
        import logging

        final_conformers = []

        for i, om in enumerate(outputs):
            freqlist = om.attrib["ORCA/THERMOCHEMISTRY_Energies"][
                "Vibrational frequencies"
            ]

            low_freq = freqlist[5]
            if low_freq >= freq_threshold and len(final_conformers) == 0:
                final_conformers.append(om)
                continue
            else:
                logging.info(
                    f"Conformer {i} of {ens}: low frequency detected: {low_freq:0.2f}"
                )

            # if low_freq >= freq_threshold and all(
            #     rmsd(m.heavy.coords, om.heavy.coords) > rmsd_threshold
            #     for m in final_conformers
            # ):
            #     final_conformers.append(om)
            final_conformers.append(om)

        new_ens = ml.ConformerEnsemble(final_conformers)
        new_ens._atoms = ens.atoms
        new_ens._bonds = ens.bonds

        return new_ens

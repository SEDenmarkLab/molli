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


import re
import shlex
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from pprint import pprint
from subprocess import PIPE, run
from tempfile import TemporaryDirectory, mkstemp
from typing import Any, Callable, Generator, Iterable
from warnings import warn

import attrs
import msgpack
import numpy as np
from joblib import Parallel, delayed

import molli as ml

from ..chem import ConformerEnsemble, Molecule
from .driver import DriverBase
from .job import Job, JobInput, JobOutput

re_delim = re.compile(r"^# -+\s", flags=re.MULTILINE)
re_geom_delim = re.compile(r"^-+ \!GEOMETRY\! -+\s", flags=re.MULTILINE)
re_prop_block = re.compile(
    r"^\$ (?P<name>[a-zA-Z_]+)\s+"
    r"description: (?P<descript>.*)\s+"
    r"geom\. index: (?P<gid>\d+)\s+"
    r"prop\. index: (?P<pid>\d+)$\s"
    r"(?P<content>.*)",
    flags=re.MULTILINE | re.DOTALL,
)

re_dft_energy_line = re.compile(r"\s+([A-Z][a-zA-Z() -]+[a-zA-Z()])\s+(-?[0-9.]+)\s+")
re_thermo_line = re.compile(
    r"^\s+(?P<key>[A-Z][a-zA-Z() -]+[a-zA-Z()])\s+:", re.MULTILINE
)

re_hessian = re.compile(
    r".*?(?P<nr>\d+).*?(?P<nc>\d+)\s+(?P<mtx>.*)", re.MULTILINE | re.DOTALL
)

re_scf_electric = re.compile(
    r"^\*\* (?P<key>[A-Z][a-zA-Z() -]+[a-zA-Z()]) \*\*", re.MULTILINE
)

re_dipole_moment = re.compile(
    r"\s+Magnitude of dipole moment \(Debye\) :\s+(?P<da>-?[0-9.]+).*"
    r"Total Dipole moment:\s(?P<dv>.*)",
    re.MULTILINE | re.DOTALL,
)

re_quadrupole_moment = re.compile(
    r"\s+Isotropic Quadrupole moment :\s+(?P<qa>-?[0-9.]+)\s+.*"
    r"Total quadrupole moment\s(?P<qv>.*[0-9])\s+"
    r"Quadrupole moment diagonaized tensor.*",
    re.MULTILINE | re.DOTALL,
)

re_mdci_energy_line = re.compile(
    r"\s+([a-zA-Z][a-zA-Z() -]+[a-zA-Z()]):?\s+(-?[0-9][0-9.]?+)\s?+"
)

re_mayer_pop = re.compile(
    r".*Number of atoms\s+: (?P<na>[0-9]+)"
    r".*Number of bond orders printed\s+: (?P<nb>[0-9]+)"
    r".*BVA\s+FA\s"
    r"(?P<aprop>.*[0-9])\s+Bond orders larger .*Bond order\s"
    r"(?P<bprop>.*[0-9])\s?+",
    re.MULTILINE | re.DOTALL,
)

re_eprnmr_nucleus = re.compile(
    r"\s+Nucleus: ([0-9]+) .*?P\(iso\)\s+(-?[0-9.]+)",
    re.MULTILINE | re.DOTALL,
)


def split_props(props_txt: str):
    groups = re_delim.split(props_txt)
    geometries = re_geom_delim.split(groups[-1])[1:]
    properties = [re_prop_block.match(blk) for blk in groups[1:-2]]
    return geometries, properties


def parse_matrix(matrix_txt: str, n_rows: int, dtype: str = None):
    """Parse matrix in ORCA format"""
    mtx_io = StringIO(matrix_txt)

    matrices = []

    while True:  # Outer loop over columns
        hdr_line = next(mtx_io, None)

        if hdr_line == None:
            break

        col_idx = list(map(int, hdr_line.split()))

        rows = []
        for i in range(n_rows):
            row_idx, *row = next(mtx_io).split()
            assert i == int(row_idx)
            row = np.fromiter(map(float, row), dtype=dtype or np.float64)
            rows.append(row)

        matrices.append(np.array(rows))

    if len(matrices) == 1:
        if matrices[0].shape[1] == 1:
            return np.concatenate(matrices[0])
        else:
            return matrices[0]
    else:
        return np.concatenate(matrices)


orca_mayer_prop_names = ["NA", "ZA", "QA", "VA", "BVA", "FA"]


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

    geoms, props = split_props(props)

    mols: list[ml.Molecule] = []

    for i, geom_blk in enumerate(geoms):
        geom_lines = geom_blk.splitlines()
        n_atoms = int(geom_lines[0].split(":")[1])
        mol = ml.Molecule(n_atoms=n_atoms)
        for i, ln in enumerate(geom_lines[3:]):
            (_, elt, x, y, z) = ln.split()
            mol.atoms[i].element = elt
            mol.coords[i] = [float(x), float(y), float(z)]
        mols.append(mol)

    for prop_match in props:
        # At this stage geom_idx can be used to retrieve the molecule from the list of mols
        #
        prop = prop_match["name"]
        geom_idx = prop_match["gid"]
        content = prop_match["content"]
        PNAME = prefix + prop
        match prop:
            case "PAL_Flags":
                pass

            case "SCF_Energy" | "VdW_Correction":
                # This is the KS or HF energy as computed by ORCA
                mols[int(geom_idx) - 1].attrib[PNAME] = float(content.split(":")[1])
            case "Mayer_Pop":
                # Mayer population analyses are default
                # Return properties of the atoms and bonds within mols
                m = re_mayer_pop.match(content)
                n_atoms = int(m["na"])
                atom_prop_block = m["aprop"]
                bond_prop_block = m["bprop"]

                for l in atom_prop_block.splitlines():
                    aidx, _, *aprop = l.split()
                    mol.atoms[int(aidx)].attrib[PNAME] = dict(
                        zip(orca_mayer_prop_names, map(float, aprop))
                    )

                for l in bond_prop_block.splitlines():
                    _a1, _, _a2, _, bo = l.split()
                    b = mol.connect(int(_a1), int(_a2), f_order=float(bo))

            case "MDCI_Energies" | "Calculation_Info" | "Solvation_Details":
                d = {}
                for l in content.splitlines():
                    m = re_mdci_energy_line.match(l)
                    if m is None:
                        raise SyntaxError(f"{l!r}")
                    d[m.group(1)] = (
                        float(m.group(2))
                        if "number" not in m.group(1).lower()
                        else int(m.group(2))
                    )
                mols[int(geom_idx) - 1].attrib[PNAME] = d

            case "SCF_Electric_Properties":
                blocks = re_scf_electric.split(content)
                it = iter(blocks[1:])
                d = {}
                for key, val in zip(it, it):
                    if key == "Dipole moment part of electric properties":
                        m = re_dipole_moment.match(val)
                        d["Magnitude of dipole moment (Debye)"] = float(m["da"])
                        d["Total Dipole moment"] = parse_matrix(
                            m["dv"], n_rows=3, dtype=np.float64
                        )
                    elif key == "Quadrupole moment part of electric properties":
                        m = re_quadrupole_moment.match(val)
                        d["Isotropic Quadrupole moment"] = float(m["qa"])
                        d["Total quadrupole moment"] = parse_matrix(
                            m["qv"], n_rows=3, dtype=np.float64
                        )
                    else:
                        d[key] = val

                mols[int(geom_idx) - 1].attrib[PNAME] = d

            case "DFT_Energy":
                d = {}
                for l in content.splitlines():
                    m = re_dft_energy_line.match(l)
                    d[m.group(1)] = float(m.group(2))
                mols[int(geom_idx) - 1].attrib[PNAME] = d

            case "Hessian":
                if keep_full_hess:
                    m = re_hessian.match(content)
                    n_rows = int(m["nr"])
                    hessm = parse_matrix(m["mtx"], n_rows=n_rows, dtype=np.float64)
                    mols[int(geom_idx) - 1].attrib[PNAME] = hessm

            case "THERMOCHEMISTRY_Energies":
                items = re_thermo_line.split(content)[1:]
                it = iter(items)
                d = {}
                for key, val in zip(it, it):
                    _v = val.strip()
                    if key.strip() == "Is Linear":
                        d[key] = _v.lower() == "true"
                    elif key.strip() == "Number of frequencies":
                        d[key] = int(_v)
                    elif key.strip() == "Vibrational frequencies":
                        d[key] = parse_matrix(
                            _v, n_rows=d["Number of frequencies"], dtype=np.float64
                        )
                    else:
                        d[key] = float(_v)

                mols[int(geom_idx) - 1].attrib[PNAME] = d

            case "EPRNMR_OrbitalShielding":
                for _a, _shift in re_eprnmr_nucleus.findall(content):
                    mol.atoms[int(_a)].attrib[PNAME] = {"P(iso)": float(_shift)}

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
        orca_suffix=None,
        **kwargs,
    ):
        """Creates an Orca Input File and runs the calculations on a Molecule.
        Here are the updates performed depending on the molecule:
        TODO: Rewrite the documentation to better reflect the current state of things

        Parameters
        ----------
        M : Molecule
            Molecule object to run the calculations on
        keywords : str, optional
            This is the orca !-delimited block with keywords, by default "rks b97-3c energy"
        input1 : str, optional
            This is the orca %-delimited block that goes ABOVE the *xyz block (most of them follow into this category), by default None
        input2 : str, optional
            This is the orca %-delimited block that goes BELOW the *xyz block (notable example is the %eprnmr block), by default None
        charge : int, optional
            Override for Molecule's net charge, by default None
        mult : int, optional
            Override for Molecule's net multiplicity, by default None
        orca_suffix : _type_, optional
            These are the parameters that go after `/path/to/orca input.inp` line. This is how ORCA sets MPI variables., by default None

        Returns
        -------
        JobInput
            The
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
            commands=[
                (
                    f"""{self.executable} m_orca.inp {('"' + orca_suffix + '"') if orca_suffix is not None else ''}""",
                    "orca",
                )
            ],
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
        freq_threshold=None,
        rmsd_threshold=0.1,
        **kwargs,
    ):
        import logging

        from rmsd import rmsd

        final_conformers = []

        for i, om in enumerate(outputs):
            if freq_threshold is not None:
                if "ORCA/THERMOCHEMISTRY_Energies" not in om.attrib:
                    continue
                else:
                    freqlist = om.attrib["ORCA/THERMOCHEMISTRY_Energies"][
                        "Vibrational frequencies"
                    ]

                    if freqlist[5] < freq_threshold:
                        continue

            if all(
                rmsd(m.coords, om.coords) > rmsd_threshold for m in final_conformers
            ):
                final_conformers.append(om)

        new_ens = ml.ConformerEnsemble(final_conformers)
        new_ens._atoms = ens.atoms
        new_ens._bonds = ens.bonds
        new_ens.attrib["conformer_properties"] = [c.attrib for c in final_conformers]

        return new_ens

    @Job().prep
    def giao_nmr_m(
        self,
        M: Molecule,
        keywords: str = "rks pbe0 pcSseg-2 verytightscf cpcm(chloroform)",
        elements: Iterable[ml.Element] = (ml.Element.C,),
        charge: int = None,
        mult: int = None,
        orca_suffix=None,
        **kwargs,
    ):
        """
        Performs a GIAO-NMR calculation with ORCA
        """

        inner = "\n".join(
            f"  Nuclei = all {ml.Element.get(e)!s} {{ shift }};" for e in elements
        )

        eprnmr_block = f"%eprnmr " + inner + "\nend"

        xyz_block = M.dumps_xyz(write_header=False)

        _inp = ORCA_INPUT_TEMPLATE_1.format(
            xyz=xyz_block,
            nprocs=self.nprocs,
            maxcore=self.memory // self.nprocs,
            keywords=keywords,
            input1="",
            input2=eprnmr_block,
            charge=charge or M.charge,
            mult=mult or M.mult,
        )

        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} m_orca.inp {('"' + orca_suffix + '"') if orca_suffix is not None else ''}""",
                    "orca",
                )
            ],
            files={
                "m_orca.inp": _inp.encode(),
            },
            return_files=(
                "m_orca.xyz",  # Final optimized geometry
                "m_orca_property.txt",  # Property file that will be the main source of information
            ),
        )

        return inp

    @giao_nmr_m.post
    def giao_nmr_m(
        self,
        out: JobOutput,
        M: Molecule,
        keywords: str = "rks pbe0 pcSseg-2 verytightscf cpcm(chloroform)",
        elements: Iterable[ml.Element] = (ml.Element.C,),
        charge: int = None,
        mult: int = None,
        **kwargs,
    ):
        """
        If atoms are supplied as int or str (interpeted as label), set of atoms for NMR computation is extended with *exact* atoms.
        If Element is supplied, it is interpreted as **ALL** elements of the sort should be included in the computation
        """

        props = out.files["m_orca_property.txt"].decode()
        M_upd = parse_orca_props(props)[-1]
        M_upd.attrib["ORCA/_keywords"] = keywords

        return M_upd

    giao_nmr_ens = Job.vectorize(giao_nmr_m)

    @giao_nmr_ens.reduce
    def giao_nmr_ens(
        self,
        outputs: Iterable[Molecule],
        ens: ConformerEnsemble,
        elements: Iterable[ml.Element] = (ml.Element.C,),
        **kwargs,
    ):
        new_ens = ConformerEnsemble(ens)

        mols = list(outputs)

        for i in range(ens.n_atoms):
            if "ORCA/EPRNMR_OrbitalShielding" not in mols[0].atoms[i].attrib:
                continue

            nmr_shieldings = np.array(
                [
                    m.atoms[i].attrib["ORCA/EPRNMR_OrbitalShielding"]["P(iso)"]
                    for m in mols
                ]
            )
            new_ens.atoms[i].attrib["NMR_shielding"] = nmr_shieldings

        new_ens.attrib["conformer_properties"] = [c.attrib for c in mols]

        return new_ens

    @Job().prep
    def scan_dihedral(
        self,
        M: Molecule,
        dihedral_atoms: tuple[ml.AtomLike],
        keywords: str = "rks b97-3c looseopt miniprint noprintmos",
        n_steps: int = 36,
        charge: int = None,
        mult: int = None,
        orca_suffix: str = None,
        **kwargs,
    ):
        """Scan the Relaxed PES corresponding to the rotation around the specified dihedral angle.

        Parameters
        ----------
        M : Molecule
            Molecule object to compute
        dihedral_atoms : tuple[ml.AtomLike]
            tuple of for AtomLike instances. Middle atoms represent the axis for dihedral angle rotation
        keywords : str, optional
            ORCA keywords line that will be used for energy evaluation and optimizatiln, by default "rks b97-3c looseopt miniprint noprintmos"
        n_steps : int, optional
            Number of steps in which the dihedral PES will be sampled (angle increment: 360 degrees // n_steps), by default 36
        charge : int, optional
            Override for molecule net charge, by default None
        mult : int, optional
            Override for molecule net multiplicity, by default None
        orca_suffix : str, optional
            Arguments , by default None

        Returns
        -------
        _type_
            _description_
        """
        from math import degrees

        d0 = degrees(M.dihedral(*dihedral_atoms))
        d1 = d0 + 360
        idx = " ".join(str(x) for x in M.get_atom_indices(*dihedral_atoms))

        inp1 = f"%geom scan D {idx} = {d0:0.3f}, {d1:0.3f}, {n_steps} end end"

        xyz_block = M.dumps_xyz(write_header=False)

        _inp = ORCA_INPUT_TEMPLATE_1.format(
            xyz=xyz_block,
            nprocs=self.nprocs,
            maxcore=self.memory // self.nprocs,
            keywords=keywords,
            input1=inp1,
            input2="",
            charge=charge or M.charge,
            mult=mult or M.mult,
        )

        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""{self.executable} m_orca.inp {('"' + orca_suffix + '"') if orca_suffix is not None else ''}""",
                    "orca",
                )
            ],
            files={
                "m_orca.inp": _inp.encode(),
            },
            return_files=(
                "m_orca.xyz",  # Final optimized geometry
                "m_orca_property.txt",  # Property file that will be the main source of information
            ),
        )

        return inp

    @scan_dihedral.post
    def scan_dihedral(
        self,
        output: JobOutput,
        M: Molecule,
        dihedral_atoms: tuple[ml.AtomLike],
        keywords: str = "rks b97-3c looseopt miniprint noprintmos",
        n_steps: int = 36,
        charge: int = None,
        mult: int = None,
        **kwargs,
    ):
        ensemble = ConformerEnsemble(M, n_conformers=n_steps)
        conformer_attrib = []
        i = 0
        for m in parse_orca_props(output.files["m_orca_property.txt"].decode()):
            # This is a bit of a messed up way of determining if Orca finished optimization step here
            # We are gonna use it nonetheless
            # ORCA only prints
            if "ORCA/SCF_Electric_Properties" in m.attrib:
                ensemble[i].coords = m.coords
                conformer_attrib.append(m.attrib)
                i += 1
        ensemble.attrib["conformer_attrib"] = conformer_attrib

        return ensemble

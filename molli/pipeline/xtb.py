import os
import re
import shlex
from pathlib import Path
from pprint import pprint
from subprocess import PIPE, run
from tempfile import TemporaryDirectory, mkstemp
from typing import Any, Callable, Generator

import attrs
import msgpack
import numpy as np
from joblib import Parallel, delayed

from ..chem import Molecule
from .job import Job, JobInput, JobOutput


class XTBDriver:
    def __init__(self, executable="xtb", nprocs: int = 1) -> None:
        self.executable = executable
        self.nprocs = nprocs

    @Job(return_files=("xtbopt.xyz",)).prep
    def optimize(
        self,
        M: Molecule,
        method: str = "gff",
        crit: str = "loose",
        xtbinp: str = "",
        maxiter: int = 500,
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"
        inp = JobInput(
            M.name,
            commands=[
                (
                    f"""xtb input.xyz --{method} --opt {crit} --charge {M.charge} --iterations {maxiter} {"--input param.inp" if xtbinp else ""} -P {self.nprocs}""",
                    "xtb",
                ),
            ],
            files={"input.xyz": M.dumps_xyz().encode()},
            return_files=self.return_files,
        )

        return inp

    @optimize.post
    def optimize(self, out: JobOutput, M: Molecule, **kwargs):
        if pls := out.files["xtbopt.xyz"]:
            xyz = pls.decode()

            # the second line of the xtb output is not needed - it is the energy line
            xyz_coords = (
                xyz.split("\n")[0] + "\n" + "\n" + "\n".join(xyz.split("\n")[2:])
            )
            optimized = Molecule(M, coords=Molecule.loads_xyz(xyz_coords).coords)

            return optimized

    @Job().prep
    def energy(
        self,
        M: Molecule,
        method: str = "gfn2",
        accuracy: float = 0.5,
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --charge {M.charge} --acc {accuracy:0.2f}""",
            files={"input.xyz": M.dumps_xyz().encode()},
        )

        return inp

    @energy.post
    def energy(self, out: JobOutput, M: Molecule, **kwargs):
        if pls := out.stdout:
            for l in pls.split("\n")[::-1]:
                if m := re.match(
                    r"\s+\|\s+TOTAL ENERGY\s+(?P<eh>[0-9.-]+)\s+Eh\s+\|.*", l
                ):
                    return float(m["eh"])

    @Job(return_files=()).prep
    def atom_properties(
        self,
        M: Molecule,
        method: str = "gfn2",
        accuracy: float = 0.5,
    ):
        assert isinstance(M, Molecule), "User did not pass a Molecule object!"

        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --charge {M.charge} --acc {accuracy:0.2f} --vfukui""",
            files={"input.xyz": M.dumps_xyz().encode()},
        )

        return inp

    @atom_properties.post
    def atom_properties(self, out: JobOutput, M: Molecule, **kwargs):
        from molli.parsing.xtbout import extract_xtb_atomic_properties

        if pls := out.stdout:
            # print(pls)

            outdf = extract_xtb_atomic_properties(pls)
            for i, a in enumerate(M.atoms):
                for j, property in enumerate(outdf.columns):
                    a.attrib[property] = outdf.iloc[i, j]
            return M

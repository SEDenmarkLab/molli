from typing import Any, Generator, Callable
import molli as ml
from subprocess import run, PIPE
from pathlib import Path
import attrs
import shlex
from tempfile import TemporaryDirectory, mkstemp
import msgpack
from pprint import pprint
from joblib import delayed,Parallel
import numpy as np
import re
import os

from molli.parsing import extract_xtb_atomic_properties

# exit()

@attrs.define(repr=True)
class JobOutput:
    stdout: str = None
    stderr: str = None
    exitcode: int = None
    files: dict[str, bytes] = None

@attrs.define(repr=True)
class JobInput:
    jid: str  # job identifier
    command: str
    files: dict[str, bytes] = None

class Job:
    """
    A convenient way to wrap a call to an external program
    """

    def __init__(
        self,
        prep: Callable[[Any], JobInput] = None,
        *,
        post: Callable[[JobOutput], Any] = None,
        envars: dict = None,
        return_files: list[str] = None,
        cache_key: str = None,
    ):
        """If no post"""
        self._envars = envars
        self._return_files = return_files
        self._prep = prep
        self._post = post
        self._cache_key = cache_key


    def __get__(self, instance, owner=None):
        self._instance = instance
        self._owner = owner
        if hasattr(self._instance, "get_cache"):
            self.cache = self._instance.get_cache(self._cache_key)
        else:
            self.cache = None
        return self

    def prep(self, func):
        self._prep = func
        self._cache_key = self._cache_key or func.__name__
        return self

    def post(self, func):
        self._post = func
        return self

    def __call__(self, *args, **kwargs):
        inp = self._prep(self._instance, *args, **kwargs)
        if self.cache and inp.jid in self.cache:
            print("Cached result found. yum!")
            out = self.cache[inp.jid]
        else:
            ### CALCULATION HAPPENS HERE ###
            with TemporaryDirectory(dir=self._instance.scratch_dir, prefix=f"molli-{inp.jid}") as td:
                # Prepping the ground
                for f in inp.files:
                    (Path(td) / f).write_bytes(inp.files[f])

                proc = run(
                    shlex.split(inp.command),
                    cwd=td,
                    stdout=PIPE,
                    stderr=PIPE,
                    encoding="utf8",
                )
                out_files = {}
                if self._return_files:
                    for f in self._return_files:
                        file_path = Path(td) / f
                        out_files[f] = file_path.read_bytes() if file_path.exists() else None
                

                out = JobOutput(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exitcode=proc.returncode,
                    files=out_files,
                )

            ### ==== ###
            if self.cache is not None and out.exitcode == 0:
                self.cache[inp.jid] = out

        result = self._post(self._instance, out, *args, **kwargs)

        return result

class Cache:
    def __init__(self, path):
        self.path = path
    
    def __setitem__(self, __name: str, __value: Any) -> None:
        import msgpack as mp
        from attrs import asdict
        with open(self.path / f"{__name}.dat", "wb") as f:
            msgpack.dump(asdict(__value), f)
    
    def __getitem__(self, __name: str) -> Any:
        import msgpack as mp
        if (pth := self.path / f"{__name}.dat").exists():
            with open(pth, "rb") as f:
                return JobOutput(**msgpack.load(f))
        return None

    def __contains__(self, key):
        return (self.path / f"{key}.dat").exists()
    
class XTBDriver:
    def __init__(self, nprocs: int = 1) -> None:
        self.nprocs = nprocs
        self.backup_dir = ml.config.BACKUP_DIR
        self.scratch_dir = ml.config.SCRATCH_DIR
        self.backup_dir.mkdir(exist_ok=True)
        self.scratch_dir.mkdir(exist_ok=True)
        self.cache = Cache(self.backup_dir)

    def get_cache(self, k):
        return self.cache

    @Job(return_files=("xtbopt.xyz",)).prep
    def optimize(
        self, 
        M: ml.Molecule,
        method: str = "gff",
        crit: str = "normal",
        xtbinp: str = "",
        maxiter: int = 50,
        # xyz_name: str = "mol", # do we need this anymore?

        ):
        
        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --opt {crit} --charge {M.charge} {"--input param.inp" if xtbinp else ""} -P {self.nprocs}""",
            files={"input.xyz": M.dumps_xyz().encode()}
        )
        
        return inp

    @optimize.post
    def optimize(self, out: JobOutput, M: ml.Molecule, **kwargs):

            if (pls := out.files['xtbopt.xyz']):
                xyz = pls.decode()

                # the second line of the xtb output is not needed - it is the energy line
                xyz_coords = xyz.split('\n')[0] +'\n' + '\n'+ '\n'.join(xyz.split('\n')[2:]) 
                optimized = ml.Molecule(M, coords = ml.Molecule.loads_xyz(xyz_coords).coords)
                
                return optimized
            
    @Job().prep
    def energy(
        self, 
        M: ml.Molecule,
        method: str = "gfn2",
        accuracy: float = 0.5,
        ):
        
        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --charge {M.charge} --acc {accuracy:0.2f}""",
            files={"input.xyz": M.dumps_xyz().encode()}
        )
        
        return inp
    
    @energy.post
    def energy(self, out: JobOutput, M: ml.Molecule, **kwargs):

            if (pls := out.stdout):

                for l in pls.split("\n")[::-1]:
                    if m := re.match(r"\s+\|\s+TOTAL ENERGY\s+(?P<eh>[0-9.-]+)\s+Eh\s+\|.*", l):
                        return float(m["eh"])
                    
    @Job(return_files=()).prep
    def atom_properties(
        self, 
        M: ml.Molecule,
        method: str = "gfn2",
        accuracy: float = 0.5,
        ):
        
        inp = JobInput(
            M.name,
            command=f"""xtb input.xyz --{method} --charge {M.charge} --acc {accuracy:0.2f} --vfukui""",
            files={"input.xyz": M.dumps_xyz().encode()}
        )
        
        return inp
    
    @atom_properties.post
    def atom_properties(self, out: JobOutput, M: ml.Molecule, **kwargs):
            assert isinstance(M, ml.Molecule), "User did not pass a Molecule object!"
            # print(out.stderr)

            if (pls := out.stdout):
                # print(pls)

                outdf = extract_xtb_atomic_properties(pls)
                for i, a in enumerate(M.atoms):
                    for j, property in enumerate(outdf.columns):
                        a.attrib[property] = outdf.iloc[i, j]
                return M


if __name__ == "__main__":

    '''
    To set the directories for scratch and backup, for now, use the following commands in the terminal as an example:

    export MOLLI_SCRATCH_DIR="/home/colen2/xtb_test/scratch_dir"
    export MOLLI_BACKUP_DIR="/home/colen2/xtb_test/backup_dir"

    Otherwise it defaults based to what is shown in the config.py file (~/.molli/*)

    '''

    ml.config.configure()

    print(f'Scratch files writing to: {ml.config.SCRATCH_DIR}')
    print(f'Backup files writing to: {ml.config.BACKUP_DIR}')

    # exit()
    

    mlib = ml.MoleculeLibrary('cinchona_base.mli')

    print(len(mlib))

    

    xtb = XTBDriver(nprocs=4)

    #Cinchonidine Charges = 1
    for m in mlib:
        m.charge = 1

    
    ######################################
    # testing geom optimization
    ######################################


    # #Note, currently the cache is based on the molecule name
    # res = Parallel(n_jobs=32, verbose=50)(
    #     # delayed(crest.conformer_search)(m) for m in ml1_mols
    #     delayed(xtb.optimize)(
    #         M=m, 
    #         method="gff",
    #         ) for m in mlib)
    
    # print(res)

    # exit()
    # with ml.MoleculeLibrary.new(f'./final.cli') as lib:
    #     for mol in res:
    #         if isinstance(mol, ml.Molecule):
    #             lib.append(mol.name, mol)


    ######################################
    # testing energies
    ######################################
    # res = Parallel(n_jobs=32, verbose=50)(
    #     # delayed(crest.conformer_search)(m) for m in ml1_mols
    #     delayed(xtb.energy)(
    #         M=m, 
    #         method="gfn2",
    #         ) for m in mlib)

    # print(res)
    # exit()

    ######################################
    # testing atom properties
    ######################################
    res = Parallel(n_jobs=32, verbose=50)(
        # delayed(crest.conformer_search)(m) for m in ml1_mols
        delayed(xtb.atom_properties)(
            M=m, 
            method="gfn2",
            ) for m in mlib)

    # print(res[0])
    for atom in res[0].atoms:
        print(atom.attrib)
    # print(res[0].columns)
    # for i in enumerate(res[0]):
    #     print(i)
    # with open('test.csv', 'w') as w:
    #     res[0].to_csv(w)
    # # exit()
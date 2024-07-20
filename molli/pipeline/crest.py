from .driver import DriverBase
from .job import Job, JobInput, JobOutput
import molli as ml


class CrestDriver(DriverBase):
    default_executable = "crest"

    @Job().prep
    def conformer_search(
        self,
        mol: ml.Molecule,
        charge: int = None,
        mult: int = None,
        method: str = None,
        temp: float = None,
        ewin: float = None,
        mdlen: float = None,
        mddump: float = None,
        vbdump: float = None,
        chk_topo: bool = None,
        misc: str = None,
    ):
        #  -{method} -ewin {ewin:0.4f} -mdlen {mdlen:0.4f} -mddump {mddump:0.4f} -vbdump {vbdump:0.4f} -T {self.nprocs}"""
        cmd = f"""{self.executable} input.xyz -T {self.nprocs}"""

        if method:
            cmd += f" -{method}"

        if charge is None:
            charge = mol.charge

        if mult is None:
            mult = mol.mult

        cmd += f" -chrg {charge} -uhf {mult - 1}"

        for var in "ewin", "mdlen", "mddump", "vbdump", "temp":
            if (val := locals()[var]) is not None:
                cmd += f" -{var} {val:0.4f}"

        if not chk_topo:
            cmd += " --noreftopo"

        if misc is not None:
            cmd += f" {misc}"

        return JobInput(
            mol.name,
            commands=[(cmd, "crest")],
            files={"input.xyz": mol.dumps_xyz().encode()},
            return_files=["crest_conformers.xyz"],
        )

    @conformer_search.post
    def conformer_search(self, output: JobOutput, mol: ml.Molecule, *args, **kwargs):
        all_geoms = ml.CartesianGeometry.loads_all_xyz(
            output.files["crest_conformers.xyz"].decode()
        )

        result = ml.ConformerEnsemble(mol, n_conformers=len(all_geoms))

        for blank_conf, conf_geom in zip(result, all_geoms):
            blank_conf.coords = conf_geom.coords

        return result

    @Job().prep
    def conformer_screen(
        self,
        ens: ml.ConformerEnsemble,
        charge: int = None,
        mult: int = None,
        method: str = None,
        temp: float = None,
        ewin: float = None,
        mdlen: float = None,
        mddump: float = None,
        vbdump: float = None,
        chk_topo: bool = None,
        misc: str = None,
    ):
        #  -{method} -ewin {ewin:0.4f} -mdlen {mdlen:0.4f} -mddump {mddump:0.4f} -vbdump {vbdump:0.4f} -T {self.nprocs}"""
        cmd = f"""{self.executable} -screen input.xyz -T {self.nprocs}"""

        if method:
            cmd += f" -{method}"

        if charge is None:
            charge = ens.charge

        if mult is None:
            mult = ens.mult

        cmd += f" -chrg {charge} -uhf {mult - 1}"

        for var in "ewin", "mdlen", "mddump", "vbdump", "temp":
            if (val := locals()[var]) is not None:
                cmd += f" -{var} {val:0.4f}"

        if not chk_topo:
            cmd += " --noreftopo"

        if misc is not None:
            cmd += f" {misc}"

        return JobInput(
            ens.name,
            commands=[(cmd, "crest")],
            files={"input.xyz": ens.dumps_xyz().encode()},
            return_files=["crest_ensemble.xyz"],
        )

    @conformer_screen.post
    def conformer_screen(self, output: JobOutput, mol: ml.Molecule, *args, **kwargs):
        all_geoms = ml.CartesianGeometry.loads_all_xyz(
            output.files["crest_ensemble.xyz"].decode()
        )

        result = ml.ConformerEnsemble(mol, n_conformers=len(all_geoms))

        for blank_conf, conf_geom in zip(result, all_geoms):
            blank_conf.coords = conf_geom.coords

        return result

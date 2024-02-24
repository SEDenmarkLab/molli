from typing import Literal, Type, IO, List
from pathlib import Path
import molli as ml

supported_fmts_molli = {"xyz", "mol2", "cdxml"}

supported_fmts_obabel = {
    "abinit",
    "acesout",
    "acr",
    "adfband",
    "adfdftb",
    "adfout",
    "alc",
    "aoforce",
    "arc",
    "axsf",
    "bgf",
    "box",
    "bs",
    "c09out",
    "c3d1",
    "c3d2",
    "caccrt",
    "can",
    "car",
    "castep",
    "ccc",
    "cdjson",
    "cdx",
    "cdxml",
    "cif",
    "ck",
    "cml",
    "cmlr",
    "cof",
    "CONFIG",
    "CONTCAR",
    "CONTFF",
    "crk2d",
    "crk3d",
    "ct",
    "cub",
    "cube",
    "dallog",
    "dalmol",
    "dat",
    "dmol",
    "dx",
    "ent",
    "exyz",
    "fa",
    "fasta",
    "fch",
    "fchk",
    "fck",
    "feat",
    "fhiaims",
    "fract",
    "fs",
    "fsa",
    "g03",
    "g09",
    "g16",
    "g92",
    "g94",
    "g98",
    "gal",
    "gam",
    "gamess",
    "gamin",
    "gamout",
    "got",
    "gpr",
    "gro",
    "gukin",
    "gukout",
    "gzmat",
    "hin",
    "HISTORY",
    "inchi",
    "inp",
    "ins",
    "jin",
    "jout",
    "log",
    "lpmd",
    "mcdl",
    "mcif",
    "MDFF",
    "mdl",
    "ml2",
    "mmcif",
    "mmd",
    "mmod",
    "mol",
    "mol2",
    "mold",
    "molden",
    "molf",
    "moo",
    "mop",
    "mopcrt",
    "mopin",
    "mopout",
    "mpc",
    "mpo",
    "mpqc",
    "mrv",
    "msi",
    "nwo",
    "orca",
    "out",
    "outmol",
    "output",
    "pc",
    "pcjson",
    "pcm",
    "pdb",
    "pdbqt",
    "png",
    "pos",
    "POSCAR",
    "POSFF",
    "pqr",
    "pqs",
    "prep",
    "pwscf",
    "qcout",
    "res",
    "rsmi",
    "rxn",
    "sd",
    "sdf",
    "siesta",
    "smi",
    "smiles",
    "smy",
    "sy2",
    "t41",
    "tdd",
    "text",
    "therm",
    "tmol",
    "txt",
    "txyz",
    "unixyz",
    "VASP",
    "vmol",
    "wln",
    "xml",
    "xsf",
    "xtc",
    "xyz",
    "yob",
}


def load(
    path: str | Path,
    fmt: str = None,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", "ensemble", None] | Type = "molecule",
    name: str = None,
):
    """This is a universal loader of molecules / ensembles"""

    if otype == "molecule":
        otype = ml.Molecule
    elif otype == "ensemble":
        otype = ml.ConformerEnsemble

    path = Path(path)

    # Default format is deduced from the file suffix.
    if fmt is None:
        fmt = path.suffix[1:]

    match parser.lower():
        case "molli":
            if fmt not in supported_fmts_molli:
                if fmt in supported_fmts_obabel:
                    raise ValueError(
                        "Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        "Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    with open(path, "rt") as f:
                        return otype.load_xyz(f, name=name)

                case "mol2":
                    with open(path, "rt") as f:
                        return otype.load_mol2(f, name=name)

                case "cdxml":
                    cdxf = ml.CDXMLFile(path)
                    return otype(cdxf._parse_fragment(cdxf.xfrags[0], name=name))

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    "Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            return openbabel.load_obmol(path, ext=fmt, connect_perceive=True, cls=otype)

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def loads():
    raise NotImplementedError


def load_all():
    raise NotImplementedError


def loads_all():
    raise NotImplementedError


__all__ = ("load", "loads", "load_all", "loads_all")

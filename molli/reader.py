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
    key: str = None,
    *,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", "ensemble", None] | Type = "molecule",
    name: str = None,
) -> ml.Molecule | ml.ConformerEnsemble:
    """Load a file as a molecule object

    Parameters
    ----------
    path : str | Path
        Path to the file
    key : str, optional
        If a molecular format supports retrieval by a key, use this key to
    fmt : str, optional
        format of the chemical file, by default None
    parser : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Parser of the chemical file, by default "molli"
    otype : Literal[&quot;molecule&quot;, &quot;ensemble&quot;, None] | Type, optional
        Output type, by default "molecule"
    name : str, optional
        Rename the molecule on the fly, by default None

    Returns
    -------
    ml.Molecule | ml.ConformerEnsemble
        Returns an instance of Molecule or ConformerEnsemble, whichever corresponds to the file contents

    Raises
    ------
    ValueError
        If the file format cannot be matched to a parser
    """

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
                        f"Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
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
                    if key is None:
                        return otype(cdxf._parse_fragment(cdxf.xfrags[0], name=name))
                    else:
                        return otype(cdxf[key])

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            if otype is ml.ConformerEnsemble:
                return ml.ConformerEnsemble(
                    openbabel.load_all_obmol(path, ext=fmt, connect_perceive=True)
                )

            else:
                return otype(openbabel.load_obmol(path, ext=fmt, connect_perceive=True))

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def loads(
    data: str,
    fmt: str,
    key: str = None,
    *,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", "ensemble", None] | Type = "molecule",
    name: str = None,
) -> ml.Molecule | ml.ConformerEnsemble:
    """Load a file as a molecule object

    Parameters
    ----------
    data : str
        Molecule data
    key : str, optional
        If a molecular format supports retrieval by a key, use this key to
    fmt : str, optional
        format of the chemical file, by default None
    parser : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Parser of the chemical file, by default "molli"
    otype : Literal[&quot;molecule&quot;, &quot;ensemble&quot;, None] | Type, optional
        Output type, by default "molecule"
    name : str, optional
        Rename the molecule on the fly, by default None

    Returns
    -------
    ml.Molecule | ml.ConformerEnsemble
        Returns an instance of Molecule or ConformerEnsemble, whichever corresponds to the file contents

    Raises
    ------
    ValueError
        If the file format cannot be matched to a parser
    """

    if otype == "molecule":
        otype = ml.Molecule
    elif otype == "ensemble":
        otype = ml.ConformerEnsemble

    match parser.lower():
        case "molli":
            if fmt not in supported_fmts_molli:
                if fmt in supported_fmts_obabel:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    return otype.loads_xyz(data, name=name)

                case "mol2":
                    return otype.loads_mol2(data, name=name)

                case "cdxml":
                    raise NotImplementedError(
                        "At this time cdxml can only be parsed from a file source"
                    )
                    cdxf = ml.CDXMLFile(data)
                    return otype(cdxf._parse_fragment(cdxf.xfrags[0], name=name))

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            if otype is ml.ConformerEnsemble:
                return ml.ConformerEnsemble(
                    openbabel.loads_all_obmol(data, ext=fmt, connect_perceive=True)
                )

            else:
                return otype(
                    openbabel.loads_obmol(data, ext=fmt, connect_perceive=True)
                )

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def load_all(
    path: str | Path,
    fmt: str = None,
    key: str = None,
    *,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", None] | Type = "molecule",
    name: str = None,
) -> list[ml.Molecule]:
    """Load a file as a molecule object

    Parameters
    ----------
    path : str | Path
        Path to the file
    key : str, optional
        If a molecular format supports retrieval by a key, use this key to
    fmt : str, optional
        format of the chemical file, by default None
    parser : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Parser of the chemical file, by default "molli"
    otype : Literal[&quot;molecule&quot;, &quot;ensemble&quot;, None] | Type, optional
        Output type, by default "molecule"
    name : str, optional
        Rename the molecule on the fly, by default None

    Returns
    -------
    ml.Molecule | ml.ConformerEnsemble
        Returns an instance of Molecule or ConformerEnsemble, whichever corresponds to the file contents

    Raises
    ------
    ValueError
        If the file format cannot be matched to a parser
    """

    if otype == "molecule":
        otype = ml.Molecule
    elif otype == "ensemble" or (
        isinstance(otype, type) and issubclass(otype, ml.ConformerEnsemble)
    ):
        raise ValueError(
            "loads_all cannot be used to produce ConformerEnsembles due to ambiguous implementation"
        )

    path = Path(path)

    # Default format is deduced from the file suffix.
    if fmt is None:
        fmt = path.suffix[1:]

    match parser.lower():
        case "molli":
            if fmt not in supported_fmts_molli:
                if fmt in supported_fmts_obabel:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    with open(path, "rt") as f:
                        return otype.load_all_xyz(f, name=name)

                case "mol2":
                    with open(path, "rt") as f:
                        return otype.load_all_mol2(f, name=name)

                case "cdxml":
                    cdxf = ml.CDXMLFile(path)
                    return [
                        otype(cdxf._parse_fragment(fg, name=name)) for fg in cdxf.xfrags
                    ]

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            return list(
                map(
                    otype,
                    openbabel.load_all_obmol(path, ext=fmt, connect_perceive=True),
                )
            )

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def loads_all(
    data: str,
    fmt: str,
    key: str = None,
    *,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", "ensemble", None] | Type = "molecule",
    name: str = None,
) -> list[ml.Molecule]:
    """Load a file as a molecule object

    Parameters
    ----------
    path : str | Path
        Path to the file
    key : str, optional
        If a molecular format supports retrieval by a key, use this key to
    fmt : str, optional
        format of the chemical file, by default None
    parser : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Parser of the chemical file, by default "molli"
    otype : Literal[&quot;molecule&quot;, &quot;ensemble&quot;, None] | Type, optional
        Output type, by default "molecule"
    name : str, optional
        Rename the molecule on the fly, by default None

    Returns
    -------
    ml.Molecule | ml.ConformerEnsemble
        Returns an instance of Molecule or ConformerEnsemble, whichever corresponds to the file contents

    Raises
    ------
    ValueError
        If the file format cannot be matched to a parser
    """

    if otype == "molecule":
        otype = ml.Molecule
    elif otype == "ensemble":
        otype = ml.ConformerEnsemble

    match parser.lower():
        case "molli":
            if fmt not in supported_fmts_molli:
                if fmt in supported_fmts_obabel:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    return otype.loads_xyz(data, name=name)

                case "mol2":
                    return otype.loads_mol2(data, name=name)

                case "cdxml":
                    raise NotImplementedError
                    cdxf = ml.CDXMLFile(path)
                    return otype(cdxf._parse_fragment(cdxf.xfrags[0], name=name))

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            return list(
                map(
                    otype,
                    openbabel.loads_all_obmol(data, ext=fmt, connect_perceive=True),
                )
            )

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


__all__ = ("load", "loads", "load_all", "loads_all")

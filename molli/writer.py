from typing import Literal, Type, IO, List
from pathlib import Path
import molli as ml

supported_fmts_molli = {"xyz", "mol2"}

supported_fmts_obabel = {
    "acesin",
    "adf",
    "alc",
    "ascii",
    "bgf",
    "box",
    "bs",
    "c3d1",
    "c3d2",
    "cac",
    "caccrt",
    "cache",
    "cacint",
    "can",
    "cdjson",
    "cdxml",
    "cht",
    "cif",
    "ck",
    "cml",
    "cmlr",
    "cof",
    "com",
    "confabreport",
    "CONFIG",
    "CONTCAR",
    "CONTFF",
    "copy",
    "crk2d",
    "crk3d",
    "csr",
    "cssr",
    "ct",
    "cub",
    "cube",
    "dalmol",
    "dmol",
    "dx",
    "ent",
    "exyz",
    "fa",
    "fasta",
    "feat",
    "fh",
    "fhiaims",
    "fix",
    "fps",
    "fpt",
    "fract",
    "fs",
    "fsa",
    "gamin",
    "gau",
    "gjc",
    "gjf",
    "gpr",
    "gr96",
    "gro",
    "gukin",
    "gukout",
    "gzmat",
    "hin",
    "inchi",
    "inchikey",
    "inp",
    "jin",
    "k",
    "lmpdat",
    "lpmd",
    "mcdl",
    "mcif",
    "MDFF",
    "mdl",
    "ml2",
    "mmcif",
    "mmd",
    "mmod",
    "mna",
    "mol",
    "mol2",
    "mold",
    "molden",
    "molf",
    "molreport",
    "mop",
    "mopcrt",
    "mopin",
    "mp",
    "mpc",
    "mpd",
    "mpqcin",
    "mrv",
    "msms",
    "nul",
    "nw",
    "orcainp",
    "outmol",
    "paint",
    "pcjson",
    "pcm",
    "pdb",
    "pdbqt",
    "png",
    "pointcloud",
    "POSCAR",
    "POSFF",
    "pov",
    "pqr",
    "pqs",
    "qcin",
    "report",
    "rinchi",
    "rsmi",
    "rxn",
    "sd",
    "sdf",
    "smi",
    "smiles",
    "stl",
    "svg",
    "sy2",
    "tdd",
    "text",
    "therm",
    "tmol",
    "txt",
    "txyz",
    "unixyz",
    "VASP",
    "vmol",
    "xed",
    "xyz",
    "yob",
    "zin",
}


def dump(
    obj: ml.Molecule | ml.ConformerEnsemble,
    path: str | Path,
    fmt: str = None,
    key: str = None,
    *,
    parser: Literal["molli", "openbabel", "obabel"] = "molli",
    otype: Literal["molecule", "ensemble", None] | Type = "molecule",
    name: str = None,
) -> ml.Molecule | ml.ConformerEnsemble:
    """Dump a file as a molecule object

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
                    return otype(cdxf._parse_fragment(cdxf.xfrags[0], name=name))

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            return openbabel.load_obmol(path, ext=fmt, connect_perceive=True, cls=otype)

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def dumps(
    data: str,
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
                        return otype.loads_xyz(f, name=name)

                case "mol2":
                    with open(path, "rt") as f:
                        return otype.loads_mol2(f, name=name)

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

            return openbabel.load_obmol(path, ext=fmt, connect_perceive=True, cls=otype)

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def dump_all(
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
                        "Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        "Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
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
                    "Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel

            return openbabel.loads_all_obmol(
                path, ext=fmt, connect_perceive=True, cls=otype
            )

        case _:
            raise ValueError(f"Molli currently does not support parser {parser!r}")


def dumps_all(
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
                        "Unsupported format {fmt!r} for parser 'molli'. Try parser='openbabel'."
                    )
                else:
                    raise ValueError(
                        "Unsupported format {fmt!r} for parser 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    with open(path, "rt") as f:
                        return otype.loads_all_xyz(f, name=name)

                case "mol2":
                    with open(path, "rt") as f:
                        return otype.loads_all_mol2(f, name=name)

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


__all__ = ("dump", "dumps", "dump_all", "dumps_all")

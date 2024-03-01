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
    io_or_path: str | Path | IO,
    fmt: str = None,
    *,
    key: str = None,
    writer: Literal["molli", "openbabel", "obabel"] = "molli",
    mode: Literal["a", "w"] = "a",
    obflags: str = None,
    **kwargs,
):
    """
    Dump an object into a stream

    A file name can be provided instead (in that case the stream is opened)

    Parameters
    ----------
    obj : ml.Molecule | ml.ConformerEnsemble
        Object to be written
    io_or_path : str | Path | IO
        Stream or file path to write into
    fmt : str, optional
        Format. If file name is given, it can also be automatically guessed from the extension.
    key : str, optional
        Unused in most cases, unless a chemical library supports a separate key, by default None
    writer : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Only molli or openbabel are valid choices for now, by default "molli"
    mode : Literal[&quot;a&quot;, &quot;w&quot;], optional
        If a file name is given, this determines the file open mode, by default "a"
    """
    assert mode in {"a", "w"}

    if isinstance(io_or_path, (str, Path)):
        to_be_closed = True
        stream = open(io_or_path, mode=mode)
        fmt = fmt or Path(io_or_path).suffix[1:]
    else:
        stream = io_or_path

    try:
        match writer.lower():
            case "molli":
                if fmt not in supported_fmts_molli:
                    if fmt in supported_fmts_obabel:
                        raise ValueError(
                            f"Unsupported format {fmt!r} for writer 'molli'. Try writer='openbabel'."
                        )
                    else:
                        raise ValueError(
                            f"Unsupported format {fmt!r} for writer 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                        )

                match fmt:
                    case "xyz":
                        obj.dump_xyz(stream, **kwargs)

                    case "mol2":
                        obj.dump_mol2(stream, **kwargs)

            case "openbabel" | "obabel":
                if fmt not in supported_fmts_obabel:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                    )

                from molli.external import openbabel
                from openbabel import pybel

                for c in (
                    obj if isinstance(obj, ml.ConformerEnsemble) else (obj,)
                ):  # This assumes conformer ensemble
                    txt = pybel.Molecule(openbabel.to_obmol(c)).write(
                        format=fmt,
                        opt=kwargs | {x: None for x in (obflags or "")},
                    )
                    stream.write(txt)

            case _:
                raise ValueError(f"Molli currently does not support parser {writer!r}")
    except:
        raise
    finally:
        if to_be_closed:
            stream.close()


def dumps(
    obj: ml.Molecule | ml.ConformerEnsemble,
    fmt: str,
    *,
    writer: Literal["molli", "openbabel", "obabel"] = "molli",
    obflags: str = None,
    **kwargs,
):
    """
    Returns a string representation of the molecular object

    **NOTE: this function will sequentially convert all conformers of the ensemble
    and dump them sequentially into the resulting string.**


    Parameters
    ----------
    obj : ml.Molecule | ml.ConformerEnsemble
        Object for output purposes
    fmt : str
        Format of the output. E. g. `mol2`, `smi` etc.
    writer : Literal[&quot;molli&quot;, &quot;openbabel&quot;, &quot;obabel&quot;], optional
        Writer, by default "molli"
    obflags : str, optional
        Sequence of flags for openbabel output, by default None

    Returns
    -------
    str
        String representation of the molecule or ensemble

    """

    match writer.lower():
        case "molli":
            if fmt not in supported_fmts_molli:
                if fmt in supported_fmts_obabel:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for writer 'molli'. Try writer='openbabel'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported format {fmt!r} for writer 'molli'. It isn't supported by openbabel either. Things must be grim indeed... "
                    )

            match fmt:
                case "xyz":
                    return obj.dumps_xyz(**kwargs)

                case "mol2":
                    return obj.dumps_mol2(**kwargs)

        case "openbabel" | "obabel":
            if fmt not in supported_fmts_obabel:
                raise ValueError(
                    f"Unsupported format {fmt!r} for parser 'openbabel'. Things must be grim indeed... "
                )

            from molli.external import openbabel
            from openbabel import pybel

            txts = []
            for c in (
                obj if isinstance(obj, ml.ConformerEnsemble) else (obj,)
            ):  # This assumes conformer ensemble
                txts.append(
                    pybel.Molecule(openbabel.to_obmol(c)).write(
                        format=fmt,
                        opt=kwargs | {x: None for x in (obflags or "")},
                    )
                )

            return "".join(txts)

        case _:
            raise ValueError(f"Molli currently does not support parser {writer!r}")


__all__ = (
    "dump",
    "dumps",
)

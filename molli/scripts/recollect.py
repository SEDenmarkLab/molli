"""
Convert molli library formats
"""

from argparse import ArgumentParser
import molli as ml
from tqdm import tqdm
import zipfile
import msgpack
from pathlib import Path

arg_parser = ArgumentParser(
    "molli recollect",
    description="Read old style molli collection and convert it to the new file format.",
)

arg_parser.add_argument(
    "input",
    metavar="XML_FILE",
    action="store",
    type=str,
    help="Old style xml collection to be converted",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="MLI_FILE",
    action="store",
    type=str,
    default=...,
    help="New style collection to be made",
)

arg_parser.add_argument(
    "-cm",
    "--charge_mult",
    metavar=("0", "1"),
    action="store",
    type=int,
    nargs=2,
    default=[0, 1],
    help="Assign these charge and multiplicity to the imported molecules",
)

arg_parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase the amount of output",
)

arg_parser.add_argument(
    "-s",
    "--skip",
    action="store_true",
    default=False,
    help="This option enables skipping malformed files within old collections. Warnings will be printed.",
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    inp = Path(parsed.input)

    if parsed.output is Ellipsis:
        out = inp.with_suffix(".mli")
        with ml.aux.ForeColor("yellow"):
            print(f"Defaulting to {out} as the output destination.")
    else:
        out = Path(parsed.output)

    charge, mult = parsed.charge_mult

    print(f"Charge and multiplicity: {charge} {mult}")
    if parsed.verbose:
        print("Full paths to files:")
        print(" - Input  ", inp.absolute())
        print(" - Output ", out.absolute())

    if parsed.skip:
        with ml.aux.ForeColor("yellow"):
            print(f"Enabled skipping malformed files.")

    try:
        zf = zipfile.ZipFile(inp, "r")
    except:
        print("This does not appear to be a valid zip file")
        exit(2)

    if "__molli__" not in zf.namelist():
        print("This does not appear to be a valid collection")
        exit(3)

    with (
        ml.aux.catch_interrupt(),
        ml.chem.ConformerLibrary.new(out) as lib,
    ):
        tnm = 0
        tnc = 0
        for f in tqdm(zf.filelist, f"Reading data from {inp.name}", dynamic_ncols=True):
            if f.filename != "__molli__":
                try:
                    ens = ml.chem.ensemble_from_molli_old_xml(zf.open(f))
                except SyntaxError:
                    tqdm.write(f"File {f} in source collection cannot be read.")
                    if parsed.skip:
                        continue
                ens.charge = charge
                ens.mult = mult

                tnm += 1
                tnc += ens.n_conformers

                lib.append(ens.name, ens)

    with ml.aux.ForeColor("green"):
        print("Success")

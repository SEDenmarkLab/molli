"""
Convert molli library formats
"""

from argparse import ArgumentParser
import molli as ml
from tqdm import tqdm
import zipfile
import msgpack
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import numpy as np
from ..storage import Collection
from molli.external import openbabel as mob
from functools import partial
import sys

arg_parser = ArgumentParser(
    "molli recollect",
    description="Read old style molli collection and convert it to the new file format.",
)

arg_parser.add_argument(
    "-it",
    "--input_type",
    # metavar="<MLIB_FILE, CLIB_File, ZIP_FILE, OLD_XML_FILE, UKV_FILE, DIRECTORY>",
    choices=["zip","mlib","mli","clib",'cli','dir','directory'],
    action="store",
    type=str.lower,
    help="This is the input type, including <mlib>, <.clib>, <.zip>, <.xml>, <.ukv>, or directory (<dir>)",
)

arg_parser.add_argument(
    "-i",
    "--input",
    metavar="<PATH>",
    action="store",
    type=Path,
    help="This is the input path",
)

arg_parser.add_argument(
    "-ot",
    "--output_type",
    choices=["mlib","clib","dir"],
    action="store",
    type=str,
    default=...,
    help="New style collection, either with or without conformers",
)

arg_parser.add_argument(
    "-o",
    "--output",
    metavar="<PATH>",
    action="store",
    type=Path,
    default=...,
    help="This is the output path",
)

arg_parser.add_argument(
    "-p",
    "--parser",
    choices=["molli","obabel"],
    action="store",
    type=str.lower,
    default='molli',
    help="This indicates the type of parser, will default to molli for xyz and mol2, all other files will default to openbabel",
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

#Currently do not have a skip function in this
# arg_parser.add_argument(
#     "-s",
#     "--skip",
#     action="store_true",
#     default=False,
#     help="This option enables skipping malformed files within old collections. Warnings will be printed.",
# )

arg_parser.add_argument(
    "-ext",
    "--extension",
    action="store",
    default='mol2',
    type=str,
    help="This option is required if reading from a <zip> or directory to indicate the File Type being searched for (<mol2>, <xyz>, etc.)",
)

def recollect(collect:Collection, lib, charge, mult, clib, mlib, dir):
    '''
    Overarching function that recollects for different types of collections:

    This is currently built assuming either "loads_all_xyz" or "loads_all_mol2" was run.
    This can be changed, but I figured we would stick to a list for now and we could change it after

    '''
    with collect.reading(), lib.writing():
        #Currently needs an implementation to show how far it is using tqdm
        for name in collect:
            res_list = collect[name]
            test_mol = res_list[0]
            # print(test_mol)
            if isinstance(test_mol, ml.Molecule):

                if clib:
                    all_coords = np.array([m.coords for m in res_list])
                    ens = ml.ConformerEnsemble(test_mol, coords=all_coords, n_conformers=len(res_list))
                    # ens = ml.ConformerEnsemble(res_list)
                    ens.charge = charge
                    ens.mult = mult
                    lib[test_mol.name] = ml.ConformerEnsemble(res_list)

                elif mlib or dir:
                    for m in res_list:
                        m.charge = charge
                        m.mult = mult
                        lib[m.name] = m             

            else:
                raise NotImplementedError(f'Currently unable to work directly from {type(test_mol)}.')

    sys.exit(f'Finished recollecting')

def recollect_legacy(zf: ZipFile, lib, charge, mult, mlib):
    with lib.writing():
        #Currently needs an implementation to show how far it is using tqdm
        for xml in zf.namelist():
            if xml != '__molli__':
                try:
                    res = ml.chem.ensemble_from_molli_old_xml(zf.open(xml), mol_lib=mlib)
                except SyntaxError:
                    print(f'File {xml} in source collection cannot be read')
                res.charge = charge
                res.mult = mult
                print(res.dumps_mol2())
                print('hello')
                lib[res.name] = res
    
    sys.exit(f'Finished recollecting legacy file')

def molli_main(args,  **kwargs):
    parsed = arg_parser.parse_args(args)

    inp = Path(parsed.input)
    it = parsed.input_type

    if not it:
        it = inp.suffix

    out = Path(parsed.output)
    ot = parsed.output_type

    # if parsed.output is Ellipsis:
    #     out = inp.with_suffix(".mli")
    #     with ml.aux.ForeColor("yellow"):
    #         print(f"Defaulting to {out} as the output destination.")
    # else:
    #     out = Path(parsed.output)

    charge, mult = parsed.charge_mult

    print(f"Charge and multiplicity: {charge} {mult}")
    if parsed.verbose:
        print("Full paths to files:")
        print(" - Input  ", inp.absolute())
        print(" - Output ", out.absolute())

    # if parsed.skip:
    #     with ml.aux.ForeColor("yellow"):
    #         print(f"Enabled skipping malformed files.")

    mlib = False
    clib = False
    dir = False
            
    encoding_method = {}
    decoding_method = {}

    match parsed.parser:
        case 'molli':
            decoding_method[f'xyz'] = ml.Molecule.loads_all_xyz
            decoding_method[f'mol2'] = ml.Molecule.loads_all_mol2
            encoding_method[f'xyz'] = (lambda x: ml.Molecule.dumps_xyz(x).encode())
            encoding_method[f'mol2'] = (lambda x: ml.Molecule.dumps_mol2(x).encode())
        case 'obabel':
            decoding_method[parsed.extension] = partial(mob.loads_obmol, input_ext=parsed.extension, connect_perceive=False, cls = ml.Molecule)
            encoding_method[parsed.extension] = partial(mob.dumps_obmol, ftype=parsed.extension)
        case _:
            raise NotImplementedError(f'{parsed.parser} not an available parser')

    match ot:
        case 'mlib':
            lib = ml.chem.MoleculeLibrary(parsed.output, readonly=False, overwrite=True)
            mlib = True
        case 'clib':
            lib = ml.chem.ConformerLibrary(parsed.output, readonly=False, overwrite=True)
            clib = True
        case 'dir':
            lib = ml.storage.Collection[dict](
                parsed.output,
                ml.storage.DirCollectionBackend,
                ext=f".{parsed.extension}",
                value_encoder=encoding_method[parsed.extension],
                # value_decoder=decoding_method[parsed.extension],
                readonly=False,
                overwrite=True
            )
            dir = True
            # raise NotImplementedError('"dir" not impelemented in current iteration')
        case _:
            raise NotImplementedError(f'{ot} not implemented in current iteration')

    match it:
        case 'zip'|'ZIP_FILE':
            # print('zip')
            if not is_zipfile(inp):
                raise ValueError(f'{inp} is not a valid zipfile!')
            else:
                with ZipFile(inp, mode='r') as zf:
                    #Searches the keys for a file unique to molli 0.2
                    if '__molli__' in zf.NameToInfo:
                        recollect_legacy(zf,lib,charge, mult, mlib)

                    suffixes = {Path(x).suffix for x in zf.namelist()}
                assert len(suffixes) == 1, f'There are not uniform file types in this ZipFile: {suffixes}'

                # if not parsed.extension:
                #     #Finds extension without period
                #     parsed.extension = suffixes.pop()[1:]
                
                zip_col = ml.storage.Collection[dict](
                    inp,
                    ml.storage.ZipCollectionBackend,
                    ext=f'.{parsed.extension}',
                    value_decoder=decoding_method[parsed.extension],
                    readonly=True,
                    overwrite=False
                )
                #Read Zip and Write New Library
                recollect(zip_col, lib, charge=charge, mult=mult, clib=clib, mlib=mlib, dir=dir)

        case 'mlib'|'mli':
            # print('mlib')
            mlib = ml.MoleculeLibrary(inp)
            assert parsed.molecules, f'This should be recollecting into an ml.MoleculeLibrary, This value must be True!'
            
            with mlib.reading(), lib.writing():
                for name in mlib:
                    m = mlib[name]
                    m.charge = charge
                    m.mult = mult
                    lib[name] = m
        
        case 'clib'|'cli':
            # print('clib')
            clib = ml.ConformerLibrary(inp)
            assert parsed.conformers, f'This should be recollecting into an ml.ConformerLibrary, This value must be True!'
            
            with clib.reading(), lib.writing():
                for name in clib:
                    ens = clib[name]
                    ens.charge = charge
                    ens.mult = mult
                    lib[name] = ens

        case 'dir'|'Directory'|'directory':
            # print('dir')
            dir_collect = ml.storage.Collection[dict](
                inp,
                ml.storage.DirCollectionBackend,
                ext=f".{parsed.extension}",
                value_decoder=decoding_method[parsed.extension],
                readonly=True,
            )

            recollect(dir_collect, lib, charge=charge, mult=mult, clib=clib, mlib=mlib, dir=dir)
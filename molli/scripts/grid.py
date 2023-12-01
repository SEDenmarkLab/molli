"""
Create a grid to be subsequently used in grid based descriptor calculations
"""

from argparse import ArgumentParser
import molli as ml
from tqdm import tqdm
import zipfile
import msgpack
import numpy as np
from pathlib import Path

arg_parser = ArgumentParser(
    "molli grid",
    description="Read a molli library and calculate a grid",
)

source = arg_parser.add_mutually_exclusive_group()

source.add_argument(
    "--mlib",
    action="store",
    default=None,
    help="Library file to perform the calculations on",
)

source.add_argument(
    "--mol2_dir",
    action="store",
    default=None,
    help="Directory of multi-conformer mol2 files to perform the calculations on",
)


arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<fpath>",
    default="grid.npy",
    help="Destination for calculation results",
)

arg_parser.add_argument(
    "-f",
    "--format",
    action="store",
    # metavar="fmt",
    choices=("npy",),
    default="npy",
    help="Select the format that will be used for data storage.",
)

arg_parser.add_argument(
    "-p",
    "--padding",
    action="store",
    metavar="0.0",
    default=0.0,
    type=float,
    help="The bounding box will be padded by this many angstroms prior to grid construction",
)

arg_parser.add_argument(
    "-s",
    "--spacing",
    action="store",
    metavar="1.0",
    default=1.0,
    type=float,
    help="Intervals at which the grid points will be placed",
)


def molli_main(args,  **kwargs):
    parsed = arg_parser.parse_args(args)

    if parsed.mlib is not None:
        mlib_path = Path(parsed.mlib)
        if mlib_path.is_dir():
            raise FileNotFoundError(
                "This operation is only meant to work on .mlib files. Received a directory"
            )
        else:
            library = tqdm(
                ml.ConformerLibrary(mlib_path, readonly=True),
                # total=len(names),
                dynamic_ncols=True,
            )
        

    elif parsed.mol2_dir is not None:
        mol2_dir = Path(parsed.mol2_dir).absolute()
        if mol2_dir.is_dir():
            print("Performing calculation on a directory of mol2 files:")
            print(mol2_dir)
        else:
            raise NotADirectoryError(
                "This operation is only meant to work on directories"
            )

        names = list(ml.aux.sglob(str(mol2_dir / "*.mol2"), lambda x: x))
        library = tqdm(
            ml.aux.sglob(str(mol2_dir / "*.mol2"), ml.ConformerEnsemble.load_mol2),
            total=len(names),
            dynamic_ncols=True,
        )


    mins = np.zeros((len(library), 3), dtype=np.float32)
    maxs = np.zeros((len(library), 3), dtype=np.float32)

    for i, ens in enumerate(library):
        mins[i] = np.min(ens._coords, axis=(0, 1))
        maxs[i] = np.max(ens._coords, axis=(0, 1))

    q1, q2 = np.min(mins, axis=0), np.max(maxs, axis=0)
    grid = ml.descriptor.rectangular_grid(q1, q2, parsed.padding, parsed.spacing)
    print(grid.shape)

    np.save(parsed.output, grid, allow_pickle=False)

    # np.save(parsed.output, grid, allow_pickle=False)

    # with ml.chem.ConformerLibrary(parsed.library, readonly=True) as lib:
    #     mins = np.zeros((len(lib), 3), dtype=np.float32)
    #     maxs = np.zeros((len(lib), 3), dtype=np.float32)

    #     for i, ens in enumerate(tqdm(lib, dynamic_ncols=True)):
    #         mins[i] = np.min(ens._coords, axis=(0, 1))
    #         maxs[i] = np.max(ens._coords, axis=(0, 1))

    # q1, q2 = np.min(mins, axis=0), np.max(maxs, axis=0)
    # grid = ml.descriptor.rectangular_grid(q1, q2, parsed.padding, parsed.spacing)
    # print(grid.shape)

    # np.save(parsed.output, grid, allow_pickle=False)

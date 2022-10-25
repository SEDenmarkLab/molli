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

arg_parser.add_argument(
    "library",
    action="store",
    help="Library file to perform the calculations on",
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


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    with ml.chem.ConformerLibrary(parsed.library, readonly=True) as lib:
        mins = np.zeros((len(lib), 3), dtype=np.float32)
        maxs = np.zeros((len(lib), 3), dtype=np.float32)

        for i, ens in enumerate(tqdm(lib, dynamic_ncols=True)):
            mins[i] = np.min(ens._coords, axis=(0, 1))
            maxs[i] = np.max(ens._coords, axis=(0, 1))

    q1, q2 = np.min(mins, axis=0), np.max(maxs, axis=0)
    grid = ml.descriptor.rectangular_grid(q1, q2, parsed.padding, parsed.spacing)
    print(grid.shape)

    np.save(parsed.output, grid, allow_pickle=False)
